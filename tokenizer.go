package main

// HuggingFace tokenizer.json support: byte-level BPE (GPT-2, Llama 3, Qwen, SmolLM), SentencePiece-style
// Metaspace BPE and Unigram, plus a streaming decoder that never splits a multi-byte character.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// The GPT-2 byte <-> printable character table used by every ByteLevel tokenizer.
var (
	byteToChar [256]rune
	charToByte = map[rune]byte{}
)

func init() {
	extra := rune(256)
	for b := 0; b < 256; b++ {
		c := rune(b)
		if !((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) {
			c = extra
			extra++
		}
		byteToChar[b], charToByte[c] = c, byte(b)
	}
}

// What the tokenizers library treats as \s. Go's own \s is ASCII only.
const wsClass = `\t\n\x0b\f\r \x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}\x{feff}`

const gpt2Split = `'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]| ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

// compileSplit turns a pre-tokenizer regex into a Go one. Go has no lookahead, so the usual `\s+(?!\S)` branch
// becomes a named group and chunks() gives back the last space itself. Returns nil when the pattern needs
// other syntax Go does not have.
func compileSplit(p string) (*regexp.Regexp, int) {
	p = strings.Replace(p, `\s+(?!\S)`, `(?P<wsla>\s+)`, 1)
	var sb strings.Builder
	inClass := false
	for i := 0; i < len(p); i++ {
		c := p[i]
		switch {
		case c == '\\' && i+1 < len(p):
			i++
			switch {
			case p[i] == 's' && inClass:
				sb.WriteString(wsClass)
			case p[i] == 's':
				sb.WriteString("[" + wsClass + "]")
			case p[i] == 'S' && !inClass:
				sb.WriteString("[^" + wsClass + "]")
			case p[i] == 'S':
				return nil, -1
			default:
				sb.WriteByte('\\')
				sb.WriteByte(p[i])
			}
		case c == '[' && !inClass:
			inClass = true
			sb.WriteByte(c)
		case c == ']' && inClass:
			inClass = false
			sb.WriteByte(c)
		default:
			sb.WriteByte(c)
		}
	}
	re, err := regexp.Compile(sb.String())
	if err != nil {
		return nil, -1
	}
	return re, re.SubexpIndex("wsla")
}

type tokenizer struct {
	vocabSize                                          int
	unigram, byteLevel, metaspace, prefixSpace, addBos bool
	hasTemplate                                        bool
	bosID, unkID                                       int // -1 when the tokenizer has none
	eos, special                                       map[int]bool
	tokID                                              map[string]int
	idTok                                              []string
	scores                                             []float32 // unigram only
	ranks                                              map[[2]string]int
	cache                                              map[string][]string
	added                                              map[string]int
	addedRe, pre                                       *regexp.Regexp
	preLA                                              int
}

// tokenString reads a token given either as "text" or as {"content": "text"}.
func tokenString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if m, ok := v.(map[string]any); ok {
		s, _ := m["content"].(string)
		return s
	}
	return ""
}

// compactJSON re-encodes a JSON value without spaces or \u escapes, so simple substring checks work.
func compactJSON(raw json.RawMessage) string {
	var v any
	if json.Unmarshal(raw, &v) != nil || v == nil {
		return "{}"
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.Encode(v)
	return b.String()
}

func newTokenizer(dir string, cfg map[string]any) (*tokenizer, error) {
	b, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return nil, errors.New("Missing tokenizer.json in model directory.")
	}
	var raw struct {
		Added []struct {
			ID      *int    `json:"id"`
			Content *string `json:"content"`
			Special bool    `json:"special"`
		} `json:"added_tokens"`
		Normalizer json.RawMessage `json:"normalizer"`
		Pre        json.RawMessage `json:"pre_tokenizer"`
		Decoder    json.RawMessage `json:"decoder"`
		Model      struct {
			Vocab    json.RawMessage   `json:"vocab"`
			Merges   []json.RawMessage `json:"merges"`
			UnkID    *int              `json:"unk_id"`
			UnkToken json.RawMessage   `json:"unk_token"`
		} `json:"model"`
	}
	bad := errors.New("Invalid tokenizer.json: missing model.vocab.")
	vocab := bytes.TrimSpace(raw.Model.Vocab)
	if json.Unmarshal(b, &raw) != nil {
		return nil, bad
	}
	if vocab = bytes.TrimSpace(raw.Model.Vocab); len(vocab) == 0 || string(vocab) == "null" {
		return nil, bad
	}
	t := &tokenizer{bosID: -1, unkID: -1, eos: map[int]bool{}, special: map[int]bool{}, tokID: map[string]int{},
		ranks: map[[2]string]int{}, cache: map[string][]string{}, added: map[string]int{}}

	// vocabulary: an object for BPE, or an array of [token, score] for Unigram
	maxID := -1
	if vocab[0] == '[' {
		var arr [][]any
		if json.Unmarshal(vocab, &arr) != nil {
			return nil, bad
		}
		t.unigram, t.scores, t.idTok = true, make([]float32, len(arr)), make([]string, len(arr))
		for i, p := range arr {
			if len(p) < 2 {
				continue
			}
			s, _ := p[0].(string)
			f, _ := p[1].(float64)
			t.tokID[s], t.idTok[i], t.scores[i] = i, s, float32(f)
		}
		maxID = len(arr) - 1
		if raw.Model.UnkID != nil {
			t.unkID = *raw.Model.UnkID
		}
	} else {
		if json.Unmarshal(vocab, &t.tokID) != nil || t.tokID == nil {
			return nil, bad
		}
		for _, id := range t.tokID {
			if id > maxID {
				maxID = id
			}
		}
		t.idTok = make([]string, maxID+1)
		for s, id := range t.tokID {
			if id >= 0 {
				t.idTok[id] = s
			}
		}
		var unk string
		if json.Unmarshal(raw.Model.UnkToken, &unk) == nil && unk != "" {
			if id, ok := t.tokID[unk]; ok {
				t.unkID = id
			}
		}
	}

	// BPE merges, written as "a b" or ["a", "b"]; the position in the list is the merge priority
	for i, m := range raw.Model.Merges {
		var key [2]string
		if len(m) > 0 && m[0] == '[' {
			var p []string
			if json.Unmarshal(m, &p) != nil || len(p) != 2 {
				continue
			}
			key = [2]string{p[0], p[1]}
		} else {
			var s string
			if json.Unmarshal(m, &s) != nil {
				continue
			}
			k := strings.IndexByte(s, ' ')
			if k < 0 {
				continue
			}
			key = [2]string{s[:k], s[k+1:]}
		}
		if _, ok := t.ranks[key]; !ok {
			t.ranks[key] = i
		}
	}

	// added and special tokens are matched as whole strings before anything else
	var addedList []string
	for _, at := range raw.Added {
		if at.ID == nil || at.Content == nil || *at.ID < 0 || *at.ID > 1<<24 {
			continue
		}
		id, s := *at.ID, *at.Content
		t.tokID[s] = id
		for id >= len(t.idTok) {
			t.idTok = append(t.idTok, "")
		}
		t.idTok[id] = s
		if id > maxID {
			maxID = id
		}
		t.added[s] = id
		if s != "" {
			addedList = append(addedList, regexp.QuoteMeta(s))
		}
		if at.Special {
			t.special[id] = true
		}
	}
	t.vocabSize = maxID + 1
	if len(addedList) > 0 {
		sort.SliceStable(addedList, func(i, j int) bool { return len(addedList[i]) > len(addedList[j]) }) // longest first
		t.addedRe, _ = regexp.Compile(strings.Join(addedList, "|"))
	}

	// which family of tokenizer is this?
	preStr, decStr, normStr := compactJSON(raw.Pre), compactJSON(raw.Decoder), compactJSON(raw.Normalizer)
	t.byteLevel = strings.Contains(preStr, `"ByteLevel"`) || strings.Contains(decStr, `"ByteLevel"`) || t.hasToken("Ġthe") || t.hasToken("Ġa")
	t.metaspace = !t.byteLevel && (strings.Contains(preStr, `"Metaspace"`) || strings.Contains(decStr, `"Metaspace"`) ||
		strings.Contains(normStr, `"Prepend"`) || t.hasToken("▁the") || t.hasToken("▁a"))
	var pre map[string]any
	json.Unmarshal(raw.Pre, &pre)
	var list []map[string]any
	if arr, ok := pre["pretokenizers"].([]any); ok && pre["type"] == "Sequence" {
		for _, x := range arr {
			if m, ok := x.(map[string]any); ok {
				list = append(list, m)
			}
		}
	} else if pre != nil {
		list = append(list, pre)
	}
	t.pre, t.preLA = compileSplit(gpt2Split)
	for _, pt := range list {
		switch pt["type"] {
		case "ByteLevel":
			if pt["add_prefix_space"] == true {
				t.prefixSpace = true
			}
		case "Metaspace":
			if pt["add_prefix_space"] != false && pt["prepend_scheme"] != "never" {
				t.prefixSpace = true
			}
		case "Split":
			if pm, ok := pt["pattern"].(map[string]any); ok {
				if rx, ok := pm["Regex"].(string); ok {
					if re, la := compileSplit(rx); re != nil {
						t.pre, t.preLA = re, la
					}
				}
			}
		}
	}
	if strings.Contains(normStr, `"value":"▁"`) {
		t.prefixSpace = true
	}

	// end-of-text and beginning-of-text tokens
	var tc map[string]any
	if tb, err := os.ReadFile(filepath.Join(dir, "tokenizer_config.json")); err == nil {
		json.Unmarshal(tb, &tc)
	}
	switch e := cfg["eos_token_id"].(type) {
	case float64:
		t.eos[int(e)] = true
	case []any:
		for _, x := range e {
			if f, ok := x.(float64); ok {
				t.eos[int(f)] = true
			}
		}
	}
	if id, ok := t.tokID[tokenString(tc["eos_token"])]; ok && tokenString(tc["eos_token"]) != "" {
		t.eos[id] = true
	}
	for _, c := range []string{"<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end|>", "</s>", "<eos>"} {
		if id, ok := t.tokID[c]; ok && t.special[id] {
			t.eos[id] = true
		}
	}
	if f, ok := cfg["bos_token_id"].(float64); ok {
		t.bosID = int(f)
	} else if s := tokenString(tc["bos_token"]); s != "" {
		if id, ok := t.tokID[s]; ok {
			t.bosID = id
		}
	}
	t.addBos = tc["add_bos_token"] == true
	tmpl, _ := tc["chat_template"].(string)
	if arr, ok := tc["chat_template"].([]any); ok && len(arr) > 0 {
		if m, ok := arr[0].(map[string]any); ok {
			tmpl, _ = m["template"].(string)
		}
	}
	t.hasTemplate = tmpl != "" || (t.hasToken("<|im_start|>") && t.hasToken("<|im_end|>")) || (t.hasToken("<|start_header_id|>") && t.hasToken("<|eot_id|>"))
	return t, nil
}

func (t *tokenizer) hasToken(s string) bool { _, ok := t.tokID[s]; return ok }

// bpe merges the characters of one word, always applying the highest-priority merge first.
func (t *tokenizer) bpe(token string) []string {
	if w, ok := t.cache[token]; ok {
		return w
	}
	word := make([]string, 0, len(token))
	for _, r := range token {
		word = append(word, string(r))
	}
	if len(word) <= 1 {
		return word
	}
	for len(word) > 1 {
		best, bi := -1, -1
		for i := 0; i < len(word)-1; i++ {
			if rank, ok := t.ranks[[2]string{word[i], word[i+1]}]; ok && (best < 0 || rank < best) {
				best, bi = rank, i
			}
		}
		if best < 0 {
			break
		}
		first, second := word[bi], word[bi+1]
		merged := word[:0:0]
		for i := 0; i < len(word); i++ {
			if i < len(word)-1 && word[i] == first && word[i+1] == second {
				merged = append(merged, first+second)
				i++
			} else {
				merged = append(merged, word[i])
			}
		}
		word = merged
	}
	if len(t.cache) > 32768 {
		t.cache = map[string][]string{}
	}
	t.cache[token] = word
	return word
}

// byteFallback encodes text as SentencePiece byte tokens (<0xNN>).
func (t *tokenizer) byteFallback(s string, out []int) []int {
	for i := 0; i < len(s); i++ {
		if id, ok := t.tokID[fmt.Sprintf("<0x%02X>", s[i])]; ok {
			out = append(out, id)
		} else if t.unkID >= 0 {
			out = append(out, t.unkID)
		}
	}
	return out
}

// encodeUnigram finds the best-scoring split of s into vocabulary pieces (Viterbi).
func (t *tokenizer) encodeUnigram(s string, out []int) []int {
	chars := []rune(s)
	n := len(chars)
	if n == 0 {
		return out
	}
	unk := t.unkID
	if unk < 0 {
		unk = 0
	}
	score, prev, ids := make([]float64, n+1), make([]int, n+1), make([]int, n+1)
	reached := make([]bool, n+1)
	for i := range prev {
		prev[i], ids[i] = -1, unk
	}
	reached[0] = true
	for i := 0; i < n; i++ {
		if !reached[i] {
			continue
		}
		maxLen := n - i
		if maxLen > 32 {
			maxLen = 32
		}
		for l := 1; l <= maxLen; l++ {
			id, ok := t.tokID[string(chars[i:i+l])]
			if !ok || id >= len(t.scores) {
				continue
			}
			if sc := score[i] + float64(t.scores[id]); !reached[i+l] || sc > score[i+l] {
				score[i+l], prev[i+l], ids[i+l], reached[i+l] = sc, i, id, true
			}
		}
		if prev[i+1] == -1 { // no piece covers this character
			score[i+1], prev[i+1], reached[i+1] = score[i]-50, i, true
			if fb := t.byteFallback(string(chars[i]), nil); len(fb) > 0 {
				ids[i+1] = fb[0]
			}
		}
	}
	start := len(out)
	for pos := n; pos > 0; pos = prev[pos] {
		out = append(out, ids[pos])
	}
	for i, j := start, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// chunks splits text into the pieces BPE works on, with the pre-tokenizer regex.
func (t *tokenizer) chunks(s string) []string {
	var out []string
	for pos := 0; pos < len(s); {
		loc := t.pre.FindStringSubmatchIndex(s[pos:])
		if loc == nil {
			out = append(out, s[pos:])
			break
		}
		a, b := pos+loc[0], pos+loc[1]
		if b == a { // an empty match would never advance
			_, sz := utf8.DecodeRuneInString(s[pos:])
			out = append(out, s[pos:pos+sz])
			pos += sz
			continue
		}
		if a > pos {
			out = append(out, s[pos:a])
		}
		// `\s+(?!\S)`: a run of spaces before a word gives its last space back to that word
		if t.preLA >= 0 && loc[2*t.preLA] >= 0 && b < len(s) {
			if _, sz := utf8.DecodeLastRuneInString(s[a:b]); b-a > sz {
				b -= sz
			}
		}
		out = append(out, s[a:b])
		pos = b
	}
	return out
}

func byteLevelMap(s string) string {
	var sb strings.Builder
	for i := 0; i < len(s); i++ {
		sb.WriteRune(byteToChar[s[i]])
	}
	return sb.String()
}

func (t *tokenizer) encodePlain(seg string, out []int) []int {
	if seg == "" {
		return out
	}
	if t.metaspace {
		if t.prefixSpace && !strings.HasPrefix(seg, " ") {
			seg = " " + seg
		}
		seg = strings.ReplaceAll(seg, " ", "▁")
		if t.unigram {
			return t.encodeUnigram(seg, out)
		}
		for _, p := range t.bpe(seg) {
			if id, ok := t.tokID[p]; ok {
				out = append(out, id)
			} else if p == "▁" {
				out = t.byteFallback(" ", out)
			} else {
				out = t.byteFallback(p, out)
			}
		}
		return out
	}
	if t.byteLevel && t.prefixSpace && !strings.HasPrefix(seg, " ") {
		seg = " " + seg
	}
	if t.unigram {
		if t.byteLevel {
			seg = byteLevelMap(seg)
		}
		return t.encodeUnigram(seg, out)
	}
	for _, chunk := range t.chunks(seg) {
		if t.byteLevel {
			for _, p := range t.bpe(byteLevelMap(chunk)) {
				if id, ok := t.tokID[p]; ok {
					out = append(out, id)
				} else if t.unkID >= 0 {
					out = append(out, t.unkID)
				}
			}
			continue
		}
		for _, p := range t.bpe(chunk) {
			if id, ok := t.tokID[p]; ok {
				out = append(out, id)
			} else {
				out = t.byteFallback(p, out)
			}
		}
	}
	return out
}

func (t *tokenizer) encode(text string, addBos bool) []int {
	var ids []int
	if addBos && t.bosID >= 0 {
		ids = append(ids, t.bosID)
	}
	if text == "" {
		return ids
	}
	if t.addedRe == nil {
		return t.encodePlain(text, ids)
	}
	pos := 0
	for _, loc := range t.addedRe.FindAllStringIndex(text, -1) {
		ids = t.encodePlain(text[pos:loc[0]], ids)
		ids = append(ids, t.added[text[loc[0]:loc[1]]])
		pos = loc[1]
	}
	return t.encodePlain(text[pos:], ids)
}

// streamDec turns token ids back into text one at a time. Bytes of a character that is split across
// tokens are held back until the character is complete.
type streamDec struct {
	t     *tokenizer
	pend  []byte
	first bool
}

func (t *tokenizer) decoder() *streamDec { return &streamDec{t: t, first: true} }

// utf8Seq looks at the first character in b the way browsers decode UTF-8: it returns how many bytes it
// covers and whether they are a valid character (0), a broken one to show as one U+FFFD (1), or the
// beginning of a character whose remaining bytes have not arrived yet (2).
func utf8Seq(b []byte) (size, status int) {
	c, need, lo, hi := b[0], 0, byte(0x80), byte(0xBF)
	switch {
	case c < 0x80:
		return 1, 0
	case c >= 0xC2 && c <= 0xDF:
		need = 1
	case c >= 0xE0 && c <= 0xEF:
		need = 2
		if c == 0xE0 {
			lo = 0xA0
		} else if c == 0xED {
			hi = 0x9F
		}
	case c >= 0xF0 && c <= 0xF4:
		need = 3
		if c == 0xF0 {
			lo = 0x90
		} else if c == 0xF4 {
			hi = 0x8F
		}
	default:
		return 1, 1
	}
	for i := 1; i <= need; i++ {
		if i >= len(b) {
			return i, 2
		}
		if b[i] < lo || b[i] > hi {
			return i, 1
		}
		lo, hi = 0x80, 0xBF
	}
	return need + 1, 0
}

func (d *streamDec) drain(final bool) string {
	var sb strings.Builder
	b := d.pend
	for len(b) > 0 {
		size, status := utf8Seq(b)
		if status == 2 && !final {
			break // incomplete character: wait for the next token
		}
		if status == 0 {
			sb.Write(b[:size])
		} else {
			sb.WriteRune(utf8.RuneError)
		}
		b = b[size:]
	}
	d.pend = append(d.pend[:0], b...)
	return sb.String()
}

func (d *streamDec) push(id int) string {
	t := d.t
	if id < 0 || id >= len(t.idTok) || t.idTok[id] == "" {
		return ""
	}
	tok := t.idTok[id]
	if tok == "<think>" || tok == "</think>" { // kept even when marked special, so reasoning can be shown
		return d.drain(true) + tok
	}
	if t.special[id] {
		return ""
	}
	if len(tok) == 6 && strings.HasPrefix(tok, "<0x") && tok[5] == '>' { // SentencePiece byte token
		var v byte
		if _, err := fmt.Sscanf(tok[3:5], "%02x", &v); err == nil {
			d.pend = append(d.pend, v)
			return d.drain(false)
		}
	}
	if t.byteLevel {
		if _, ok := t.added[tok]; ok {
			return d.drain(true) + tok
		}
		for _, r := range tok {
			if b, ok := charToByte[r]; ok {
				d.pend = append(d.pend, b)
			} else {
				d.pend = utf8.AppendRune(d.pend, r)
			}
		}
		return d.drain(false)
	}
	if t.metaspace {
		s := strings.ReplaceAll(tok, "▁", " ")
		if d.first && t.prefixSpace {
			s = strings.TrimPrefix(s, " ")
		}
		d.first = false
		return d.drain(true) + s
	}
	return d.drain(true) + tok
}

func (d *streamDec) flush() string { return d.drain(true) }

func (t *tokenizer) decode(ids []int) string {
	d := t.decoder()
	var sb strings.Builder
	for _, id := range ids {
		sb.WriteString(d.push(id))
	}
	sb.WriteString(d.flush())
	return sb.String()
}
