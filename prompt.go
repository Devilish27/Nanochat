package main

// Prompt formats for local models, and the two small stream filters that sit between the model and the
// page: one stops at an end marker without leaking it, one moves <think>...</think> into the Thinking box.

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

type chatMsg struct{ Role, Content string }

type prompt struct {
	text     string
	stops    []string
	echo     string // shown before the first generated token (story continuation)
	trimLead bool   // drop one leading space of the reply
	format   string
}

var (
	storyRe   = regexp.MustCompile(`(?i)^(once upon a time|one (sunny|warm|cool|cold|rainy|bright|quiet|windy|snowy|winter|summer|spring|autumn|fall) (day|morning|afternoon|evening|night)|one day,|long ago,|in a (small|quiet|little|cozy|sunny|green)|title\s*:)`)
	qPrefixRe = regexp.MustCompile(`(?i)^Question\s*:\s*`)
	aSuffixRe = regexp.MustCompile(`(?i)\n*Answer\s*:\s*$`)
)

// detectFormat picks the chat template from the special tokens the tokenizer knows.
func detectFormat(modelType string, t *tokenizer) string {
	switch {
	case t.hasToken("<|im_start|>") && t.hasToken("<|im_end|>"):
		return "chatml"
	case t.hasToken("<|start_header_id|>") && t.hasToken("<|eot_id|>"):
		return "llama3"
	case t.hasToken("<|user|>") && t.hasToken("<|assistant|>"):
		return "zephyr"
	case t.hasToken("[INST]") && t.hasToken("[/INST]"):
		return "mistral"
	case modelType == "tinylm":
		return "qa"
	}
	return "chat"
}

func buildPrompt(msgs []chatMsg, system, format, modelType string, t *tokenizer) prompt {
	var sysParts []string
	if s := strings.TrimSpace(system); s != "" {
		sysParts = append(sysParts, s)
	}
	var turns []chatMsg
	lastUser := ""
	for _, m := range msgs {
		switch m.Role {
		case "system":
			if s := strings.TrimSpace(m.Content); s != "" {
				sysParts = append(sysParts, s)
			}
		case "user":
			lastUser = strings.TrimSpace(m.Content)
			turns = append(turns, m)
		case "assistant":
			turns = append(turns, m)
		}
	}
	sys := strings.Join(sysParts, "\n")

	mode := format
	if mode == "auto" {
		mode = detectFormat(modelType, t)
		// base TinyLM models continue a story opener better than they answer it
		if mode == "qa" && storyRe.MatchString(lastUser) {
			c := strings.TrimRightFunc(lastUser, unicode.IsSpace)
			return prompt{text: c, stops: []string{"<|endoftext|>", "\n\nQuestion:", "\nQuestion:"}, echo: c, format: "story"}
		}
	}
	var sb strings.Builder
	switch mode {
	case "raw":
		return prompt{text: lastUser, stops: []string{"<|endoftext|>"}, format: "raw"}
	case "chatml":
		if sys != "" {
			sb.WriteString("<|im_start|>system\n" + sys + "<|im_end|>\n")
		}
		for _, m := range turns {
			sb.WriteString("<|im_start|>" + m.Role + "\n" + m.Content + "<|im_end|>\n")
		}
		sb.WriteString("<|im_start|>assistant\n")
		return prompt{text: sb.String(), stops: []string{"<|im_end|>", "<|im_start|>", "<|endoftext|>"}, format: "chatml"}
	case "llama3":
		if t.hasToken("<|begin_of_text|>") {
			sb.WriteString("<|begin_of_text|>")
		}
		if sys != "" {
			sb.WriteString("<|start_header_id|>system<|end_header_id|>\n\n" + sys + "<|eot_id|>")
		}
		for _, m := range turns {
			sb.WriteString("<|start_header_id|>" + m.Role + "<|end_header_id|>\n\n" + m.Content + "<|eot_id|>")
		}
		sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
		return prompt{text: sb.String(), stops: []string{"<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"}, format: "llama3"}
	case "zephyr":
		if sys != "" {
			sb.WriteString("<|system|>\n" + sys + "</s>\n")
		}
		for _, m := range turns {
			sb.WriteString("<|" + m.Role + "|>\n" + m.Content + "</s>\n")
		}
		sb.WriteString("<|assistant|>\n")
		return prompt{text: sb.String(), stops: []string{"</s>", "<|user|>", "<|endoftext|>"}, format: "zephyr"}
	case "mistral":
		if t.hasToken("<s>") {
			sb.WriteString("<s>")
		}
		sysDone := sys == ""
		for _, m := range turns {
			if m.Role == "user" {
				c := m.Content
				if !sysDone {
					c = sys + "\n\n" + c
				}
				sysDone = true
				sb.WriteString("[INST] " + c + " [/INST]")
			} else {
				sb.WriteString(" " + m.Content + "</s>")
			}
		}
		return prompt{text: sb.String(), stops: []string{"</s>", "[INST]"}, trimLead: true, format: "mistral"}
	case "qa":
		var parts []string
		if sys != "" {
			parts = append(parts, sys)
		}
		for _, m := range turns {
			if m.Role == "user" {
				q := aSuffixRe.ReplaceAllString(qPrefixRe.ReplaceAllString(strings.TrimSpace(m.Content), ""), "")
				parts = append(parts, "Question: "+q)
			} else {
				parts = append(parts, "Answer: "+strings.TrimSpace(m.Content)+"\n")
			}
		}
		parts = append(parts, "Answer:")
		return prompt{text: strings.Join(parts, "\n"), trimLead: true, format: "qa",
			stops: []string{"\n\nQuestion:", "\nQuestion:", "\n\nUser:", "\nUser:", "<|endoftext|>"}}
	}
	// default: a plain User / Assistant dialogue
	var parts []string
	if sys != "" {
		parts = append(parts, "System: "+sys)
	}
	for _, m := range turns {
		label := "Assistant"
		if m.Role == "user" {
			label = "User"
		}
		parts = append(parts, label+": "+strings.TrimSpace(m.Content))
	}
	parts = append(parts, "Assistant:")
	return prompt{text: strings.Join(parts, "\n"), trimLead: true, format: "chat",
		stops: []string{"\n\nUser:", "\nUser:", "\n\nHuman:", "\nHuman:", "\n\nQuestion:", "<|endoftext|>"}}
}

// stopFilter passes text through until a stop sequence appears. Text that could be the start of a stop
// sequence is held back until the next piece decides it.
type stopFilter struct {
	buf     string
	stops   []string
	stopped bool
}

func (f *stopFilter) push(chunk string) (emit string, hit bool) {
	if f.stopped || chunk == "" {
		return "", f.stopped
	}
	f.buf += chunk
	first := -1
	for _, s := range f.stops {
		if i := strings.Index(f.buf, s); s != "" && i >= 0 && (first < 0 || i < first) {
			first = i
		}
	}
	if first >= 0 {
		emit, f.buf, f.stopped = f.buf[:first], "", true
		return emit, true
	}
	hold := 0
	for _, s := range f.stops {
		n := len(s) - 1
		if len(f.buf) < n {
			n = len(f.buf)
		}
		for ; n > hold; n-- {
			if utf8.RuneStart(s[n]) && strings.HasSuffix(f.buf, s[:n]) {
				hold = n
				break
			}
		}
	}
	safe := len(f.buf) - hold
	if safe <= 0 {
		return "", false
	}
	emit, f.buf = f.buf[:safe], f.buf[safe:]
	return emit, false
}

func (f *stopFilter) flush() string {
	if f.stopped {
		return ""
	}
	out := f.buf
	f.buf = ""
	return out
}

// thinkSplit separates <think>...</think> blocks (Qwen3, R1 distills) from the visible answer.
type thinkSplit struct {
	in, started bool
	pend        string
}

// partialTag reports how many trailing bytes of s could be the beginning of tag.
func partialTag(s, tag string) int {
	k := len(tag) - 1
	if len(s) < k {
		k = len(s)
	}
	for ; k >= 1; k-- {
		if strings.HasSuffix(s, tag[:k]) {
			return k
		}
	}
	return 0
}

func (t *thinkSplit) push(text string) (content, reasoning string) {
	const open, shut = "<think>", "</think>"
	if text == "" {
		return
	}
	t.pend += text
	for len(t.pend) > 0 {
		if !t.in {
			if !t.started {
				tr := strings.TrimLeftFunc(t.pend, unicode.IsSpace)
				if len(tr) < len(open) && strings.HasPrefix(open, tr) {
					break // could still become <think>: wait
				}
				t.started = true
				if strings.HasPrefix(tr, open) {
					t.in, t.pend = true, strings.TrimPrefix(tr[len(open):], "\n")
					continue
				}
			}
			if i := strings.Index(t.pend, open); i >= 0 {
				content += t.pend[:i]
				t.in, t.pend = true, strings.TrimPrefix(t.pend[i+len(open):], "\n")
				continue
			}
			cut := len(t.pend) - partialTag(t.pend, open)
			content += t.pend[:cut]
			t.pend = t.pend[cut:]
			break
		}
		if i := strings.Index(t.pend, shut); i >= 0 {
			reasoning += t.pend[:i]
			t.in, t.pend = false, strings.TrimLeftFunc(t.pend[i+len(shut):], unicode.IsSpace)
			continue
		}
		cut := len(t.pend) - partialTag(t.pend, shut)
		reasoning += t.pend[:cut]
		t.pend = t.pend[cut:]
		break
	}
	return
}

func (t *thinkSplit) flush() (content, reasoning string) {
	rest := t.pend
	t.pend = ""
	if t.in {
		return "", rest
	}
	return rest, ""
}
