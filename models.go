package main

// Local model library (data/models/<id>/) and its HTTP API: list, inspect, load into RAM, delete, import
// from a .zip URL or the HuggingFace Hub, upload from the browser, and an OpenAI-style chat endpoint.

import (
	"archive/zip"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"
)

const (
	maxModelBytes = 900 << 20 // size limit for one imported or uploaded model
	maxZipFiles   = 64
	metaFile      = "nanochat-model.json" // what Nanochat knows about a model, kept next to its files
	stagePrefix   = "_tmp_"               // work folders inside data/models; removed at startup
)

var (
	modelExts  = map[string]bool{".safetensors": true, ".json": true, ".md": true, ".txt": true, ".model": true}
	modelIDRe  = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,62}$`)
	slugBadRe  = regexp.MustCompile(`[^a-z0-9._-]+|\.{2,}`)
	reservedRe = regexp.MustCompile(`^(con|prn|aux|nul|com[0-9]|lpt[0-9]|import|upload)(\..*)?$`) // Windows device names and our own routes
	hfRepoRe   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,80}/[A-Za-z0-9][A-Za-z0-9._-]{0,96}$`)
	hfTrimRe   = regexp.MustCompile(`(?i)^https?://huggingface\.co/|/(tree|blob|resolve)/.*$|/+$`)
	frontRe    = regexp.MustCompile(`(?s)^---\r?\n.*?\r?\n---\r?\n`)
	quantSufRe = regexp.MustCompile(`(?i)-(int8|bf16|fp16|f32)$`)
	sizeWordRe = regexp.MustCompile(`(?i)^(\d+[mbk]|gpt\d*)$`)
	modelsMu   sync.Mutex // one change to the library at a time
)

type modelMeta struct {
	ID              string  `json:"id"`
	ModelID         string  `json:"modelId"` // "local:<id>", as used in the model picker
	Name            string  `json:"name"`
	Arch            string  `json:"arch"`
	ModelType       string  `json:"modelType"`
	ParamsTotal     float64 `json:"paramsTotal"`
	Quantization    string  `json:"quantization"`
	ContextLength   int     `json:"contextLength"`
	VocabSize       int     `json:"vocabSize"`
	Layers          int     `json:"layers"`
	HiddenSize      int     `json:"hiddenSize"`
	SizeBytes       int64   `json:"sizeBytes"`
	SourceType      string  `json:"sourceType"` // local, url, huggingface or upload
	SourceURL       string  `json:"sourceUrl"`
	HasChatTemplate bool    `json:"hasChatTemplate"`
	DefaultFormat   string  `json:"defaultFormat"`
	ModelCard       string  `json:"modelCard"`
	Loaded          bool    `json:"loaded"`
	RAMBytes        int64   `json:"ramBytes"`
	Active          bool    `json:"active"`
	CreatedAt       string  `json:"createdAt"`
	WeightsBytes    int64   `json:"weightsBytes"` // size of model.safetensors when inspected
}

func mb(n int64) int64 { return n >> 20 }

func nowISO() string { return time.Now().UTC().Format("2006-01-02T15:04:05.000Z") }

func randHex() string {
	b := make([]byte, 3)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// truncate cuts s to at most n bytes without splitting a character.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	for n > 0 && !utf8.RuneStart(s[n]) {
		n--
	}
	return s[:n]
}

// sanitizeSlug makes a safe folder name (also safe on Windows) out of any text.
func sanitizeSlug(raw string) string {
	s := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(raw)), ".zip")
	s = strings.Trim(slugBadRe.ReplaceAllString(s, "-"), "._-")
	if len(s) > 60 {
		s = strings.Trim(s[:60], "._-")
	}
	if s == "" {
		return "model-" + strconv.FormatInt(time.Now().UnixMilli(), 36)
	}
	if reservedRe.MatchString(s) {
		s = "model-" + s
	}
	return s
}

func modelDir(id string) (string, error) {
	if !modelIDRe.MatchString(id) || strings.Contains(id, "..") {
		return "", errors.New("Invalid local model identifier.")
	}
	return filepath.Join(modelsDir, id), nil
}

// cleanStaging removes work folders a crash or power cut left behind.
func cleanStaging() {
	es, _ := os.ReadDir(modelsDir)
	for _, e := range es {
		if strings.HasPrefix(e.Name(), stagePrefix) {
			os.RemoveAll(filepath.Join(modelsDir, e.Name()))
		}
	}
}

func prettyName(slug, quant string) string {
	var words []string
	for _, w := range strings.FieldsFunc(quantSufRe.ReplaceAllString(slug, ""), func(r rune) bool { return r == '-' || r == '_' }) {
		lw := strings.ToLower(w)
		switch {
		case sizeWordRe.MatchString(w):
			w = strings.ToUpper(w)
		case lw == "tinylm":
			w = "TinyLM"
		case strings.HasPrefix(lw, "smollm"):
			w = "SmolLM" + w[6:]
		default:
			w = strings.ToUpper(w[:1]) + w[1:]
		}
		words = append(words, w)
	}
	name := strings.Join(words, "-")
	if name == "" {
		name = slug
	}
	if quant != "" {
		name += " (" + strings.ToLower(quant) + ")"
	}
	return name
}

// inspectModelDir checks that a folder holds a model the engine can run and collects what the UI shows.
func inspectModelDir(dir, slug, customName string) (modelMeta, error) {
	var m modelMeta
	st := func(name string) (int64, bool) {
		fi, err := os.Stat(filepath.Join(dir, name))
		if err != nil || fi.IsDir() {
			return 0, false
		}
		return fi.Size(), true
	}
	if _, ok := st("config.json"); !ok {
		return m, errors.New("Archive is missing config.json.")
	}
	if _, ok := st("tokenizer.json"); !ok {
		return m, errors.New("Archive is missing tokenizer.json (HuggingFace fast tokenizer required).")
	}
	weights, ok := st("model.safetensors")
	if !ok {
		return m, errors.New("Archive is missing model.safetensors.")
	}
	cfg, err := readJSONFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return m, errors.New("config.json is not valid JSON.")
	}
	info, err := inspectSafetensors(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return m, err
	}
	a, err := parseArch(cfg, info.Tensors)
	if err != nil {
		return m, err
	}
	tk, err := newTokenizer(dir, cfg)
	if err != nil {
		return m, err
	}
	for _, f := range []string{"MODEL_CARD.md", "README.md", "model_card.md", "readme.md"} {
		if b, err := os.ReadFile(filepath.Join(dir, f)); err == nil {
			// HuggingFace cards start with a YAML block that is not meant for readers
			m.ModelCard = truncate(strings.TrimSpace(frontRe.ReplaceAllString(strings.ToValidUTF8(string(b), "\uFFFD"), "")), 24000)
			break
		}
	}
	m.ParamsTotal = float64(info.TotalParams)
	if f, ok := cfg["params_total"].(float64); ok && f > 0 {
		m.ParamsTotal = f
	}
	if m.Name = truncate(strings.TrimSpace(customName), 80); m.Name == "" {
		m.Name = prettyName(slug, info.Quant)
	}
	m.Arch, m.ModelType, m.Quantization, m.ContextLength = a.label, a.modelType, info.Quant, a.maxSeq
	m.VocabSize, m.Layers, m.HiddenSize, m.WeightsBytes = a.vocab, a.layers, a.d, weights
	m.HasChatTemplate, m.DefaultFormat = tk.hasTemplate, detectFormat(a.modelType, tk)
	es, _ := os.ReadDir(dir)
	for _, e := range es {
		if fi, err := e.Info(); err == nil && !e.IsDir() && e.Name() != metaFile {
			m.SizeBytes += fi.Size()
		}
	}
	return m, nil
}

func writeMeta(dir string, m modelMeta) {
	m.Loaded, m.RAMBytes, m.Active = false, 0, false
	if b, err := json.MarshalIndent(m, "", " "); err == nil {
		tmp := filepath.Join(dir, metaFile+".tmp")
		if os.WriteFile(tmp, b, 0o600) == nil {
			os.Rename(tmp, filepath.Join(dir, metaFile))
		}
	}
}

// listModels reads data/models. A folder copied in by hand is inspected once and remembered.
func listModels() []modelMeta {
	modelsMu.Lock()
	defer modelsMu.Unlock()
	out := []modelMeta{}
	es, _ := os.ReadDir(modelsDir)
	for _, e := range es {
		id := e.Name()
		if !e.IsDir() || !modelIDRe.MatchString(id) || strings.Contains(id, "..") {
			continue
		}
		dir := filepath.Join(modelsDir, id)
		fi, err := os.Stat(filepath.Join(dir, "model.safetensors"))
		if err != nil {
			continue
		}
		var m modelMeta
		b, err := os.ReadFile(filepath.Join(dir, metaFile))
		if err != nil || json.Unmarshal(b, &m) != nil || m.Name == "" || m.WeightsBytes != fi.Size() {
			old := m
			if m, err = inspectModelDir(dir, id, old.Name); err != nil {
				continue // unfinished or broken folder
			}
			m.SourceType, m.SourceURL, m.CreatedAt = old.SourceType, old.SourceURL, old.CreatedAt
			if m.SourceType == "" {
				m.SourceType = "local"
			}
			if m.CreatedAt == "" {
				m.CreatedAt = nowISO()
			}
			writeMeta(dir, m)
		}
		m.ID, m.ModelID = id, "local:"+id
		m.Loaded, m.RAMBytes, _, m.Active = rt.info(id)
		out = append(out, m)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].CreatedAt > out[j].CreatedAt })
	return out
}

func findModel(id string) (modelMeta, bool) {
	for _, m := range listModels() {
		if m.ID == id {
			return m, true
		}
	}
	return modelMeta{}, false
}

func deleteModel(id string) error {
	dir, err := modelDir(id)
	if err != nil {
		return err
	}
	if err := rt.unload(id); err != nil {
		return err
	}
	modelsMu.Lock()
	defer modelsMu.Unlock()
	return os.RemoveAll(dir)
}

// extractZip unpacks a model archive. It refuses unsafe paths and links, skips files that are not part of
// a model, and enforces the size limit on the bytes actually written (not on what the archive claims).
func extractZip(zipPath, outDir string) (folder string, err error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return "", errors.New("Could not open .zip archive (invalid or corrupted zip file).")
	}
	defer zr.Close()
	var total int64
	count, tops := 0, map[string]bool{}
	for _, f := range zr.File {
		name := strings.ReplaceAll(f.Name, "\\", "/")
		var parts []string
		for _, p := range strings.Split(name, "/") {
			if p == ".." || p == "." {
				return "", errors.New("Rejected archive containing unsafe file paths.")
			}
			if p != "" {
				parts = append(parts, p)
			}
		}
		if strings.HasPrefix(name, "/") || strings.Contains(name, ":") {
			return "", errors.New("Rejected archive containing unsafe file paths.")
		}
		if strings.HasSuffix(name, "/") || len(parts) == 0 || parts[0] == "__MACOSX" {
			continue
		}
		base := parts[len(parts)-1]
		if strings.HasPrefix(base, "._") || base == ".DS_Store" {
			continue
		}
		if f.Mode()&os.ModeSymlink != 0 {
			return "", errors.New("Rejected archive containing symbolic links.")
		}
		if !modelExts[strings.ToLower(path.Ext(base))] || base == metaFile {
			continue
		}
		if count++; count > maxZipFiles {
			return "", errors.New("Archive contains too many files.")
		}
		if total += int64(f.UncompressedSize64); total > maxModelBytes || total < 0 {
			return "", fmt.Errorf("Uncompressed model exceeds the %d MB limit.", mb(maxModelBytes))
		}
		if len(parts) > 1 {
			tops[parts[0]] = true
		}
		if len(parts) > 2 {
			continue // only the root or one wrapper folder holds the model
		}
		rc, err := f.Open()
		if err != nil {
			return "", fmt.Errorf("Failed to extract %q.", base)
		}
		dst, err := os.OpenFile(filepath.Join(outDir, base), os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
		if err != nil {
			rc.Close()
			return "", fmt.Errorf("Failed to extract %q.", base)
		}
		n, err := io.Copy(dst, io.LimitReader(rc, int64(f.UncompressedSize64)+1))
		rc.Close()
		if cerr := dst.Close(); err == nil {
			err = cerr
		}
		if err != nil || n != int64(f.UncompressedSize64) {
			return "", fmt.Errorf("Failed to extract %q.", base)
		}
		touch()
	}
	if len(tops) == 1 {
		for t := range tops {
			folder = t
		}
	}
	return folder, nil
}

/* ---------- downloads ---------- */

// publicIP is false for loopback, private, link-local, carrier-NAT, multicast and reserved addresses.
func publicIP(ip net.IP) bool {
	if ip == nil || ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() || ip.IsLinkLocalUnicast() || ip.IsMulticast() {
		return false
	}
	if v4 := ip.To4(); v4 != nil {
		return !(v4[0] == 0 || v4[0] >= 224 || (v4[0] == 100 && v4[1] >= 64 && v4[1] <= 127))
	}
	return true
}

// checkPublicURL allows only http(s) URLs whose host is on the public internet.
func checkPublicURL(u *url.URL) error {
	if u.Scheme != "https" && u.Scheme != "http" {
		return errors.New("Only http:// and https:// URLs are supported.")
	}
	host := strings.ToLower(u.Hostname())
	if host == "" || host == "localhost" || strings.HasSuffix(host, ".local") || strings.HasSuffix(host, ".internal") || strings.HasSuffix(host, ".localhost") {
		return errors.New("Downloads from private or local network hosts are blocked.")
	}
	if ip := net.ParseIP(host); ip != nil {
		if !publicIP(ip) {
			return errors.New("Downloads from private IP addresses are blocked.")
		}
		return nil
	}
	ips, err := net.LookupIP(host)
	if err != nil || len(ips) == 0 {
		return fmt.Errorf("Could not resolve hostname %q.", host)
	}
	for _, ip := range ips {
		if !publicIP(ip) {
			return errors.New("Downloads from private IP addresses are blocked.")
		}
	}
	return nil
}

// download saves a URL to dest. Every redirect is checked again, and (without a proxy) so is the address
// that is finally dialed, so a public name cannot be pointed at a private machine.
func download(ctx context.Context, rawURL, dest string, max int64, progress func(done, total int64)) error {
	u, err := url.Parse(rawURL)
	if err != nil {
		return errors.New("Invalid URL format.")
	}
	if err := checkPublicURL(u); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
	if err != nil {
		return errors.New("Invalid URL format.")
	}
	req.Header.Set("User-Agent", "Nanochat/1.2")
	dialer := &net.Dialer{Timeout: 20 * time.Second}
	if proxy, _ := http.ProxyFromEnvironment(req); proxy == nil {
		dialer.Control = func(_, address string, _ syscall.RawConn) error {
			if host, _, err := net.SplitHostPort(address); err != nil || !publicIP(net.ParseIP(host)) {
				return errors.New("Downloads from private IP addresses are blocked.")
			}
			return nil
		}
	}
	client := &http.Client{
		Transport: &http.Transport{Proxy: http.ProxyFromEnvironment, DialContext: dialer.DialContext, TLSHandshakeTimeout: 20 * time.Second, ResponseHeaderTimeout: 60 * time.Second},
		CheckRedirect: func(r *http.Request, via []*http.Request) error {
			if len(via) > 5 {
				return errors.New("Too many redirects.")
			}
			return checkPublicURL(r.URL)
		},
	}
	defer client.CloseIdleConnections()
	res, err := client.Do(req)
	if err != nil {
		var ue *url.Error
		if errors.As(err, &ue) {
			err = ue.Err
		}
		return fmt.Errorf("Download failed: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode < 200 || res.StatusCode > 299 {
		return fmt.Errorf("Download failed: %s", res.Status)
	}
	total := res.ContentLength
	if total < 0 {
		total = 0
	}
	if total > max {
		return fmt.Errorf("Remote file is %d MB, exceeding the %d MB local import limit.", mb(total), mb(max))
	}
	f, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	buf, done, last := make([]byte, 256<<10), int64(0), time.Time{}
	for {
		n, rerr := res.Body.Read(buf)
		if n > 0 {
			if done += int64(n); done > max {
				return fmt.Errorf("Download exceeded the %d MB size limit.", mb(max))
			}
			if _, err := f.Write(buf[:n]); err != nil {
				return err
			}
			if progress != nil && time.Since(last) > 180*time.Millisecond {
				last = time.Now()
				progress(done, total)
				touch()
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return fmt.Errorf("Download failed: %v", rerr)
		}
	}
	if progress != nil {
		if total == 0 {
			total = done
		}
		progress(done, total)
	}
	return f.Close()
}

/* ---------- installing ---------- */

type progressFn func(phase, message string, done, total int64)

func newStage() (string, error) { return os.MkdirTemp(modelsDir, stagePrefix) }

// finalizeStaged validates a staged folder and moves it into the library, replacing an older copy.
func finalizeStaged(staging, preferredSlug, customName, sourceType, sourceURL string) (modelMeta, error) {
	slug := sanitizeSlug(preferredSlug)
	os.Remove(filepath.Join(staging, metaFile)) // never trust metadata that came with the files
	m, err := inspectModelDir(staging, slug, customName)
	if err != nil {
		return m, err
	}
	final, err := modelDir(slug)
	if err != nil {
		return m, err
	}
	if err := rt.unload(slug); err != nil {
		return m, err
	}
	modelsMu.Lock()
	backup := ""
	if _, err := os.Stat(final); err == nil {
		backup = filepath.Join(modelsDir, stagePrefix+"old_"+slug+"_"+randHex())
		err = os.Rename(final, backup)
	}
	if err == nil || os.IsNotExist(err) {
		if err = os.Rename(staging, final); err != nil && backup != "" {
			os.Rename(backup, final)
		}
	}
	if err == nil {
		m.SourceType, m.SourceURL, m.CreatedAt = sourceType, sourceURL, nowISO()
		writeMeta(final, m)
		if backup != "" {
			os.RemoveAll(backup)
		}
	}
	modelsMu.Unlock()
	if err != nil {
		return m, fmt.Errorf("Could not move the model into the library: %v", err)
	}
	m.ID, m.ModelID = slug, "local:"+slug
	return m, nil
}

func mbText(what string, done, total int64) string {
	if total > 0 {
		return fmt.Sprintf("%s %.1f / %.1f MB", what, float64(done)/1048576, float64(total)/1048576)
	}
	return fmt.Sprintf("%s %.1f MB", what, float64(done)/1048576)
}

func importZipURL(ctx context.Context, rawURL, customName string, p progressFn) (modelMeta, error) {
	rawURL = strings.TrimSpace(rawURL)
	work, err := newStage()
	if err != nil {
		return modelMeta{}, err
	}
	defer os.RemoveAll(work)
	zipPath, stage := filepath.Join(work, "archive.zip"), filepath.Join(work, "model")
	os.MkdirAll(stage, 0o755)
	p("connecting", "Connecting to model archive…", 0, 0)
	if err := download(ctx, rawURL, zipPath, maxModelBytes, func(d, t int64) { p("downloading", mbText("Downloading archive…", d, t), d, t) }); err != nil {
		return modelMeta{}, err
	}
	p("extracting", "Extracting and verifying safetensors…", 0, 0)
	folder, err := extractZip(zipPath, stage)
	if err != nil {
		return modelMeta{}, err
	}
	os.Remove(zipPath)
	slug := folder
	if slug == "" {
		slug = customName
	}
	if u, err := url.Parse(rawURL); slug == "" && err == nil {
		slug = strings.TrimSuffix(path.Base(u.Path), ".zip")
	}
	p("validating", "Inspecting architecture and tokenizer…", 0, 0)
	m, err := finalizeStaged(stage, slug, customName, "url", rawURL)
	if err == nil {
		p("done", "Installed "+m.Name, 0, 0)
	}
	return m, err
}

// importHF downloads a single-file safetensors model from the HuggingFace Hub (repo "owner/name").
func importHF(ctx context.Context, repo, customName string, p progressFn) (modelMeta, error) {
	repo = strings.TrimSpace(repo)
	for prev := ""; prev != repo; {
		prev, repo = repo, hfTrimRe.ReplaceAllString(repo, "")
	}
	if !hfRepoRe.MatchString(repo) {
		return modelMeta{}, errors.New(`Enter a valid HuggingFace repository ID like "HuggingFaceTB/SmolLM2-135M-Instruct".`)
	}
	base := "https://huggingface.co/" + repo + "/resolve/main/"
	work, err := newStage()
	if err != nil {
		return modelMeta{}, err
	}
	defer os.RemoveAll(work)
	stage := filepath.Join(work, "model")
	os.MkdirAll(stage, 0o755)
	p("connecting", "Fetching config & tokenizer from "+repo+"…", 0, 0)
	if err := download(ctx, base+"config.json", filepath.Join(stage, "config.json"), 4<<20, nil); err != nil {
		return modelMeta{}, err
	}
	// check the architecture before downloading hundreds of megabytes
	cfg, err := readJSONFile(filepath.Join(stage, "config.json"))
	if err != nil {
		return modelMeta{}, errors.New("config.json is not valid JSON.")
	}
	if _, err := parseArch(cfg, nil); err != nil {
		return modelMeta{}, err
	}
	if err := download(ctx, base+"tokenizer.json", filepath.Join(stage, "tokenizer.json"), 32<<20, nil); err != nil {
		return modelMeta{}, err
	}
	for _, opt := range []string{"tokenizer_config.json", "README.md"} {
		if download(ctx, base+opt, filepath.Join(stage, opt), 4<<20, nil) != nil {
			os.Remove(filepath.Join(stage, opt))
		}
	}
	p("downloading", "Downloading model.safetensors from "+repo+"…", 0, 0)
	if err := download(ctx, base+"model.safetensors", filepath.Join(stage, "model.safetensors"), maxModelBytes,
		func(d, t int64) { p("downloading", mbText("Downloading weights…", d, t), d, t) }); err != nil {
		return modelMeta{}, err
	}
	p("validating", "Validating tensors and registering model…", 0, 0)
	name := repo[strings.IndexByte(repo, '/')+1:]
	if strings.TrimSpace(customName) == "" {
		customName = name
	}
	m, err := finalizeStaged(stage, name, customName, "huggingface", "https://huggingface.co/"+repo)
	if err == nil {
		p("done", "Installed "+m.Name, 0, 0)
	}
	return m, err
}

/* ---------- HTTP ---------- */

func jsonOut(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func errOut(w http.ResponseWriter, code int, msg string) {
	jsonOut(w, code, map[string]string{"error": msg})
}

// sse starts a server-sent event stream and returns a function that sends one JSON event.
func sse(w http.ResponseWriter) func(v any) {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache, no-transform")
	h.Set("X-Accel-Buffering", "no")
	fl, _ := w.(http.Flusher)
	if fl != nil {
		fl.Flush()
	}
	return func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		if fl != nil {
			fl.Flush()
		}
	}
}

func registerLocal(mux *http.ServeMux) {
	mux.HandleFunc("/api/local/models", localList)
	mux.HandleFunc("/api/local/models/", localOne)
	mux.HandleFunc("/api/local/models/import", localImport)
	mux.HandleFunc("/api/local/models/upload", localUpload)
	mux.HandleFunc("/api/local/chat", localChat)
}

func localList(w http.ResponseWriter, r *http.Request) {
	snap := rt.snapshot()
	snap["supportedArchitectures"] = []string{"TinyLM", "Llama / SmolLM", "Qwen2 / Qwen3", "Mistral", "Gemma", "GPT-2"}
	snap["supportedQuantizations"] = []string{"INT8", "BF16", "FP16", "FP32"}
	jsonOut(w, 200, map[string]any{"models": listModels(), "runtime": snap})
}

// localOne serves /api/local/models/<id> (GET, DELETE) and /api/local/models/<id>/load (POST).
func localOne(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/api/local/models/")
	load := strings.HasSuffix(id, "/load")
	id = strings.TrimSuffix(id, "/load")
	dir, err := modelDir(id)
	if err != nil {
		errOut(w, 400, err.Error())
		return
	}
	if load && r.Method == "POST" {
		var body struct {
			Action string `json:"action"`
		}
		b, _ := io.ReadAll(io.LimitReader(r.Body, 1<<16))
		json.Unmarshal(b, &body)
		if body.Action == "unload" {
			if err := rt.unload(id); err != nil {
				errOut(w, 400, err.Error())
				return
			}
			jsonOut(w, 200, map[string]any{"id": id, "loaded": false, "ramBytes": 0, "loadMs": 0, "runtime": rt.snapshot()})
			return
		}
		if _, ok := findModel(id); !ok {
			errOut(w, 404, "Local model not found.")
			return
		}
		m, cold, err := rt.acquire(id, dir)
		if err != nil {
			errOut(w, 400, err.Error())
			return
		}
		rt.release(m)
		if cold == 0 {
			cold = m.loadMs
		}
		jsonOut(w, 200, map[string]any{"id": id, "loaded": true, "ramBytes": m.ram, "loadMs": cold, "contextLength": m.arch.seq, "runtime": rt.snapshot()})
		return
	}
	m, ok := findModel(id)
	switch {
	case load:
		errOut(w, 405, "Method not allowed.")
	case !ok:
		errOut(w, 404, "Local model not found.")
	case r.Method == "GET":
		cfg, _ := readJSONFile(filepath.Join(dir, "config.json"))
		if cfg == nil {
			cfg = map[string]any{}
		}
		jsonOut(w, 200, struct {
			modelMeta
			ConfigJSON map[string]any `json:"configJson"`
		}{m, cfg})
	case r.Method == "DELETE":
		if err := deleteModel(id); err != nil {
			errOut(w, 400, err.Error())
			return
		}
		w.WriteHeader(204)
	default:
		errOut(w, 405, "Method not allowed.")
	}
}

func localImport(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		errOut(w, 405, "Method not allowed.")
		return
	}
	var body struct {
		Source string `json:"source"`
		URL    string `json:"url"`
		Repo   string `json:"repo"`
		Name   string `json:"name"`
		Stream bool   `json:"stream"`
	}
	b, _ := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	if json.Unmarshal(b, &body) != nil {
		errOut(w, 400, "Invalid JSON body.")
		return
	}
	hf, name := body.Source == "huggingface", strings.TrimSpace(body.Name)
	target := strings.TrimSpace(body.URL)
	if hf {
		target = strings.TrimSpace(body.Repo)
	}
	if target == "" {
		if hf {
			errOut(w, 400, "Please enter a HuggingFace repository ID.")
		} else {
			errOut(w, 400, "Please enter a model .zip URL.")
		}
		return
	}
	run := func(p progressFn) (modelMeta, error) {
		if hf {
			return importHF(r.Context(), target, name, p)
		}
		return importZipURL(r.Context(), target, name, p)
	}
	if !body.Stream {
		m, err := run(func(string, string, int64, int64) { touch() })
		if err != nil {
			errOut(w, 400, err.Error())
			return
		}
		jsonOut(w, 201, map[string]any{"model": m})
		return
	}
	send := sse(w) // progress events while downloading and unpacking
	m, err := run(func(phase, msg string, done, total int64) {
		touch()
		ev := map[string]any{"type": "progress", "phase": phase, "message": msg}
		if phase == "downloading" && done > 0 {
			ev["downloadedBytes"], ev["totalBytes"] = done, total
		}
		send(ev)
	})
	if err != nil {
		send(map[string]any{"type": "error", "error": err.Error()})
		return
	}
	send(map[string]any{"type": "done", "model": m})
}

// localUpload installs a model sent by the browser: one .zip, or the loose model files together.
func localUpload(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		errOut(w, 405, "Method not allowed.")
		return
	}
	mr, err := r.MultipartReader()
	if err != nil {
		errOut(w, 400, "Upload failed.")
		return
	}
	work, err := newStage()
	if err != nil {
		errOut(w, 400, "Upload failed: "+err.Error())
		return
	}
	defer os.RemoveAll(work)
	stage := filepath.Join(work, "model")
	os.MkdirAll(stage, 0o755)
	var name, folder, zipName string
	var total int64
	loose := 0
	field := func(p io.Reader) string {
		b, _ := io.ReadAll(io.LimitReader(p, 512))
		return strings.TrimSpace(strings.ToValidUTF8(string(b), ""))
	}
	for {
		p, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			errOut(w, 400, "Upload failed: the connection was interrupted.")
			return
		}
		switch p.FormName() {
		case "name":
			name = field(p)
		case "folder":
			folder = field(p)
		case "files", "file":
			base := p.FileName() // already reduced to its last path element by the standard library
			ext := strings.ToLower(filepath.Ext(base))
			dest := ""
			switch {
			case base == "" || base == "." || base == ".." || strings.ContainsAny(base, `/\:`) || base == metaFile:
			case ext == ".zip" && zipName == "":
				zipName, dest = base, filepath.Join(work, "upload.zip")
			case modelExts[ext]:
				dest = filepath.Join(stage, base)
				loose++
			}
			if dest == "" {
				continue
			}
			f, err := os.OpenFile(dest, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
			if err != nil {
				errOut(w, 400, "Upload failed: "+err.Error())
				return
			}
			n, err := io.Copy(f, io.LimitReader(p, maxModelBytes-total+1))
			f.Close()
			touch()
			if total += n; total > maxModelBytes {
				errOut(w, 400, fmt.Sprintf("Uploaded files exceed the %d MB limit.", mb(maxModelBytes)))
				return
			}
			if err != nil {
				errOut(w, 400, "Upload failed: the connection was interrupted.")
				return
			}
		}
	}
	var m modelMeta
	switch {
	case loose == 0 && zipName == "":
		errOut(w, 400, "Select a .zip model archive or upload model.safetensors, config.json, and tokenizer.json together.")
		return
	case loose == 0: // a zipped model folder
		inner, err := extractZip(filepath.Join(work, "upload.zip"), stage)
		if err != nil {
			errOut(w, 400, err.Error())
			return
		}
		slug := name
		if slug == "" {
			slug = inner
		}
		if slug == "" {
			slug = strings.TrimSuffix(zipName, filepath.Ext(zipName))
		}
		m, err = finalizeStaged(stage, slug, name, "upload", zipName)
		if err != nil {
			errOut(w, 400, err.Error())
			return
		}
	default: // loose files
		slug := name
		if slug == "" {
			slug = folder
		}
		if slug == "" {
			slug = "uploaded-" + randHex()
		}
		if m, err = finalizeStaged(stage, slug, name, "upload", "local upload"); err != nil {
			errOut(w, 400, err.Error())
			return
		}
	}
	jsonOut(w, 201, map[string]any{"model": m})
}

// num reads an optional JSON number, falling back to def for anything else.
func num(v any, def float64) float64 {
	if f, ok := v.(float64); ok && f == f && f-f == 0 {
		return f
	}
	return def
}

// localChat is an OpenAI-compatible chat completion endpoint for the built-in engine (SSE by default).
func localChat(w http.ResponseWriter, r *http.Request) {
	fail := func(code int, msg string) {
		jsonOut(w, code, map[string]any{"error": map[string]string{"message": msg}})
	}
	if r.Method != "POST" {
		fail(405, "Method not allowed.")
		return
	}
	var body struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content any    `json:"content"`
		} `json:"messages"`
		Stream any `json:"stream"`
		Temp   any `json:"temperature"`
		TopP   any `json:"top_p"`
		TopK   any `json:"top_k"`
		Rep    any `json:"repetition_penalty"`
		Max    any `json:"max_tokens"`
		Format any `json:"prompt_format"`
		System any `json:"system_prompt"`
	}
	b, _ := io.ReadAll(io.LimitReader(r.Body, 50<<20))
	if json.Unmarshal(b, &body) != nil {
		fail(400, "Invalid JSON body.")
		return
	}
	id := strings.TrimPrefix(strings.TrimSpace(body.Model), "local:")
	if id == "" {
		fail(400, "Model ID is required.")
		return
	}
	dir, err := modelDir(id)
	if err != nil {
		fail(400, err.Error())
		return
	}
	meta, ok := findModel(id)
	if !ok {
		fail(404, fmt.Sprintf("Local model %q is not installed.", id))
		return
	}
	var msgs []chatMsg
	hasUser := false
	for _, m := range body.Messages {
		c, ok := m.Content.(string)
		if !ok || (m.Role != "user" && m.Role != "assistant" && m.Role != "system") {
			continue
		}
		msgs = append(msgs, chatMsg{m.Role, c})
		hasUser = hasUser || (m.Role == "user" && strings.TrimSpace(c) != "")
	}
	if !hasUser {
		fail(400, "At least one user message is required.")
		return
	}
	d := defaultLocalParams
	p := localParams{Temperature: num(body.Temp, d.Temperature), TopP: num(body.TopP, d.TopP), TopK: int(clampF(num(body.TopK, float64(d.TopK)), 0, 500)),
		RepeatPenalty: num(body.Rep, d.RepeatPenalty), MaxTokens: int(clampF(num(body.Max, float64(d.MaxTokens)), 0, 1<<20)), Format: "auto"}
	if f, _ := body.Format.(string); f == "qa" || f == "chat" || f == "chatml" || f == "raw" {
		p.Format = f
	}
	if s, ok := body.System.(string); ok {
		p.System = truncate(s, 16000)
	}
	cid, created := "chatcmpl-local-"+strconv.FormatInt(time.Now().UnixMilli(), 36), time.Now().Unix()

	if body.Stream == false { // one JSON answer instead of a stream
		var content, reasoning strings.Builder
		st, err := rt.generate(r.Context(), id, dir, msgs, p, func(c, rs string) { content.WriteString(c); reasoning.WriteString(rs) })
		if err != nil {
			fail(500, err.Error())
			return
		}
		if st == nil {
			return // cancelled
		}
		st.ModelName = meta.Name
		msg := map[string]any{"role": "assistant", "content": content.String()}
		if reasoning.Len() > 0 {
			msg["reasoning"] = reasoning.String()
		}
		jsonOut(w, 200, map[string]any{"id": cid, "object": "chat.completion", "created": created, "model": "local:" + id,
			"choices":        []any{map[string]any{"index": 0, "message": msg, "finish_reason": st.FinishReason}},
			"usage":          map[string]int{"prompt_tokens": st.PromptTokens, "completion_tokens": st.CompletionTokens, "total_tokens": st.PromptTokens + st.CompletionTokens},
			"nanochat_stats": st})
		return
	}

	send := sse(w)
	chunk := func(delta map[string]string, finish any, st *genStats) {
		o := map[string]any{"id": cid, "object": "chat.completion.chunk", "created": created, "model": "local:" + id,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
		if st != nil {
			o["nanochat_stats"] = st
		}
		send(o)
	}
	st, err := rt.generate(r.Context(), id, dir, msgs, p, func(c, rs string) {
		delta := map[string]string{}
		if c != "" {
			delta["content"] = c
		}
		if rs != "" {
			delta["reasoning"] = rs
		}
		chunk(delta, nil, nil)
	})
	if r.Context().Err() != nil {
		return
	}
	if err != nil {
		send(map[string]any{"error": map[string]string{"message": err.Error()}})
		return
	}
	if st != nil {
		st.ModelName = meta.Name
		chunk(map[string]string{}, st.FinishReason, st)
	}
	io.WriteString(w, "data: [DONE]\n\n")
}
