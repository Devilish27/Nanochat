package main

// Built-in CPU inference for small language models stored as .safetensors: TinyLM, the Llama family
// (Llama, SmolLM, Mistral, Qwen2/3, Gemma) and GPT-2. One token is processed at a time with a key/value
// cache, so generating a token costs one pass through the network. No GPU and no C code are involved.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	maxRuntimeSeq = 2048       // context is capped here even if the model allows more
	maxCacheRAM   = 1200 << 20 // loaded models are evicted (least recently used first) above this
	maxLoadBytes  = 4 << 30    // refuse weight files the CPU engine could not run in any reasonable time
)

type archCfg struct {
	family, modelType, label                                    string
	vocab, d, layers, heads, kvHeads, headDim, mlp, maxSeq, seq int
	theta, eps                                                  float64
	qkNorm, tie, gemma                                          bool
}

// cfgNum returns the first of the keys that is set, or def.
func cfgNum(m map[string]any, def float64, keys ...string) float64 {
	for _, k := range keys {
		if v, ok := m[k]; ok && v != nil {
			if f, ok := v.(float64); ok {
				return f
			}
			return math.NaN()
		}
	}
	return def
}

func truthy(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case float64:
		return x != 0
	case string:
		return x != ""
	}
	return v != nil
}

// parseArch reads config.json (TinyLM's nested model_cfg, or the usual HuggingFace fields). The tensor
// names, when known, settle details the config leaves out.
func parseArch(c map[string]any, names map[string]stTensor) (archCfg, error) {
	has := func(n string) bool { _, ok := names[n]; return ok }
	nan := math.NaN()
	var a archCfg
	var dims []float64
	mt := ""
	if s, ok := c["model_type"].(string); ok {
		mt = strings.ToLower(s)
	}
	llama := map[string]bool{"llama": true, "mistral": true, "qwen2": true, "qwen3": true, "smollm": true, "gemma": true, "gemma2": true}
	if mc, ok := c["model_cfg"].(map[string]any); ok {
		d, heads := cfgNum(mc, nan, "d_model"), cfgNum(mc, nan, "n_heads")
		dims = []float64{cfgNum(mc, nan, "vocab_size"), d, cfgNum(mc, nan, "n_layers"), heads, cfgNum(mc, heads, "n_kv_heads"),
			cfgNum(mc, d/heads, "head_dim"), cfgNum(mc, nan, "mlp_hidden"), cfgNum(mc, 512, "max_seq_len")}
		a = archCfg{family: "tinylm", modelType: "tinylm", label: "TinyLM", theta: cfgNum(mc, 10000, "rope_theta"), eps: cfgNum(mc, 1e-5, "rms_eps"),
			qkNorm: truthy(mc["qk_norm"]) || has("blocks.0.attn.q_norm.weight"), tie: mc["tie_embeddings"] != false}
		if s, ok := c["arch"].(string); ok {
			if l := strings.TrimSpace(strings.SplitN(s, "(", 2)[0]); l != "" {
				a.label = l
			}
		}
	} else if llama[mt] || has("model.layers.0.self_attn.q_proj.weight") {
		d, heads := cfgNum(c, nan, "hidden_size"), cfgNum(c, nan, "num_attention_heads")
		dims = []float64{cfgNum(c, nan, "vocab_size"), d, cfgNum(c, nan, "num_hidden_layers"), heads, cfgNum(c, heads, "num_key_value_heads"),
			cfgNum(c, d/heads, "head_dim"), cfgNum(c, nan, "intermediate_size"), cfgNum(c, 2048, "max_position_embeddings", "max_seq_len")}
		a = archCfg{family: "llama", modelType: mt, label: strings.ToUpper(mt), theta: cfgNum(c, 10000, "rope_theta"), eps: cfgNum(c, 1e-5, "rms_norm_eps"),
			qkNorm: mt == "qwen3" || has("model.layers.0.self_attn.q_norm.weight"), tie: truthy(c["tie_word_embeddings"]), gemma: mt == "gemma" || mt == "gemma2"}
		if a.modelType == "" {
			a.modelType, a.label = "llama", "LlamaForCausalLM"
		}
		if arr, ok := c["architectures"].([]any); ok && len(arr) > 0 {
			if s, ok := arr[0].(string); ok && s != "" {
				a.label = s
			}
		}
	} else if mt == "gpt2" || has("h.0.attn.c_attn.weight") || has("transformer.h.0.attn.c_attn.weight") {
		d, heads := cfgNum(c, nan, "n_embd", "hidden_size"), cfgNum(c, nan, "n_head", "num_attention_heads")
		dims = []float64{cfgNum(c, 50257, "vocab_size"), d, cfgNum(c, nan, "n_layer", "num_hidden_layers"), heads, heads,
			d / heads, cfgNum(c, d*4, "n_inner"), cfgNum(c, 1024, "n_positions", "n_ctx")}
		a = archCfg{family: "gpt2", modelType: "gpt2", label: "GPT2LMHeadModel", theta: 10000, eps: cfgNum(c, 1e-5, "layer_norm_epsilon"), tie: true}
	} else {
		if mt == "" {
			mt = "unknown"
		}
		return a, fmt.Errorf("Unsupported model architecture %q. Supported: TinyLM, Llama, SmolLM, Qwen2, Qwen3, Mistral, Gemma, GPT-2 (safetensors).", mt)
	}
	for _, v := range dims {
		if !(v >= 1 && v < 1e8) || v != math.Trunc(v) {
			return a, fmt.Errorf("Incomplete or invalid %s config.json dimensions.", a.modelType)
		}
	}
	a.vocab, a.d, a.layers, a.heads, a.kvHeads, a.headDim, a.mlp, a.maxSeq = int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4]), int(dims[5]), int(dims[6]), int(dims[7])
	if a.heads%a.kvHeads != 0 || a.maxSeq < 16 || (a.family != "gpt2" && a.headDim%2 != 0) || !(a.theta > 0) || !(a.eps >= 0) {
		return a, fmt.Errorf("Incomplete or invalid %s config.json dimensions.", a.modelType)
	}
	if a.seq = a.maxSeq; a.seq > maxRuntimeSeq {
		a.seq = maxRuntimeSeq
	}
	return a, nil
}

type llamaBlock struct {
	attnNorm, qNorm, kNorm, mlpNorm []float32
	q, k, v, o, gate, up, down      *linW
}

type gptBlock struct {
	ln1W, ln1B, ln2W, ln2B  []float32
	attn, proj, fc, mlpProj *linW
}

type localModel struct {
	id       string
	arch     archCfg
	tok      *tokenizer
	ram      int64
	loadMs   int64
	lastUsed int64 // unix nanoseconds, atomic
	active   int32 // generations running or waiting, atomic
	mu       sync.Mutex
	rng      *rand.Rand
	threads  int

	tokEmb, lmHead       *linW
	posEmb, normW, normB []float32
	lb                   []llamaBlock
	gb                   []gptBlock
	cos, sin             []float32
	kc, vc               [][]float32 // key/value cache per layer
	x, xn, q, k, v, qkv  []float32   // scratch buffers, reused for every token
	att, proj, gate, up  []float32
	scores, logits       []float32
}

// ldr remembers the first load error, so the loading code below can stay a plain list of tensors.
type ldr struct {
	r   *stReader
	err error
}

func (l *ldr) f32(name string, required bool) []float32 {
	if l.err != nil {
		return nil
	}
	v, err := l.r.f32(name, required)
	l.err = err
	return v
}

func (l *ldr) lin(prefix string, transpose, withBias bool) *linW {
	if l.err != nil {
		return nil
	}
	v, err := l.r.linear(prefix, transpose, withBias)
	l.err = err
	return v
}

func readJSONFile(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil || m == nil {
		return nil, errors.New("not a JSON object")
	}
	return m, nil
}

func loadModel(id, dir string) (m *localModel, err error) {
	defer func() {
		if r := recover(); r != nil {
			m, err = nil, fmt.Errorf("Could not load this model: %v", r)
		}
	}()
	t0 := time.Now()
	cfg, err := readJSONFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, errors.New("config.json is missing or not valid JSON.")
	}
	rd, err := openSafetensors(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	defer rd.close()
	if rd.info.FileSize > maxLoadBytes {
		return nil, errors.New("This model is too large for the built-in CPU engine (limit 4 GB of weights).")
	}
	a, err := parseArch(cfg, rd.info.Tensors)
	if err != nil {
		return nil, err
	}
	tk, err := newTokenizer(dir, cfg)
	if err != nil {
		return nil, err
	}
	m = &localModel{id: id, arch: a, tok: tk, rng: rand.New(rand.NewSource(time.Now().UnixNano())), threads: runtime.NumCPU()}
	if m.threads > 16 {
		m.threads = 16
	}
	l := &ldr{r: rd}
	switch a.family {
	case "tinylm":
		m.tokEmb = l.lin("tok_emb", false, false)
		for i := 0; i < a.layers; i++ {
			p := fmt.Sprintf("blocks.%d", i)
			m.lb = append(m.lb, llamaBlock{
				attnNorm: l.f32(p+".attn_norm.weight", true),
				q:        l.lin(p+".attn.q_proj", false, false), k: l.lin(p+".attn.k_proj", false, false),
				v: l.lin(p+".attn.v_proj", false, false), o: l.lin(p+".attn.o_proj", false, false),
				qNorm: l.f32(p+".attn.q_norm.weight", false), kNorm: l.f32(p+".attn.k_norm.weight", false),
				mlpNorm: l.f32(p+".mlp_norm.weight", true),
				gate:    l.lin(p+".mlp.gate_proj", false, false), up: l.lin(p+".mlp.up_proj", false, false), down: l.lin(p+".mlp.down_proj", false, false),
			})
		}
		m.normW = l.f32("norm.weight", true)
		switch {
		case rd.has("lm_head.weight"):
			m.lmHead = l.lin("lm_head", false, false)
		case m.tokEmb != nil && m.tokEmb.q != nil && rd.has("lm_head.weight.scale"): // tied int8 weights, own scale
			cp := *m.tokEmb
			cp.scale = l.f32("lm_head.weight.scale", true)
			m.lmHead = &cp
		default:
			m.lmHead = m.tokEmb
		}
	case "llama":
		emb, layers, norm := "embed_tokens", "layers", "norm.weight"
		if rd.has("model.embed_tokens.weight") {
			emb = "model.embed_tokens"
		}
		if rd.has("model.layers.0.self_attn.q_proj.weight") {
			layers = "model.layers"
		}
		if rd.has("model.norm.weight") {
			norm = "model.norm.weight"
		}
		m.tokEmb = l.lin(emb, false, false)
		for i := 0; i < a.layers; i++ {
			p := fmt.Sprintf("%s.%d", layers, i)
			m.lb = append(m.lb, llamaBlock{
				attnNorm: l.f32(p+".input_layernorm.weight", true),
				q:        l.lin(p+".self_attn.q_proj", false, true), k: l.lin(p+".self_attn.k_proj", false, true),
				v: l.lin(p+".self_attn.v_proj", false, true), o: l.lin(p+".self_attn.o_proj", false, true),
				qNorm: l.f32(p+".self_attn.q_norm.weight", false), kNorm: l.f32(p+".self_attn.k_norm.weight", false),
				mlpNorm: l.f32(p+".post_attention_layernorm.weight", true),
				gate:    l.lin(p+".mlp.gate_proj", false, true), up: l.lin(p+".mlp.up_proj", false, true), down: l.lin(p+".mlp.down_proj", false, true),
			})
		}
		m.normW = l.f32(norm, true)
		if rd.has("lm_head.weight") {
			m.lmHead = l.lin("lm_head", false, false)
		} else {
			m.lmHead = m.tokEmb
		}
	default: // gpt2: Conv1D layers are stored transposed
		root := ""
		if rd.has("transformer.wte.weight") {
			root = "transformer."
		}
		m.tokEmb = l.lin(root+"wte", false, false)
		m.posEmb = l.f32(root+"wpe.weight", true)
		for i := 0; i < a.layers; i++ {
			p := fmt.Sprintf("%sh.%d", root, i)
			m.gb = append(m.gb, gptBlock{
				ln1W: l.f32(p+".ln_1.weight", true), ln1B: l.f32(p+".ln_1.bias", false),
				attn: l.lin(p+".attn.c_attn", true, true), proj: l.lin(p+".attn.c_proj", true, true),
				ln2W: l.f32(p+".ln_2.weight", true), ln2B: l.f32(p+".ln_2.bias", false),
				fc: l.lin(p+".mlp.c_fc", true, true), mlpProj: l.lin(p+".mlp.c_proj", true, true),
			})
		}
		m.normW, m.normB = l.f32(root+"ln_f.weight", true), l.f32(root+"ln_f.bias", false)
		if rd.has("lm_head.weight") {
			m.lmHead = l.lin("lm_head", false, false)
		} else {
			m.lmHead = m.tokEmb
		}
	}
	if l.err != nil {
		return nil, l.err
	}

	// Check every shape now: a wrong size must be a clear error here, not a crash while generating.
	qDim, kvDim := a.heads*a.headDim, a.kvHeads*a.headDim
	lin := func(w *linW, out, in int, what string) {
		if err == nil && (w == nil || w.in != in || (out > 0 && w.out != out) || w.out < 1 || (w.q != nil && len(w.scale) != w.out) || (w.bias != nil && len(w.bias) != w.out)) {
			err = fmt.Errorf("Tensor shape mismatch in %s (does config.json belong to these weights?).", what)
		}
	}
	vec := func(v []float32, n int, optional bool, what string) {
		if err == nil && len(v) != n && !(optional && v == nil) {
			err = fmt.Errorf("Tensor shape mismatch in %s (does config.json belong to these weights?).", what)
		}
	}
	lin(m.tokEmb, 0, a.d, "token embeddings")
	lin(m.lmHead, 0, a.d, "lm_head")
	vec(m.normW, a.d, false, "final norm")
	vec(m.normB, a.d, true, "final norm")
	for i, b := range m.lb {
		w := fmt.Sprintf("layer %d", i)
		vec(b.attnNorm, a.d, false, w)
		vec(b.mlpNorm, a.d, false, w)
		vec(b.qNorm, a.headDim, true, w)
		vec(b.kNorm, a.headDim, true, w)
		lin(b.q, qDim, a.d, w)
		lin(b.k, kvDim, a.d, w)
		lin(b.v, kvDim, a.d, w)
		lin(b.o, a.d, qDim, w)
		lin(b.gate, a.mlp, a.d, w)
		lin(b.up, a.mlp, a.d, w)
		lin(b.down, a.d, a.mlp, w)
	}
	for i, b := range m.gb {
		w := fmt.Sprintf("layer %d", i)
		vec(b.ln1W, a.d, false, w)
		vec(b.ln1B, a.d, true, w)
		vec(b.ln2W, a.d, false, w)
		vec(b.ln2B, a.d, true, w)
		lin(b.attn, 3*a.d, a.d, w)
		lin(b.proj, a.d, a.d, w)
		lin(b.fc, a.mlp, a.d, w)
		lin(b.mlpProj, a.d, a.mlp, w)
	}
	if err == nil && a.family == "gpt2" && (qDim != a.d || len(m.posEmb) < a.seq*a.d) {
		err = errors.New("Tensor shape mismatch in position embeddings (does config.json belong to these weights?).")
	}
	if err != nil {
		return nil, err
	}

	half := a.headDim / 2
	m.cos, m.sin = make([]float32, a.seq*half), make([]float32, a.seq*half)
	if a.family != "gpt2" { // rotary position tables
		for pos := 0; pos < a.seq; pos++ {
			for i := 0; i < half; i++ {
				ang := float64(pos) / math.Pow(a.theta, float64(2*i)/float64(a.headDim))
				m.cos[pos*half+i], m.sin[pos*half+i] = float32(math.Cos(ang)), float32(math.Sin(ang))
			}
		}
	}
	for i := 0; i < a.layers; i++ {
		m.kc, m.vc = append(m.kc, make([]float32, a.seq*kvDim)), append(m.vc, make([]float32, a.seq*kvDim))
	}
	f := func(n int) []float32 { return make([]float32, n) }
	m.x, m.xn, m.q, m.k, m.v, m.att, m.proj, m.gate = f(a.d), f(a.d), f(qDim), f(kvDim), f(kvDim), f(qDim), f(a.d), f(a.mlp)
	m.qkv, m.up, m.scores, m.logits = f(3*a.d), f(a.mlp), f(a.seq), f(m.lmHead.out)
	m.ram = rd.alloc + int64(a.layers)*2*int64(a.seq)*int64(kvDim)*4 + int64(len(m.cos))*8
	m.loadMs = time.Since(t0).Milliseconds()
	atomic.StoreInt64(&m.lastUsed, time.Now().UnixNano())
	return m, nil
}

/* ---------- numeric kernels ---------- */

// Four running sums keep the CPU busy on independent additions; Go does not vectorize loops by itself.
func dotF32(w, x []float32) float32 {
	w = w[:len(x)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+8 <= len(x); i += 8 {
		a, b := w[i:i+8:i+8], x[i:i+8:i+8]
		s0 += a[0]*b[0] + a[4]*b[4]
		s1 += a[1]*b[1] + a[5]*b[5]
		s2 += a[2]*b[2] + a[6]*b[6]
		s3 += a[3]*b[3] + a[7]*b[7]
	}
	for ; i < len(x); i++ {
		s0 += w[i] * x[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func dotI8(w []int8, x []float32) float32 {
	w = w[:len(x)]
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+8 <= len(x); i += 8 {
		a, b := w[i:i+8:i+8], x[i:i+8:i+8]
		s0 += float32(a[0])*b[0] + float32(a[4])*b[4]
		s1 += float32(a[1])*b[1] + float32(a[5])*b[5]
		s2 += float32(a[2])*b[2] + float32(a[6])*b[6]
		s3 += float32(a[3])*b[3] + float32(a[7])*b[7]
	}
	for ; i < len(x); i++ {
		s0 += float32(w[i]) * x[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func matvecRows(out []float32, lw *linW, x []float32, from, to int) {
	in := lw.in
	for r := from; r < to; r++ {
		var s float32
		if lw.q != nil {
			s = dotI8(lw.q[r*in:(r+1)*in], x) * lw.scale[r]
		} else {
			s = dotF32(lw.w[r*in:(r+1)*in], x)
		}
		if lw.bias != nil {
			s += lw.bias[r]
		}
		out[r] = s
	}
}

// linear computes out = W x (+ bias), splitting the rows of large layers across CPU cores.
func (m *localModel) linear(out []float32, lw *linW, x []float32) {
	n := m.threads
	if n > lw.out {
		n = lw.out
	}
	if n <= 1 || lw.out*lw.in < 1<<16 {
		matvecRows(out, lw, x, 0, lw.out)
		return
	}
	var wg sync.WaitGroup
	step := (lw.out + n - 1) / n
	for from := 0; from < lw.out; from += step {
		to := from + step
		if to > lw.out {
			to = lw.out
		}
		wg.Add(1)
		go func(from, to int) {
			defer wg.Done()
			matvecRows(out, lw, x, from, to)
		}(from, to)
	}
	wg.Wait()
}

func rmsNorm(out, x, w []float32, eps float64, gemma bool) {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+eps))
	if gemma {
		for i, v := range x {
			out[i] = v * inv * (1 + w[i])
		}
		return
	}
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
}

func layerNorm(out, x, w, b []float32, eps float64) {
	var mean, vr float64
	for _, v := range x {
		mean += float64(v)
	}
	mean /= float64(len(x))
	for _, v := range x {
		d := float64(v) - mean
		vr += d * d
	}
	inv := 1 / math.Sqrt(vr/float64(len(x))+eps)
	for i, v := range x {
		out[i] = float32((float64(v) - mean) * inv * float64(w[i]))
		if b != nil {
			out[i] += b[i]
		}
	}
}

// gelu is the tanh approximation used by GPT-2 and Gemma.
func gelu(g float32) float32 {
	x := float64(g)
	return float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
}

func (m *localModel) rope(v []float32, heads, pos int) {
	hd := m.arch.headDim
	half := hd / 2
	cos, sin := m.cos[pos*half:(pos+1)*half], m.sin[pos*half:(pos+1)*half]
	for h := 0; h < heads; h++ {
		a, b := v[h*hd:h*hd+half], v[h*hd+half:h*hd+2*half]
		for i := range a {
			x1, x2 := a[i], b[i]
			a[i], b[i] = x1*cos[i]-x2*sin[i], x2*cos[i]+x1*sin[i]
		}
	}
}

// attend stores this position's key and value, then mixes the values of all positions seen so far.
func (m *localModel) attend(layer, pos int) {
	a := m.arch
	hd, kvDim, mul := a.headDim, a.kvHeads*a.headDim, a.heads/a.kvHeads
	inv := float32(1 / math.Sqrt(float64(hd)))
	kc, vc := m.kc[layer], m.vc[layer]
	copy(kc[pos*kvDim:], m.k)
	copy(vc[pos*kvDim:], m.v)
	for h := 0; h < a.heads; h++ {
		q, o, off := m.q[h*hd:(h+1)*hd], m.att[h*hd:(h+1)*hd], (h/mul)*hd
		maxS := float32(math.Inf(-1))
		for t := 0; t <= pos; t++ {
			kk := kc[t*kvDim+off : t*kvDim+off+hd]
			var dot float32
			for d, qv := range q {
				dot += qv * kk[d]
			}
			dot *= inv
			m.scores[t] = dot
			if dot > maxS {
				maxS = dot
			}
		}
		var sum float32
		for t := 0; t <= pos; t++ {
			e := float32(math.Exp(float64(m.scores[t] - maxS)))
			m.scores[t] = e
			sum += e
		}
		for d := range o {
			o[d] = 0
		}
		for t := 0; t <= pos; t++ {
			wt, vv := m.scores[t]/sum, vc[t*kvDim+off:t*kvDim+off+hd]
			for d := range o {
				o[d] += wt * vv[d]
			}
		}
	}
}

func addTo(x, y []float32) {
	for i := range x {
		x[i] += y[i]
	}
}

// forward feeds one token at position pos and returns the next-token logits (nil unless wanted).
func (m *localModel) forward(token, pos int, wantLogits bool) []float32 {
	a := m.arch
	x, xn := m.x, m.xn
	if token < 0 || token >= m.tokEmb.out {
		token = 0
	}
	if e := m.tokEmb; e.q != nil {
		sc := e.scale[token]
		if a.gemma {
			sc *= float32(math.Sqrt(float64(a.d)))
		}
		for i, v := range e.q[token*a.d : (token+1)*a.d] {
			x[i] = float32(v) * sc
		}
	} else {
		copy(x, e.w[token*a.d:(token+1)*a.d])
		if a.gemma {
			sc := float32(math.Sqrt(float64(a.d)))
			for i := range x {
				x[i] *= sc
			}
		}
	}

	if a.family == "gpt2" {
		addTo(x, m.posEmb[pos*a.d:(pos+1)*a.d])
		for l := range m.gb {
			b := &m.gb[l]
			layerNorm(xn, x, b.ln1W, b.ln1B, a.eps)
			m.linear(m.qkv, b.attn, xn)
			copy(m.q, m.qkv[:a.d])
			copy(m.k, m.qkv[a.d:2*a.d])
			copy(m.v, m.qkv[2*a.d:])
			m.attend(l, pos)
			m.linear(m.proj, b.proj, m.att)
			addTo(x, m.proj)
			layerNorm(xn, x, b.ln2W, b.ln2B, a.eps)
			m.linear(m.gate, b.fc, xn)
			for i, g := range m.gate {
				m.gate[i] = gelu(g)
			}
			m.linear(m.proj, b.mlpProj, m.gate)
			addTo(x, m.proj)
		}
		if !wantLogits {
			return nil
		}
		layerNorm(xn, x, m.normW, m.normB, a.eps)
		m.linear(m.logits, m.lmHead, xn)
		return m.logits
	}

	hd := a.headDim
	for l := range m.lb {
		b := &m.lb[l]
		rmsNorm(xn, x, b.attnNorm, a.eps, a.gemma)
		m.linear(m.q, b.q, xn)
		m.linear(m.k, b.k, xn)
		m.linear(m.v, b.v, xn)
		if b.qNorm != nil {
			for h := 0; h < a.heads; h++ {
				rmsNorm(m.q[h*hd:(h+1)*hd], m.q[h*hd:(h+1)*hd], b.qNorm, a.eps, a.gemma)
			}
		}
		if b.kNorm != nil {
			for h := 0; h < a.kvHeads; h++ {
				rmsNorm(m.k[h*hd:(h+1)*hd], m.k[h*hd:(h+1)*hd], b.kNorm, a.eps, a.gemma)
			}
		}
		m.rope(m.q, a.heads, pos)
		m.rope(m.k, a.kvHeads, pos)
		m.attend(l, pos)
		m.linear(m.proj, b.o, m.att)
		addTo(x, m.proj)

		rmsNorm(xn, x, b.mlpNorm, a.eps, a.gemma)
		m.linear(m.gate, b.gate, xn)
		m.linear(m.up, b.up, xn)
		if a.gemma {
			for i, g := range m.gate {
				m.gate[i] = gelu(g) * m.up[i]
			}
		} else { // SwiGLU: silu(gate) * up
			for i, g := range m.gate {
				m.gate[i] = g / (1 + float32(math.Exp(float64(-g)))) * m.up[i]
			}
		}
		m.linear(m.proj, b.down, m.gate)
		addTo(x, m.proj)
	}
	if !wantLogits {
		return nil
	}
	rmsNorm(xn, x, m.normW, a.eps, a.gemma)
	m.linear(m.logits, m.lmHead, xn)
	return m.logits
}

/* ---------- sampling ---------- */

type sampling struct {
	temp, topP, penalty float64
	topK                int
}

// sampleNext applies the repetition penalty (last 96 tokens), then greedy, top-k and top-p sampling.
func sampleNext(logits []float32, recent []int, p sampling, rng *rand.Rand) int {
	vocab := len(logits)
	if p.penalty > 1.0001 && len(recent) > 0 {
		start := len(recent) - 96
		if start < 0 {
			start = 0
		}
		seen := map[int]bool{}
		for _, id := range recent[start:] {
			if id >= 0 && id < vocab && !seen[id] {
				seen[id] = true
				if v := float64(logits[id]); v > 0 {
					logits[id] = float32(v / p.penalty)
				} else {
					logits[id] = float32(v * p.penalty)
				}
			}
		}
	}
	if p.temp <= 0.01 || p.topK == 1 {
		best := 0
		for i := 1; i < vocab; i++ {
			if logits[i] > logits[best] {
				best = i
			}
		}
		return best
	}
	k := p.topK
	if k <= 0 {
		k = 256
	}
	if k > vocab {
		k = vocab
	}
	// keep the k best candidates without sorting the whole vocabulary
	ids, vals := make([]int, k), make([]float32, k)
	negInf := float32(math.Inf(-1))
	for i := range vals {
		vals[i] = negInf
	}
	minV, minP := negInf, 0
	for i, v := range logits {
		if v > minV {
			ids[minP], vals[minP] = i, v
			minV, minP = vals[0], 0
			for j := 1; j < k; j++ {
				if vals[j] < minV {
					minV, minP = vals[j], j
				}
			}
		}
	}
	order := make([]int, k)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return vals[order[a]] > vals[order[b]] })
	probs, sum, top := make([]float64, k), 0.0, float64(vals[order[0]])
	for i, o := range order {
		probs[i] = math.Exp((float64(vals[o]) - top) / p.temp)
		sum += probs[i]
	}
	cut, kept := k, sum
	if p.topP > 0 && p.topP < 0.9999 { // nucleus cutoff
		acc := 0.0
		for i := 0; i < k; i++ {
			if acc += probs[i]; acc >= p.topP*sum {
				cut, kept = i+1, acc
				break
			}
		}
	}
	r := rng.Float64() * kept
	for i := 0; i < cut; i++ {
		if r -= probs[i]; r <= 0 {
			return ids[order[i]]
		}
	}
	return ids[order[0]]
}

/* ---------- runtime: loaded models and generation ---------- */

type localRuntime struct {
	mu     sync.Mutex
	loaded map[string]*localModel
}

var rt = &localRuntime{loaded: map[string]*localModel{}}

// info never waits: while a model is being loaded it simply reports what it can.
func (rt *localRuntime) info(id string) (loaded bool, ram, loadMs int64, active bool) {
	if !rt.mu.TryLock() {
		return
	}
	defer rt.mu.Unlock()
	if m := rt.loaded[id]; m != nil {
		return true, m.ram, m.loadMs, atomic.LoadInt32(&m.active) > 0
	}
	return
}

func (rt *localRuntime) snapshot() map[string]any {
	ids, total := []string{}, int64(0)
	if rt.mu.TryLock() {
		for id, m := range rt.loaded {
			ids = append(ids, id)
			total += m.ram
		}
		rt.mu.Unlock()
	}
	sort.Strings(ids)
	return map[string]any{"loadedIds": ids, "totalRamBytes": total, "maxRamBytes": int64(maxCacheRAM)}
}

// evict drops idle models, least recently used first, when more than two are loaded or RAM is over the limit.
func (rt *localRuntime) evict(except string) {
	var total int64
	for _, m := range rt.loaded {
		total += m.ram
	}
	if total <= maxCacheRAM && len(rt.loaded) <= 2 {
		return
	}
	var c []*localModel
	for _, m := range rt.loaded {
		if m.id != except && atomic.LoadInt32(&m.active) == 0 {
			c = append(c, m)
		}
	}
	sort.Slice(c, func(i, j int) bool { return atomic.LoadInt64(&c[i].lastUsed) < atomic.LoadInt64(&c[j].lastUsed) })
	for _, m := range c {
		if float64(total) <= maxCacheRAM*0.7 && len(rt.loaded) <= 2 {
			break
		}
		delete(rt.loaded, m.id)
		total -= m.ram
	}
	go debug.FreeOSMemory()
}

// acquire returns the loaded model (loading it if needed) and marks it busy until release.
func (rt *localRuntime) acquire(id, dir string) (*localModel, int64, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if m := rt.loaded[id]; m != nil {
		atomic.StoreInt64(&m.lastUsed, time.Now().UnixNano())
		atomic.AddInt32(&m.active, 1)
		return m, 0, nil
	}
	rt.evict(id)
	m, err := loadModel(id, dir)
	if err != nil {
		go debug.FreeOSMemory()
		return nil, 0, err
	}
	rt.loaded[id] = m
	rt.evict(id)
	atomic.AddInt32(&m.active, 1)
	return m, m.loadMs, nil
}

func (rt *localRuntime) release(m *localModel) {
	atomic.StoreInt64(&m.lastUsed, time.Now().UnixNano())
	atomic.AddInt32(&m.active, -1)
}

func (rt *localRuntime) unload(id string) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if m := rt.loaded[id]; m != nil {
		if atomic.LoadInt32(&m.active) > 0 {
			return errors.New("Cannot unload a model while it is generating a reply.")
		}
		delete(rt.loaded, id)
		go debug.FreeOSMemory()
	}
	return nil
}

type localParams struct {
	Temperature, TopP, RepeatPenalty float64
	TopK, MaxTokens                  int
	Format, System                   string
}

var defaultLocalParams = localParams{Temperature: 0.5, TopP: 0.9, TopK: 40, RepeatPenalty: 1.1, MaxTokens: 160, Format: "auto"}

type genStats struct {
	PromptTokens     int     `json:"promptTokens"`
	CompletionTokens int     `json:"completionTokens"`
	ContextLimit     int     `json:"contextLimit"`
	LoadMs           int64   `json:"loadMs"`
	PrefillMs        int64   `json:"prefillMs"`
	GenMs            int64   `json:"genMs"`
	TokensPerSec     float64 `json:"tokensPerSec"`
	ResolvedFormat   string  `json:"resolvedFormat"`
	ModelName        string  `json:"modelName"`
	FinishReason     string  `json:"-"` // stop, length or abort
}

func clampF(v, lo, hi float64) float64 {
	if v != v || v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// generate runs one reply. emit receives text as soon as it exists; stats are nil when the request was
// cancelled before the first token.
func (rt *localRuntime) generate(ctx context.Context, id, dir string, msgs []chatMsg, p localParams, emit func(content, reasoning string)) (st *genStats, err error) {
	m, coldMs, err := rt.acquire(id, dir)
	if err != nil {
		return nil, err
	}
	defer rt.release(m)
	m.mu.Lock() // one generation per model at a time: the cache and scratch buffers are shared
	defer m.mu.Unlock()
	defer func() {
		if r := recover(); r != nil {
			st, err = nil, fmt.Errorf("Local engine error: %v", r)
		}
	}()
	if ctx.Err() != nil {
		return nil, nil
	}

	pr := buildPrompt(msgs, p.System, p.Format, m.arch.modelType, m.tok)
	addBos := m.tok.addBos && m.tok.bosID >= 0 && !strings.HasPrefix(pr.text, "<|begin_of_text|>") && !strings.HasPrefix(pr.text, "<s>")
	ids := m.tok.encode(pr.text, addBos)
	if len(ids) == 0 {
		ids = []int{0}
		if m.tok.bosID >= 0 {
			ids[0] = m.tok.bosID
		}
	}

	// Budget: always leave room for the reply, dropping the oldest prompt tokens if the chat is too long.
	seq := m.arch.seq
	maxNew := p.MaxTokens
	if maxNew <= 0 {
		maxNew = 160
	}
	if maxNew > seq-16 {
		maxNew = seq - 16
	}
	if maxNew < 8 {
		maxNew = 8
	}
	reserve := seq * 35 / 100
	if reserve < 48 {
		reserve = 48
	}
	if reserve > maxNew {
		reserve = maxNew
	}
	maxPrompt := seq - reserve
	if maxPrompt < 16 {
		maxPrompt = 16
	}
	if maxPrompt > seq-1 {
		maxPrompt = seq - 1
	}
	if len(ids) > maxPrompt {
		ids = ids[len(ids)-maxPrompt:]
	}
	if maxNew > seq-len(ids) {
		maxNew = seq - len(ids)
	}
	sp := sampling{temp: clampF(p.Temperature, 0, 2), topP: clampF(p.TopP, 0.05, 1), penalty: clampF(p.RepeatPenalty, 1, 2), topK: int(clampF(float64(p.TopK), 0, 500))}

	t0 := time.Now()
	var logits []float32
	for i, tok := range ids { // read the prompt into the key/value cache
		if ctx.Err() != nil {
			return nil, nil
		}
		logits = m.forward(tok, i, i == len(ids)-1)
		if i%16 == 0 {
			touch()
		}
	}
	prefill := time.Since(t0)

	t1 := time.Now()
	dec, stop, think := m.tok.decoder(), &stopFilter{stops: pr.stops}, &thinkSplit{}
	history := append([]int(nil), ids...)
	completion, finish, needTrim := 0, "length", pr.trimLead
	if pr.echo != "" {
		emit(pr.echo, "")
	}
	emitText := func(s string) {
		if s == "" {
			return
		}
		if needTrim {
			needTrim = false
			if strings.Trim(s, " \t") == "" {
				return
			}
			if s[0] == ' ' || s[0] == '\t' {
				s = s[1:]
			}
		}
		if c, r := think.push(s); c != "" || r != "" {
			emit(c, r)
		}
	}
	for step := 0; step < maxNew; step++ {
		if ctx.Err() != nil {
			finish = "abort"
			break
		}
		next := sampleNext(logits, history, sp, m.rng)
		if m.tok.eos[next] {
			finish = "stop"
			break
		}
		completion++
		history = append(history, next)
		if piece := dec.push(next); piece != "" {
			out, hit := stop.push(piece)
			emitText(out)
			if hit {
				finish = "stop"
				break
			}
		}
		touch()
		if pos := len(ids) + completion - 1; step < maxNew-1 && pos < seq {
			logits = m.forward(next, pos, true)
		}
	}
	if finish != "abort" { // let the held-back tails through
		if tail := dec.flush(); tail != "" {
			out, _ := stop.push(tail)
			emitText(out)
		}
		emitText(stop.flush())
		if c, r := think.flush(); c != "" || r != "" {
			emit(c, r)
		}
	}
	genMs := time.Since(t1).Milliseconds()
	if genMs < 1 {
		genMs = 1
	}
	st = &genStats{PromptTokens: len(ids), CompletionTokens: completion, ContextLimit: seq, LoadMs: coldMs, PrefillMs: prefill.Milliseconds(),
		GenMs: genMs, ResolvedFormat: pr.format, FinishReason: finish}
	if completion > 0 {
		st.TokensPerSec = math.Round(float64(completion)/(float64(genMs)/1000)*10) / 10
	}
	return st, nil
}
