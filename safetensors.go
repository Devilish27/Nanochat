package main

// Reader for .safetensors weight files: header inspection without loading data, tensor loading as
// float32 (from F32, BF16, F16, F64 or scaled I8) and symmetric per-row int8 quantization.

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
)

type stTensor struct {
	Dtype      string
	Shape      []int
	Start, End int64
}

type stInfo struct {
	DataOffset, FileSize int64
	Tensors              map[string]stTensor
	TotalParams          int64
	Quant                string // dominant storage type: INT8, BF16, FP16, FP32 ...
}

var (
	stWidth = map[string]int64{"BOOL": 1, "U8": 1, "I8": 1, "F16": 2, "BF16": 2, "I16": 2, "U16": 2, "F32": 4, "I32": 4, "U32": 4, "F64": 8, "I64": 8, "U64": 8}
	stQuant = map[string]string{"I8": "INT8", "U8": "UINT8", "BF16": "BF16", "F16": "FP16", "F32": "FP32", "F64": "FP64"}
)

// readSTHeader validates the JSON header (8-byte length, then JSON, then raw tensor data).
func readSTHeader(f *os.File) (*stInfo, error) {
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.Size() < 16 {
		return nil, errors.New("Invalid safetensors file: file is too small.")
	}
	var lb [8]byte
	if _, err := f.ReadAt(lb[:], 0); err != nil {
		return nil, err
	}
	hl := binary.LittleEndian.Uint64(lb[:])
	if hl == 0 || hl > 64<<20 {
		return nil, errors.New("Invalid safetensors file: unreasonable header length.")
	}
	info := &stInfo{DataOffset: 8 + int64(hl), FileSize: fi.Size(), Tensors: map[string]stTensor{}}
	if info.DataOffset > fi.Size() {
		return nil, errors.New("Invalid safetensors file: truncated header.")
	}
	hdr := make([]byte, hl)
	if _, err := f.ReadAt(hdr, 8); err != nil {
		return nil, err
	}
	var raw map[string]json.RawMessage
	if json.Unmarshal(hdr, &raw) != nil {
		return nil, errors.New("Invalid safetensors header JSON.")
	}
	region := float64(fi.Size() - info.DataOffset)
	counts := map[string]int64{}
	for key, val := range raw {
		if key == "__metadata__" {
			continue
		}
		var e struct {
			Dtype string    `json:"dtype"`
			Shape []float64 `json:"shape"`
			Off   []float64 `json:"data_offsets"`
		}
		if json.Unmarshal(val, &e) != nil || stWidth[e.Dtype] == 0 || e.Shape == nil || len(e.Off) != 2 {
			return nil, fmt.Errorf("Invalid tensor descriptor for %q.", key)
		}
		start, end := e.Off[0], e.Off[1]
		if start < 0 || end < start || end > region || start != math.Trunc(start) || end != math.Trunc(end) {
			return nil, fmt.Errorf("Out-of-bounds tensor offsets for %q.", key)
		}
		numel, shape := int64(1), make([]int, len(e.Shape))
		for i, d := range e.Shape {
			if d < 0 || d != math.Trunc(d) || d > 1<<40 || (d > 0 && numel > (1<<40)/int64(d)) {
				return nil, fmt.Errorf("Invalid tensor shape for %q.", key)
			}
			numel *= int64(d)
			shape[i] = int(d)
		}
		if want := numel * stWidth[e.Dtype]; int64(end-start) != want {
			return nil, fmt.Errorf("Byte length mismatch for %q: expected %d, got %d.", key, want, int64(end-start))
		}
		info.Tensors[key] = stTensor{Dtype: e.Dtype, Shape: shape, Start: int64(start), End: int64(end)}
		// quantization scale tensors are not model parameters
		if !strings.HasSuffix(key, ".scale") && !strings.HasSuffix(key, "_scale") {
			info.TotalParams += numel
			counts[e.Dtype] += numel
		}
	}
	dts := make([]string, 0, len(counts))
	for dt := range counts {
		dts = append(dts, dt)
	}
	sort.Strings(dts)
	dom, best := "F32", int64(-1)
	for _, dt := range dts {
		if counts[dt] > best {
			dom, best = dt, counts[dt]
		}
	}
	if info.Quant = stQuant[dom]; info.Quant == "" {
		info.Quant = dom
	}
	return info, nil
}

func inspectSafetensors(path string) (*stInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readSTHeader(f)
}

var (
	f16Once sync.Once
	f16Tbl  []float32
)

// f16Table maps every IEEE 754 half-precision bit pattern to its float32 value.
func f16Table() []float32 {
	f16Once.Do(func() {
		f16Tbl = make([]float32, 65536)
		for h := 0; h < 65536; h++ {
			s, e, f := uint32(h>>15)<<31, uint32(h>>10)&0x1f, uint32(h)&0x3ff
			switch {
			case e == 0: // zero and subnormals
				v := float32(float64(f) / 1024 * math.Pow(2, -14))
				if s != 0 {
					v = -v
				}
				f16Tbl[h] = v
			case e == 31: // infinities and NaN
				f16Tbl[h] = math.Float32frombits(s | 0x7f800000 | f<<13)
			default:
				f16Tbl[h] = math.Float32frombits(s | (e+112)<<23 | f<<13)
			}
		}
	})
	return f16Tbl
}

// linW is one linear layer [out, in]: either float32 weights in w, or int8 weights in q with a per-row scale.
type linW struct {
	q       []int8
	w       []float32
	scale   []float32
	bias    []float32
	out, in int
}

// quantizeRows turns float32 rows into int8 with one scale per row (4x less RAM).
func quantizeRows(src []float32, out, in int, bias []float32) *linW {
	q, scale := make([]int8, out*in), make([]float32, out)
	for r := 0; r < out; r++ {
		row := src[r*in : (r+1)*in]
		var mx float32
		for _, v := range row {
			if v < 0 {
				v = -v
			}
			if v > mx {
				mx = v
			}
		}
		if mx == 0 {
			continue
		}
		sc := float64(mx) / 127
		scale[r] = float32(sc)
		dst := q[r*in : (r+1)*in]
		for c, v := range row {
			x := math.Floor(float64(v)/sc + 0.5)
			switch {
			case x != x:
				x = 0
			case x < -127:
				x = -127
			case x > 127:
				x = 127
			}
			dst[c] = int8(x)
		}
	}
	return &linW{q: q, scale: scale, bias: bias, out: out, in: in}
}

type stReader struct {
	info  *stInfo
	f     *os.File
	quant bool  // quantize dense float layers to int8 while loading (large models)
	alloc int64 // bytes kept in memory so far
}

func openSafetensors(path string) (*stReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := readSTHeader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	return &stReader{info: info, f: f, quant: info.Quant != "INT8" && info.TotalParams > 110_000_000}, nil
}

func (r *stReader) close() { r.f.Close() }

func (r *stReader) has(name string) bool { _, ok := r.info.Tensors[name]; return ok }

// each streams a tensor's bytes in 1 MB pieces, so a big tensor is never held twice in memory.
func (r *stReader) each(e stTensor, fn func(b []byte)) error {
	sr := io.NewSectionReader(r.f, r.info.DataOffset+e.Start, e.End-e.Start)
	buf := make([]byte, 1<<20)
	for left := e.End - e.Start; left > 0; {
		n := int64(len(buf))
		if left < n {
			n = left
		}
		if _, err := io.ReadFull(sr, buf[:n]); err != nil {
			return errors.New("Unexpected EOF reading safetensors data.")
		}
		fn(buf[:n])
		left -= n
	}
	return nil
}

// f32 loads a tensor as float32. A missing tensor is an error only when required.
func (r *stReader) f32(name string, required bool) ([]float32, error) {
	e, ok := r.info.Tensors[name]
	if !ok {
		if required {
			return nil, fmt.Errorf("Missing required tensor %q.", name)
		}
		return nil, nil
	}
	out := make([]float32, (e.End-e.Start)/stWidth[e.Dtype])
	i := 0
	var conv func(b []byte)
	switch e.Dtype {
	case "F32":
		conv = func(b []byte) {
			for k := 0; k+4 <= len(b); k += 4 {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[k:]))
				i++
			}
		}
	case "BF16":
		conv = func(b []byte) {
			for k := 0; k+2 <= len(b); k += 2 {
				out[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[k:])) << 16)
				i++
			}
		}
	case "F16":
		tbl := f16Table()
		conv = func(b []byte) {
			for k := 0; k+2 <= len(b); k += 2 {
				out[i] = tbl[binary.LittleEndian.Uint16(b[k:])]
				i++
			}
		}
	case "F64":
		conv = func(b []byte) {
			for k := 0; k+8 <= len(b); k += 8 {
				out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(b[k:])))
				i++
			}
		}
	case "I8":
		conv = func(b []byte) {
			for _, c := range b {
				out[i] = float32(int8(c))
				i++
			}
		}
	default:
		return nil, fmt.Errorf("Unsupported tensor dtype %q for %q.", e.Dtype, name)
	}
	if err := r.each(e, conv); err != nil {
		return nil, err
	}
	if e.Dtype == "I8" { // dequantize with the stored scale (per row, or one value for all)
		sc, err := r.f32(name+".scale", false)
		if sc == nil && err == nil {
			sc, err = r.f32(name+"_scale", false)
		}
		if err != nil {
			return nil, err
		}
		if len(e.Shape) == 2 && len(sc) == e.Shape[0] && sc != nil {
			cols := e.Shape[1]
			for row, s := range sc {
				for c := row * cols; c < (row+1)*cols; c++ {
					out[c] *= s
				}
			}
		} else if len(sc) == 1 {
			for c := range out {
				out[c] *= sc[0]
			}
		}
	}
	r.alloc += int64(len(out)) * 4
	return out, nil
}

// linear loads a 2D layer stored as [out, in], or as [in, out] when transpose is set (GPT-2 Conv1D).
func (r *stReader) linear(prefix string, transpose, withBias bool) (*linW, error) {
	wn := prefix + ".weight"
	e, ok := r.info.Tensors[wn]
	if !ok {
		return nil, fmt.Errorf("Missing linear weight tensor %q.", wn)
	}
	if len(e.Shape) != 2 {
		return nil, fmt.Errorf("Expected 2D tensor for %q, got %v.", wn, e.Shape)
	}
	var bias []float32
	var err error
	if withBias {
		if bias, err = r.f32(prefix+".bias", false); err != nil {
			return nil, err
		}
	}
	if e.Dtype == "I8" && !transpose { // keep int8 weights as they are
		out, in := e.Shape[0], e.Shape[1]
		q, i := make([]int8, out*in), 0
		if err := r.each(e, func(b []byte) {
			for _, c := range b {
				q[i] = int8(c)
				i++
			}
		}); err != nil {
			return nil, err
		}
		var sc []float32
		for _, n := range []string{wn + ".scale", wn + "_scale", prefix + ".scale"} {
			if sc, err = r.f32(n, false); err != nil {
				return nil, err
			}
			if sc != nil {
				break
			}
		}
		if sc == nil {
			return nil, fmt.Errorf("Missing scale tensor for INT8 weight %q.", wn)
		}
		if len(sc) == 1 && out > 1 {
			v := sc[0]
			sc = make([]float32, out)
			for k := range sc {
				sc[k] = v
			}
		} else if len(sc) != out {
			return nil, fmt.Errorf("Scale length mismatch for %q: expected %d, got %d.", wn, out, len(sc))
		}
		r.alloc += int64(len(q))
		return &linW{q: q, scale: sc, bias: bias, out: out, in: in}, nil
	}
	before := r.alloc
	f, err := r.f32(wn, true)
	if err != nil {
		return nil, err
	}
	r.alloc = before
	out, in := e.Shape[0], e.Shape[1]
	if transpose {
		in, out = e.Shape[0], e.Shape[1]
		t := make([]float32, out*in)
		for i := 0; i < in; i++ {
			for o, v := range f[i*out : (i+1)*out] {
				t[o*in+i] = v
			}
		}
		f = t
	}
	if r.quant {
		lw := quantizeRows(f, out, in, bias)
		r.alloc += int64(len(lw.q)) + int64(out)*4
		return lw, nil
	}
	r.alloc += int64(len(f)) * 4
	return &linW{w: f, bias: bias, out: out, in: in}, nil
}
