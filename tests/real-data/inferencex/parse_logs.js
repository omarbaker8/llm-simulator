// Pull memory facts out of InferenceX server logs (vLLM, SGLang, TRT-LLM) into one JSON file.
const fs = require('fs'), path = require('path');
const dir = process.argv[2], out = process.argv[3];
const num = (s) => s == null ? null : Number(String(s).replace(/,/g, ''));
const first = (t, re) => { const m = t.match(re); return m ? m[1] : null; };
const all = (t, re) => [...t.matchAll(re)].map((m) => m[1]);
const res = {};
for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.log'))) {
  const id = f.replace('.log', ''), t = fs.readFileSync(path.join(dir, f), 'utf8');
  if (!t.length) continue;
  const r = { id };
  // vLLM
  r.vllm_weights_gib = num(first(t, /Model loading took ([\d.]+) GiB/));
  r.vllm_kv_gib = num(first(t, /Available KV cache memory: ([\d.]+) GiB/));
  r.vllm_kv_tokens = num(first(t, /GPU KV cache size: ([\d,]+) tokens/));
  r.vllm_graph_gib = num(first(t, /Estimated CUDA graph memory: ([\d.]+) GiB/));
  r.gpu_mem_util = num(first(t, /'gpu_memory_utilization': ([\d.]+)/) || first(t, /gpu_memory_utilization=([\d.]+)/) || first(t, /--gpu-memory-utilization[ =]([\d.]+)/));
  r.max_num_seqs = num(first(t, /'max_num_seqs': (\d+)/) || first(t, /max_num_seqs=(\d+)/));
  r.max_num_batched_tokens = num(first(t, /'max_num_batched_tokens': (\d+)/) || first(t, /max_num_batched_tokens=(\d+)/));
  r.kv_cache_dtype = first(t, /'kv_cache_dtype': '([\w]+)'/) || first(t, /kv_cache_dtype=['"]?([\w]+)/) || (/KvCacheConfig\(/.test(t) && /dtype='fp8'/.test(t) ? 'fp8' : null);
  r.max_model_len = num(first(t, /'max_model_len': (\d+)/));
  // SGLang
  const lw = all(t, /Load weight end\..*?mem usage=([\d.]+) GB/g).map(Number);
  r.sgl_weights_gib = lw.length ? Math.max(...lw) : null;
  r.sgl_mem_fraction = num(first(t, /mem_fraction_static=([\d.]+)/));
  const kvl = t.match(/KV Cache is allocated\.(?: dtype: [^,]+,)? #tokens: (\d+),(.*)/);
  if (kvl) {
    r.sgl_kv_tokens = num(kvl[1]);
    const sizes = [...kvl[2].matchAll(/([KV]) size: ([\d.]+) GB/g)].map((m) => Number(m[2]));
    const kvs = kvl[2].match(/KV size: ([\d.]+) GB/);
    r.sgl_kv_gib = sizes.length ? sizes.reduce((a, b) => a + b, 0) : kvs ? Number(kvs[1]) : null;
  }
  r.sgl_max_total_tokens = num(first(t, /max_total_num_tokens=(\d+)/));
  // TRT-LLM
  const tk = t.match(/Allocated ([\d.]+) GiB for max tokens in paged KV cache \((\d+)\)/);
  if (tk) { r.trt_kv_gib = Number(tk[1]); r.trt_kv_tokens = Number(tk[2]); }
  r.trt_free_frac = num(first(t, /free_gpu_memory_fraction['"]?[:=] ?([\d.]+)/));
  r.trt_weights_gib = num(first(t, /Memory used after loading model weights \(inside torch\) in memory usage profiling: ([\d.]+) GiB/));
  for (const k of Object.keys(r)) if (r[k] == null) delete r[k];
  res[id] = r;
}
fs.writeFileSync(out, JSON.stringify(res, null, 1));
const n = (k) => Object.values(res).filter((r) => r[k] != null).length;
console.log('logs', Object.keys(res).length, 'vllm weights', n('vllm_weights_gib'), 'vllm kv', n('vllm_kv_tokens'), 'sgl weights', n('sgl_weights_gib'), 'sgl kv', n('sgl_kv_tokens'), 'trt kv', n('trt_kv_tokens'), 'trt weights', n('trt_weights_gib'), 'util', n('gpu_mem_util'));
