// 200 scenario tests across every major use of the calculator.
// Expected values are hand-derived from published configs / datasheets written out as literals here (not read from
// calc.js), physical invariants, or published measurements (source noted next to each).
// Usage: node tests/scenarios.js
const fs = require('fs'), vm = require('vm'), path = require('path');
const dir = process.argv[2] || path.join(__dirname, '..');
const ctx = {}; ctx.window = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(dir, 'data.js'), 'utf8'), ctx);
const D = ctx.VRAM_DATA, C = require(path.join(dir, 'calc.js'));
const GiB = 2 ** 30;
const M = (s) => { const m = D.models.find((x) => x.slug === s); if (!m) throw new Error('model ' + s); return m; };
const G = (k) => { const g = D.gpus.find((x) => x.key === k); if (!g) throw new Error('gpu ' + k); return g; };

const INF = { mode: 'inference', quant: 'fp16', kvQuant: 'fp16', runtime: 'vllm', numGpus: 1, cluster: false, parallelism: 'tensor', interconnect: 'nvlink_gen4',
  batch: 1, users: 1, seq: 8192, outputTokens: 512, prefixShare: 0, offload: false, offloadTarget: 'cpu_ram', offloadFrac: 0, offloadKv: false,
  spec: false, specDraft: 1, specAccept: 0.7, specTokens: 4, specType: 'sequential', elec: 0.12, carbon: 400, opt: {} };
const FT = { ...INF, mode: 'finetuning', seq: 2048, outputTokens: 1, ftMethod: 'full', ftQuant: 'fp16', loraRank: 16, gradSteps: 1, numSamples: 1000, tokensPerSample: 2048, numEpochs: 1,
  opt: { optimizer_type: 'adamw', zero_stage: 0, flash_attention: true, gradient_checkpointing: true, gradient_checkpointing_ratio: 1, use_8bit_optimizer: false, use_bf16_true: false, use_fused_kernels: true, sequence_packing: true, fused_cross_entropy: false } };
const DIF = { ...INF, runtime: 'diffusers', imageWidth: 1024, imageHeight: 1024, numFrames: 1, inferenceSteps: 28, guidanceScale: 3.5, encoderOffload: true, vaeTiling: false };

const run = (base, slug, gpu, o = {}) => C.calculate({ ...base, ...o, opt: { ...base.opt, ...(o.opt || {}) } }, typeof slug === 'string' ? M(slug) : slug, typeof gpu === 'string' ? G(gpu) : gpu);
const seg = (r, k) => (r.memory.segments.find((s) => s.key === k) || { gib: 0 }).gib;
const tr = (r, k) => { const t = r.trace.find((x) => x.k === k); if (!t) return NaN; const m = String(t.v).match(/[-\d.,]+/); return m ? Number(m[0].replace(/,/g, '')) : NaN; };
const warned = (r, re) => r.warnings.some((w) => re.test(w));

const results = [];
let area = '';
const section = (name) => { area = name; };
function exact(name, formula, expected, actual, tolPct = 0.5) {
  const dev = (actual - expected) / Math.abs(expected || 1) * 100;
  results.push({ area, name, detail: `${formula}: hand ${fmt(expected)} vs repo ${fmt(actual)} (${dev >= 0 ? '+' : ''}${dev.toFixed(2)}%, ±${tolPct}%)`, pass: Number.isFinite(actual) && Math.abs(dev) <= tolPct });
}
function check(name, cond, detail = '') { results.push({ area, name, detail, pass: Boolean(cond) }); }
function bench(name, src, measured, actual, tolPct) {
  const dev = (actual - measured) / measured * 100;
  results.push({ area, name, detail: `${src}: measured ${fmt(measured)} vs repo ${fmt(actual)} (${dev >= 0 ? '+' : ''}${dev.toFixed(1)}%, ±${tolPct}%)`, pass: Number.isFinite(actual) && Math.abs(dev) <= tolPct });
}
const fmt = (x) => !Number.isFinite(x) ? String(x) : Math.abs(x) >= 1e5 ? x.toExponential(4) : Math.abs(x) >= 100 ? x.toFixed(1) : x.toFixed(4);

/* Published config literals */
const LL8 = { P: 8.03e9, L: 32, h: 4096, kv: 8, hd: 128, V: 128256 };
const LL70 = { P: 70.55e9, L: 80, h: 8192, kv: 8, hd: 128, V: 128256 };
const kvTok = (L, kv, hd, B = 2) => L * 2 * kv * hd * B;

/* =========================================================================================================== */
section('A. Weights (GiB)');
{
  const w = (slug, gpu, o) => seg(run(INF, slug, gpu, o), 'weights');
  exact('Llama-3.2-1B FP16 (tied)', '1.24e9 × 2', 1.24e9 * 2 / GiB, w('llama-3-2-1b', '4090_24'), 1);
  exact('Llama-3.2-3B FP16 (tied)', '3.21e9 × 2', 3.21e9 * 2 / GiB, w('llama-3-2-3b', '4090_24'), 1);
  exact('Qwen2.5-7B FP16', '7.62e9 × 2', 7.62e9 * 2 / GiB, w('qwen2-5-7b', 'h100_80'), 1);
  exact('Llama-3.1-8B INT8 W8A16 (emb 16-bit)', '(P − 2Vh)×8.0625/8 + 2Vh×2', ((LL8.P - 2 * LL8.V * LL8.h) * 8.0625 / 8 + 2 * LL8.V * LL8.h * 2) / GiB, w('llama-3-1-8b', 'h100_80', { quant: 'int8' }));
  exact('Llama-3.1-8B NVFP4 dense (emb 16-bit)', '(P − 2Vh)×4.5/8 + 2Vh×2', ((LL8.P - 2 * LL8.V * LL8.h) * 4.5 / 8 + 2 * LL8.V * LL8.h * 2) / GiB, w('llama-3-1-8b', 'b200_192', { quant: 'nvfp4' }));
  exact('Llama-3.1-8B MLX 4-bit (all tensors 4.5 bpw)', 'P × 4.5/8', LL8.P * 4.5 / 8 / GiB, w('llama-3-1-8b', 'm4_max_128', { quant: 'int4', runtime: 'mlx' }));
  exact('Llama-3.1-70B FP8 (emb 16-bit), TP2 → ×2', '(P − 2Vh) + 2Vh×2', ((LL70.P - 2 * LL70.V * LL70.h) + 2 * LL70.V * LL70.h * 2) / GiB, w('llama-3-1-70b', 'h100_80', { quant: 'fp8', numGpus: 2 }) * 2);
  exact('Llama-3.1-70B FP16 split over PP4', 'P × 2 ÷ 4', LL70.P * 2 / 4 / GiB, w('llama-3-1-70b', 'a100_80', { numGpus: 4, parallelism: 'pipeline' }));
  exact('Llama-3.1-70B FP16 TP8 per GPU', 'P × 2 ÷ 8', LL70.P * 2 / 8 / GiB, w('llama-3-1-70b', 'h100_80', { numGpus: 8 }));
  exact('Qwen3.5-27B FP16 + 0.46B vision', '(26.9 + 0.46)e9 × 2', (26.9e9 + 0.46e9) * 2 / GiB, w('qwen35-27b', 'h200_141'), 0.5);
  exact('Qwen3.8-Flash-Next FP16 incl. 52.36B lookup', '(125.05 + 52.36)e9 × 2', (125.05e9 + 52.36e9) * 2 / GiB, w('qwen-38-flash', 'b300_288', { numGpus: 2 }) * 2, 0.5);
  exact('DeepSeek-V4.1-Flash FP8 incl. 196B Engram (16-bit)', '(552e9 − 2Vh)×1 + 2Vh×2 + 196e9×2', ((552e9 - 2 * 129280 * 5120) + 2 * 129280 * 5120 * 2 + 196e9 * 2) / GiB, w('deepseek-v4-1-flash', 'b300_288', { quant: 'fp8', numGpus: 8 }) * 8, 0.5);
  exact('Wan2.2 A14B BF16 denoiser (2 experts)', '28.58e9 × 2', 28.58e9 * 2 / GiB, seg(run(DIF, 'wan-2-2-t2v-a14b', 'h100_80'), 'weights'), 0.1);
  // GGUF vs real Hugging Face file sizes (API, Nov 2026)
  const ggufW = (slug, q) => { const r = run(INF, slug, '4090_24', { quant: q, runtime: 'llamacpp', seq: 1024, outputTokens: 64 }); return r.memory.weightsGiB - C.unreadBytes(C.geometry(M(slug)), q) / GiB; };
  exact('GGUF Llama-3.1-70B Q4_K_M vs bartowski file 42.52 GB', 'file', 42.52e9 / GiB, ggufW('llama-3-1-70b', 'q4_k_m'), 2);
  exact('GGUF Qwen3-8B Q4_K_M vs Qwen file 5.028 GB', 'file', 5.028e9 / GiB, ggufW('qwen3-8b', 'q4_k_m'), 2);
  exact('GGUF Qwen3-0.6B Q8_0 vs Qwen file 0.639 GB', 'file', 0.639e9 / GiB, ggufW('qwen3-0-6b', 'q8_0'), 2);
  exact('GGUF Gemma-3-27B Q4_K_M vs unsloth file 16.547 GB', 'file', 16.547e9 / GiB, ggufW('gemma-3-27b', 'q4_k_m'), 2);
  exact('GGUF Gemma-3-4B Q4_K_M vs unsloth file 2.49 GB', 'file', 2.49e9 / GiB, ggufW('gemma-3-4b', 'q4_k_m'), 2);
  exact('GGUF Gemma-3-1B Q4_K_M (h=1152 → Q5_0 fallback) vs 0.806 GB', 'file', 0.806e9 / GiB, ggufW('gemma-3-1b', 'q4_k_m'), 4);
  exact('GGUF Qwen2.5-7B Q4_K_M vs Qwen file 4.683 GB', 'file', 4.683e9 / GiB, ggufW('qwen2-5-7b', 'q4_k_m'), 2);
  exact('GGUF gpt-oss-20b MXFP4 vs ggml-org file 12.11 GB', 'file', 12.11e9 / GiB, ggufW('gpt-oss-20b', 'mxfp4'), 2);
  exact('GGUF Llama-3.1-8B Q8_0 = README 7.95 GiB', 'README', 7.95, ggufW('llama-3-1-8b', 'q8_0'), 1);
  exact('GGUF Llama-3.1-8B IQ4_XS = README 4.17 GiB', 'README', 4.17, ggufW('llama-3-1-8b', 'iq4_xs'), 1);
  exact('GGUF Llama-3.1-8B Q2_K = README 2.95 GiB', 'README', 2.95, ggufW('llama-3-1-8b', 'q2_k'), 1);
}

/* =========================================================================================================== */
section('B. KV cache per token (bytes) and totals');
{
  const kvpt = (slug, o = {}) => run(INF, slug, 'b300_288', { numGpus: 1, ...o }).memory.kvPerToken;
  exact('Llama-3.2-1B', '16 × 2 × 8 × 64 × 2', kvTok(16, 8, 64), kvpt('llama-3-2-1b'), 0.01);
  exact('Llama-3.2-3B', '28 × 2 × 8 × 128 × 2', kvTok(28, 8, 128), kvpt('llama-3-2-3b'), 0.01);
  exact('Qwen3-8B', '36 × 2 × 8 × 128 × 2', kvTok(36, 8, 128), kvpt('qwen3-8b'), 0.01);
  exact('Qwen3-32B', '64 × 2 × 8 × 128 × 2', kvTok(64, 8, 128), kvpt('qwen3-32b'), 0.01);
  exact('Qwen3-30B-A3B', '48 × 2 × 4 × 128 × 2', kvTok(48, 4, 128), kvpt('qwen3-30b-a3b'), 0.01);
  exact('Phi-4', '40 × 2 × 10 × 128 × 2', kvTok(40, 10, 128), kvpt('phi-4'), 0.01);
  exact('Gemma-2-27B', '46 × 2 × 16 × 128 × 2', kvTok(46, 16, 128), kvpt('gemma-2-27b'), 0.01);
  exact('Gemma-3-27B', '62 × 2 × 16 × 128 × 2', kvTok(62, 16, 128), kvpt('gemma-3-27b'), 0.01);
  exact('Gemma-3-1B', '26 × 2 × 1 × 256 × 2', kvTok(26, 1, 256), kvpt('gemma-3-1b'), 0.01);
  exact('Gemma-4-E2B (12 sliding + 3 global source layers)', '12×2×1×256×2 + 3×2×1×512×2', 12 * 2 * 1 * 256 * 2 + 3 * 2 * 1 * 512 * 2, kvpt('gemma-4-e2b'), 0.01);
  exact('Gemma-4-26B-A4B (25 sliding + 5 global K=V)', '25×2×8×256×2 + 5×1×2×512×2', 25 * 2 * 8 * 256 * 2 + 5 * 1 * 2 * 512 * 2, kvpt('gemma-4-26b-a4b'), 0.01);
  exact('gpt-oss-20b', '24 × 2 × 8 × 64 × 2', kvTok(24, 8, 64), kvpt('gpt-oss-20b', { quant: 'mxfp4' }), 0.01);
  exact('Mixtral-8x7B', '32 × 2 × 8 × 128 × 2', kvTok(32, 8, 128), kvpt('mixtral-8x7b-v0-1'), 0.01);
  exact('Nemotron-3-Nano (6 GQA layers)', '6 × 2 × 2 × 128 × 2', kvTok(6, 2, 128), kvpt('nvidia-nemotron-3-nano-30b-a3b'), 0.01);
  exact('DeepSeek-V3.2 DSA (latent 576 + 132 B indexer)', '61 × (576×2 + 132)', 61 * (576 * 2 + 132), kvpt('deepseek-v32', { quant: 'fp8' }), 0.01);
  exact('GLM-5 DSA', '78 × (576×2 + 132)', 78 * (576 * 2 + 132), kvpt('glm-5', { quant: 'fp8' }), 0.01);
  exact('GLM-5.3 DSA (indexer every 4th layer)', '78 × (576×2 + 132/4)', 78 * (576 * 2 + 33), kvpt('glm-53', { quant: 'fp8' }), 0.01);
  exact('DeepSeek-V4-Flash (21 ÷4 + 20 ÷128)', '(21/4 + 20/128) × (576×2 + 132)', (21 / 4 + 20 / 128) * (576 * 2 + 132), kvpt('deepseek-v4-flash', { quant: 'fp8' }), 0.01);
  exact('DeepSeek-V4-Pro (30 ÷4 + 31 ÷128)', '(30/4 + 31/128) × (576×2 + 132)', (30 / 4 + 31 / 128) * (576 * 2 + 132), kvpt('deepseek-v4-pro', { quant: 'fp8' }), 0.01);
  exact('DeepSeek-V4.1-Flash FP4 KV = model card 890 B', 'card', 890, kvpt('deepseek-v4-1-flash', { quant: 'fp8', kvQuant: 'nvfp4' }), 0.01);
  exact('Llama-3.1-8B Q8_0 KV (llama.cpp)', '32 × 2 × 8 × 128 × 1.0625', kvTok(32, 8, 128, 1.0625), kvpt('llama-3-1-8b', { runtime: 'llamacpp', kvQuant: 'q8_0' }), 0.01);
  exact('Llama-3.1-8B Q4_0 KV (llama.cpp)', '32 × 2 × 8 × 128 × 0.5625', kvTok(32, 8, 128, 0.5625), kvpt('llama-3-1-8b', { runtime: 'llamacpp', kvQuant: 'q4_0' }), 0.01);
  // totals
  const kv = (slug, gpu, o) => seg(run(INF, slug, gpu, o), 'kv');
  exact('Gemma-3-27B @32K (10 full + 52 SWA@1024)', '(10×32768 + 52×1024) × 2·16·128·2 B', ((62 - Math.round(62 * 5 / 6)) * 32768 + Math.round(62 * 5 / 6) * 1024) * 2 * 16 * 128 * 2 / GiB, kv('gemma-3-27b', 'h100_80', { quant: 'fp8', seq: 32768 }), 0.01);
  exact('gpt-oss-120b @128K (18 full + 18 SWA@128)', '(18×131072 + 18×128) × 4096 B', (18 * 131072 + 18 * 128) * 2 * 8 * 64 * 2 / GiB, kv('gpt-oss-120b', 'h200_141', { quant: 'mxfp4', seq: 131072 }), 0.01);
  exact('Llama-8B 8 users × 8K', '8 × 8192 × 131072', 8 * 8192 * 131072 / GiB, kv('llama-3-1-8b', 'h100_80', { users: 8 }), 0.01);
  exact('Llama-8B 8 users, 100% prompt shared', '(7680 + 8×512) × 131072', (7680 + 8 * 512) * 131072 / GiB, kv('llama-3-1-8b', 'h100_80', { users: 8, prefixShare: 1 }), 0.01);
  exact('Llama-70B KV TP8 per GPU (8 KV heads → ÷8)', '8192 × 327680 ÷ 8', 8192 * 327680 / 8 / GiB, kv('llama-3-1-70b', 'h100_80', { numGpus: 8 }), 0.01);
  exact('Qwen3-30B-A3B KV TP8 per GPU (4 KV heads → ÷4 only)', '8192 × 98304 ÷ 4', 8192 * kvTok(48, 4, 128) / 4 / GiB, kv('qwen3-30b-a3b', 'h100_80', { numGpus: 8 }), 0.01);
  exact('DeepSeek-V3 KV TP8 per GPU (MLA replicated)', '8192 × 70272', 8192 * 70272 / GiB, kv('deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8 }), 0.01);
  exact('Llama-70B KV PP4 per GPU (layers split ÷4)', '8192 × 327680 ÷ 4', 8192 * 327680 / 4 / GiB, kv('llama-3-1-70b', 'a100_80', { numGpus: 4, parallelism: 'pipeline' }), 0.01);
  exact('Qwen3.5-27B llama.cpp: GDN state in F32 (2×)', '16×8192×(2·4·256·2) + 48×1634304×2', (16 * 8192 * 4096 + 48 * 1634304 * 2) / GiB, kv('qwen35-27b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp' }), 0.01);
  const dp = run(INF, 'llama-3-1-8b', 'h100_80', { numGpus: 2, parallelism: 'replicated', users: 3 });
  const nonKv = dp.memory.perGpu - seg(dp, 'kv') - seg(dp, 'frag');
  check('DP2 with 3 users: cluster total counts 3 sequences of KV, not 4', Math.abs(dp.memory.total - (2 * nonKv + 3 * 1.0 + 0.03 * (3 * 1.0 + 2 * seg(dp, 'act')))) < 0.01, `total ${dp.memory.total.toFixed(3)} GiB`);
}

/* =========================================================================================================== */
section('C. Runtime budgets, offload, speculative decoding');
{
  const v = run(INF, 'llama-3-1-8b', 'h100_80');
  exact('vLLM budget = 0.90 × 80 GiB', '0.9 × 80', 72, v.memory.capPerGpu, 0.01);
  exact('vLLM % = per-GPU ÷ 72 GiB', 'perGpu/72', v.memory.perGpu / 72 * 100, v.memory.pct, 0.01);
  const sg = run(INF, 'llama-3-1-8b', 'h100_80', { runtime: 'sglang' });
  exact('SGLang budget = 0.90 × 80 GiB', '0.9 × 80', 72, sg.memory.capPerGpu, 0.01);
  const lc = run(INF, 'llama-3-1-8b', 'h100_80', { runtime: 'llamacpp', quant: 'q4_k_m' });
  exact('llama.cpp budget = full 80 GiB', '80', 80, lc.memory.capPerGpu, 0.01);
  const t = run(INF, 'llama-3-1-8b', 'h100_80', { runtime: 'trtllm' });
  const nonKv = t.memory.perGpu - seg(t, 'kv');
  exact('TensorRT-LLM budget = 80 − 0.1 × (80 − non-KV)', '80 − 0.1(80 − nonKV)', 80 - 0.1 * (80 - nonKv), t.memory.capPerGpu, 0.01);
  exact('Apple M4 Max 128 GB: 75% wired limit', '0.75 × 128', 96, run(INF, 'llama-3-1-8b', 'm4_max_128', { runtime: 'mlx', quant: 'int4' }).memory.capPerGpu, 0.01);
  exact('Apple M4 Pro 32 GB: 67% wired limit', '⅔ × 32', 32 * 2 / 3, run(INF, 'llama-3-1-8b', 'm4_pro_32', { runtime: 'mlx', quant: 'int4' }).memory.capPerGpu, 0.01);
  exact('B200 usable 180 GiB (llama.cpp)', 'usable_memory', 180, run(INF, 'llama-3-1-8b', 'b200_192', { runtime: 'llamacpp', quant: 'q8_0' }).memory.capPerGpu, 0.01);
  check('70B FP16 does not fit one H200 under vLLM 0.90 (131 GiB > 127 GiB)', run(INF, 'llama-3-1-70b', 'h200_141').memory.pct > 100);
  const off = run(INF, 'llama-3-1-70b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp', offload: true, offloadFrac: 0.6 });
  exact('Offload 60%: GPU weights = 40% of GGUF', '0.4 × weights', 0.4 * off.memory.weightsGiB, seg(off, 'weights'), 0.01);
  const offN = run(INF, 'llama-3-1-70b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp', offload: true, offloadFrac: 0.6, offloadTarget: 'nvme' });
  check('NVMe offload is slower than system-RAM offload', offN.perf.tps < off.perf.tps, `${offN.perf.tps.toFixed(2)} < ${off.perf.tps.toFixed(2)}`);
  const offKv = run(INF, 'llama-3-1-8b', '4090_24', { offload: true, offloadFrac: 0, offloadKv: true, seq: 32768 });
  check('KV offload removes KV from the GPU', seg(offKv, 'kv') === 0);
  const sp = run(INF, 'llama-3-1-70b', 'h200_141', { quant: 'fp8', spec: true, specDraft: 1, specAccept: 0.7, specTokens: 4 });
  exact('Spec decoding E[tokens/cycle] α=0.7, k=4', '(1 − 0.7⁵)/(1 − 0.7)', (1 - 0.7 ** 5) / 0.3, sp.perf.spec.E, 0.01);
  const kvRep = run(INF, 'llama-3-1-70b', 'h200_141', { quant: 'fp8' });
  exact('Draft memory = 1B × 8 bits + KV × √(1/70.55)', '1e9 + KV·√(1/70.55)', (1e9 + seg(kvRep, 'kv') * GiB * Math.sqrt(1e9 / 70.55e9)) / GiB, seg(sp, 'draft'), 1);
  check('Speculative decoding: 1 < speed-up < E (draft costs time)', sp.perf.spec.speedup > 1 && sp.perf.spec.speedup < sp.perf.spec.E, `${sp.perf.spec.speedup.toFixed(2)}×`);
  const spPar = run(INF, 'llama-3-1-70b', 'h200_141', { quant: 'fp8', spec: true, specType: 'parallel' });
  check('Parallel drafting (1 draft pass) beats sequential (k passes)', spPar.perf.tps > sp.perf.tps);
  const spLo = run(INF, 'llama-3-1-70b', 'h200_141', { quant: 'fp8', spec: true, specAccept: 0.3 });
  check('Lower acceptance → lower speed-up', spLo.perf.spec.speedup < sp.perf.spec.speedup);
}

/* =========================================================================================================== */
section('D. FLOPs (hand-derived)');
{
  const pf = (slug, gpu, o) => tr(run(INF, slug, gpu, o), 'Prefill FLOPs');
  exact('Qwen3-8B prefill 7,680 tok (TFLOP)', '2(P − Vh)T + 4·36·T²/2·32·128', (2 * (8.19e9 - 151936 * 4096) * 7680 + 4 * 36 * 7680 ** 2 / 2 * 32 * 128) / 1e12, pf('qwen3-8b', 'h100_80'), 0.5);
  exact('Llama-3.2-1B prefill (tied: lm_head is the table)', '2P·T + 4·16·T²/2·32·64', (2 * 1.24e9 * 7680 + 4 * 16 * 7680 ** 2 / 2 * 32 * 64) / 1e12, pf('llama-3-2-1b', '4090_24'), 0.5);
  {
    const T = 32768 - 512, Wn = 1024, swaPairs = Wn * (T - Wn / 2), nSwa = Math.round(62 * 5 / 6), nFull = 62 - nSwa;
    exact('Gemma-3-27B prefill 32K (10 full + 52 SWA@1024)', '2(P)T + 4·(10·T²/2 + 52·W(T−W/2))·32·128', (2 * 27.0e9 * T + 4 * (nFull * T * T / 2 + nSwa * swaPairs) * 32 * 128) / 1e12, pf('gemma-3-27b', 'h100_80', { quant: 'fp8', seq: 32768 }), 0.5);
  }
  {
    const T = 32768 - 512, idx = 61 * T * T / 2 * 2 * 64 * 128, attn = 61 * 128 * 640 * (2048 * T - 2048 * 2048 / 2);
    exact('DeepSeek-V3.2 prefill 32K: top-2048 sparse attention + full indexer', '2(A−Vh)T + 61·128·640·(2048T − 2048²/2) + 61·T²/2·2·64·128', (2 * (37e9 - 129280 * 7168) * T + attn + idx) / 1e12, pf('deepseek-v32', 'h200_141', { quant: 'fp8', numGpus: 8, seq: 32768 }), 0.5);
  }
  {
    const T = 7680;
    const r = run(INF, 'nvidia-nemotron-3-nano-30b-a3b', 'h100_80');
    const Gm = C.geometry(M('nvidia-nemotron-3-nano-30b-a3b'));
    exact('Nemotron-3-Nano prefill: attention only on 6 of 52 layers', '2·Amm·T + 4·6·T²/2·32·128', (2 * Gm.Amm * T + 4 * 6 * T * T / 2 * 32 * 128) / 1e12, tr(r, 'Prefill FLOPs'), 0.5);
  }
  {
    const T = 7680, nSwa = Math.round(60 * 5 / 6), nFull = 60 - nSwa;
    exact('Gemma-4-31B prefill: global layers use head_dim 512', '2PT + 4·(10·T²/2·512 + 50·W(T−W/2)·256)·32', (2 * 30.7e9 * T + 4 * 32 * (nFull * T * T / 2 * 512 + nSwa * 1024 * (T - 512) * 256)) / 1e12, pf('gemma-4-31b', 'h100_80', { quant: 'fp8' }), 1.5);
  }
  const ft = (o) => tr(run(FT, 'llama-3-1-8b', 'h100_80', o), 'FLOPs / token');
  const lin = 2 * (LL8.P - LL8.V * LL8.h), att = 4 * 32 * 1024 * 32 * 128;
  exact('Full FT no checkpointing: (lin + attn) × 3', '×3', (lin + att) * 3 / 1e9, ft({ opt: { gradient_checkpointing: false } }), 0.2);
  exact('LoRA no checkpointing: lin × 2 + attn × 3', 'lin×2 + attn×3', (lin * 2 + att * 3) / 1e9, ft({ ftMethod: 'lora', opt: { gradient_checkpointing: false } }), 0.2);
  exact('Full FT 50% checkpointing: (lin + attn) × 3.5', '×3.5', (lin + att) * 3.5 / 1e9, ft({ opt: { gradient_checkpointing_ratio: 0.5 } }), 0.2);
  exact('Full FT @8K context: attention grows to T/2 = 4096', '(lin + 4·32·4096·4096) × 4', (lin + 4 * 32 * 4096 * 32 * 128) * 4 / 1e9, ft({ seq: 8192 }), 0.2);
  {
    const r = run(DIF, 'stable-diffusion-3-5-large', 'h100_80');
    const N = 4096 + 333;
    exact('SD3.5-L FLOPs/step (MMDiT: 2.70B/token, 333 text tokens, CFG 2)', '(2·8.1e9·0.333·N + 4·38·N²·2432) × 2', (2 * 8.1e9 * 0.333 * N + 4 * 38 * N * N * 2432) * 2 / 1e12, tr(r, 'FLOPs per step'), 1);
  }
  {
    const r = run(DIF, 'wan-2-1-t2v-14b', 'h100_80', { imageWidth: 1280, imageHeight: 720, numFrames: 81, inferenceSteps: 50, guidanceScale: 5 });
    const N = 21 * 80 * 45;
    exact('Wan2.1-14B 720p×81 FLOPs/step (cross-attn: no text tokens, CFG 2)', '(2·14.29e9·0.837·N + 4·40·N²·5120) × 2', (2 * 14.29e9 * 0.837 * N + 4 * 40 * N * N * 5120) * 2 / 1e12, tr(r, 'FLOPs per step'), 0.5);
  }
  {
    const r = run(DIF, 'qwen-image-2512', 'h100_80', { guidanceScale: 4 });
    const N = 4096 + 256;
    exact('Qwen-Image FLOPs/step (6.8B/token, CFG 2)', '(2·20.43e9·0.333·N + 4·60·N²·3072) × 2', (2 * 20.43e9 * 0.333 * N + 4 * 60 * N * N * 3072) * 2 / 1e12, tr(r, 'FLOPs per step'), 1);
  }
}

/* =========================================================================================================== */
section('E. Decode physics & invariants');
{
  // Memory-bound roofline: a token can never be produced faster than reading the active weights once.
  const roof = (slug, gpu, o, bytes) => { const r = run(INF, slug, gpu, o); return { r, roof: G(gpu).bandwidth_gbs * 1e9 * (o.numGpus || 1) / bytes }; };
  for (const [slug, gpu, o, bytes, label] of [
    ['llama-3-1-8b', 'h100_80', {}, (LL8.P - LL8.V * LL8.h) * 2, 'Llama-8B FP16 H100'],
    ['llama-3-1-8b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp' }, 4.3e9, 'Llama-8B Q4_K_M 4090'],
    ['llama-3-1-70b', 'h100_80', { quant: 'fp8', numGpus: 4 }, (LL70.P - LL70.V * LL70.h), 'Llama-70B FP8 TP4'],
    ['qwen3-30b-a3b', 'h100_80', { quant: 'fp8' }, 3.0e9, 'Qwen3-30B-A3B FP8 (≥3B active bytes)'],
    ['deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8 }, 36e9, 'DeepSeek-V3 FP8 8×H200'],
    ['gpt-oss-120b', 'h100_80', { quant: 'mxfp4' }, 3.5e9, 'gpt-oss-120b MXFP4 H100'],
    ['llama-3-1-8b', 'm4_max_128', { runtime: 'mlx', quant: 'int4' }, 4.0e9, 'Llama-8B MLX 4-bit M4 Max'],
    ['llama-3-1-8b', 'mi300x_192', {}, (LL8.P - LL8.V * LL8.h) * 2, 'Llama-8B FP16 MI300X'],
  ]) { const x = roof(slug, gpu, o, bytes); check(`${label}: tok/s below bandwidth roofline`, x.r.perf && x.r.perf.tps < x.roof, `${x.r.perf && x.r.perf.tps.toFixed(1)} < ${x.roof.toFixed(1)}`); }
  const ctxSweep = (slug, gpu, o) => [4096, 16384, 65536].map((s) => run(INF, slug, gpu, { ...o, seq: s }).perf.tps);
  for (const [slug, gpu, o] of [['llama-3-1-8b', 'h100_80', {}], ['deepseek-v32', 'h200_141', { quant: 'fp8', numGpus: 8 }], ['gemma-3-27b', 'h100_80', { quant: 'fp8' }], ['qwen35-27b', 'h100_80', { quant: 'fp8' }], ['gpt-oss-120b', 'h100_80', { quant: 'mxfp4' }]]) {
    const t = ctxSweep(slug, gpu, o); check(`${slug}: decode slows (or holds) as context grows`, t[0] >= t[1] && t[1] >= t[2], t.map((x) => x.toFixed(1)).join(' ≥ '));
  }
  const lin = ctxSweep('qwen35-27b', 'h100_80', { quant: 'fp8' }), dense = ctxSweep('qwen3-32b', 'h100_80', { quant: 'fp8' });
  check('Hybrid linear attention slows less with context than full attention', lin[2] / lin[0] > dense[2] / dense[0], `${(lin[2] / lin[0]).toFixed(3)} vs ${(dense[2] / dense[0]).toFixed(3)}`);
  const v32 = run(INF, 'deepseek-v32', 'h200_141', { quant: 'fp8', numGpus: 8, seq: 131072, users: 4 }), v3 = run(INF, 'deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8, seq: 131072, users: 4 });
  check('DSA sparse (V3.2) decodes faster than dense MLA (V3) at 128K × 4 users', v32.perf.tps > v3.perf.tps, `${v32.perf.tps.toFixed(1)} > ${v3.perf.tps.toFixed(1)}`);
  const u = [1, 8, 32].map((n) => run(INF, 'llama-3-1-8b', 'h100_80', { users: n, seq: 4096 }));
  check('Throughput rises with users (1 → 8 → 32)', u[0].perf.throughput < u[1].perf.throughput && u[1].perf.throughput < u[2].perf.throughput);
  check('Per-user speed falls with users', u[0].perf.tps > u[1].perf.tps && u[1].perf.tps > u[2].perf.tps);
  check('Queued TTFT grows with users', u[2].perf.ttftQueued > u[1].perf.ttftQueued && u[1].perf.ttftQueued >= u[0].perf.ttftQueued);
  const tp = [1, 2, 4, 8].map((n) => run(INF, 'llama-3-1-70b', 'h100_80', { quant: 'fp8', numGpus: n }));
  check('70B FP8: single-user speed rises with TP 2 → 4 → 8', tp[1].perf.tps < tp[2].perf.tps && tp[2].perf.tps < tp[3].perf.tps, tp.slice(1).map((x) => x.perf.tps.toFixed(1)).join(' < '));
  check('70B FP8 fits on one H100 (vLLM 0.90 budget)', tp[0].perf != null || tp[0].memory.pct > 100);
  const f8 = run(INF, 'llama-3-1-8b', 'h100_80', { quant: 'fp8' }), f16 = run(INF, 'llama-3-1-8b', 'h100_80');
  check('FP8 decodes faster than FP16 (fewer bytes)', f8.perf.tps > f16.perf.tps);
  check('FP8 prefill faster than FP16 on H100 (FP8 tensor cores)', f8.perf.prefillTps > f16.perf.prefillTps * 1.3);
  const a8 = run(INF, 'llama-3-1-8b', 'a100_80', { quant: 'fp8' }), a16 = run(INF, 'llama-3-1-8b', 'a100_80');
  check('A100 FP8 (no FP8 cores): prefill no faster than FP16', a8.perf.prefillTps <= a16.perf.prefillTps * 1.001 && warned(a8, /no FP8 tensor cores/));
  const nv4 = run(INF, 'llama-3-1-70b', 'b200_192', { quant: 'nvfp4' }), fp8b = run(INF, 'llama-3-1-70b', 'b200_192', { quant: 'fp8' });
  check('NVFP4 prefill faster than FP8 on B200', nv4.perf.prefillTps > fp8b.perf.prefillTps);
  const llc = run(INF, 'llama-3-1-8b', 'h100_80', { runtime: 'llamacpp', quant: 'q8_0' }), vq = run(INF, 'llama-3-1-8b', 'h100_80', { quant: 'fp8' });
  check('vLLM FP8 prefill ≫ llama.cpp Q8_0 prefill on H100', vq.perf.prefillTps > 3 * llc.perf.prefillTps);
  const nvl = run(INF, 'llama-3-1-70b', 'h100_80', { quant: 'fp8', numGpus: 4, interconnect: 'nvlink_gen4' }), pcie = run(INF, 'llama-3-1-70b', 'h100pcie_80', { quant: 'fp8', numGpus: 4, interconnect: 'pcie5' });
  check('TP4 over NVLink beats TP4 over PCIe', nvl.perf.tps > pcie.perf.tps);
  const xg2 = run(INF, 'llama-3-1-70b', 'mi300x_192', { numGpus: 2, interconnect: 'amd_xgmi', users: 32, seq: 4096 }), xg8 = run(INF, 'llama-3-1-70b', 'mi300x_192', { numGpus: 8, interconnect: 'amd_xgmi', users: 32, seq: 4096 });
  check('xGMI TP2 uses one 64 GB/s peer link (comm per GPU-step higher than TP8)', tr(xg2, 'Fixed overhead') > 0 && xg8.perf.throughput > xg2.perf.throughput);
  const pp1 = run(INF, 'llama-3-1-405b', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'pipeline' }), tp1 = run(INF, 'llama-3-1-405b', 'h200_141', { quant: 'fp8', numGpus: 8 });
  check('405B: TP ≥ 4× faster than PP for one user (NVIDIA: 5.6×)', tp1.perf.tps / pp1.perf.tps > 4 && tp1.perf.tps / pp1.perf.tps < 8, `${(tp1.perf.tps / pp1.perf.tps).toFixed(2)}×`);
  const off = [0.6, 0.8].map((f) => run(INF, 'llama-3-1-70b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp', offload: true, offloadFrac: f }));
  check('More CPU offload → slower decode (60% vs 80%)', off[0].perf && off[1].perf && off[0].perf.tps > off[1].perf.tps);
  const draftKv = run(INF, 'qwen35-27b', 'h100_80', { quant: 'fp8', users: 16 }), draftKvLc = run(INF, 'qwen35-27b', 'h100_80', { quant: 'fp8', users: 16, runtime: 'llamacpp' });
  check('Recurrent state is larger under llama.cpp (F32) than vLLM (BF16)', seg(draftKvLc, 'kv') > seg(draftKv, 'kv'));
  const pre = run(INF, 'llama-3-1-8b', 'h100_80', { users: 16, prefixShare: 0.9 }), noPre = run(INF, 'llama-3-1-8b', 'h100_80', { users: 16 });
  check('Shared prompt prefix raises multi-user throughput (fewer prefills, less KV)', pre.perf.throughput > noPre.perf.throughput);
  const b4 = run(INF, 'llama-3-1-8b', 'h100_80', { batch: 4 }), b1 = run(INF, 'llama-3-1-8b', 'h100_80', { batch: 1 });
  exact('n=4 parallel samples share one prompt prefill (TTFT ≈ same)', 'TTFT(n=4) ≈ TTFT(n=1)', b1.perf.ttft, b4.perf.ttft, 5);
  check('Power stays within TDP × GPUs', [f16, tp[3], nv4].every((r) => r.watts <= r.layout.n * 1200 + 1));
}

/* =========================================================================================================== */
section('F. Published benchmarks');
{
  const llama2_7b = { slug: 'llama-2-7b-ref', name: 'Llama-2-7B', task_type: 'text_generation', architecture: 'dense', num_of_params: '6.74', hidden_dim_size: 4096, ffn_intermediate_size: 11008, vocab_size: 32000, num_of_layers: 32, num_attention_heads: 32, num_key_value_heads: 32, attention_head_dim: 128, attention_structure: 'mha', context_length: 4096, tie_word_embeddings: false };
  const sb = (gpu) => run(INF, llama2_7b, gpu, { quant: 'q4_0', runtime: 'llamacpp', seq: 640, outputTokens: 128 });
  bench('llama.cpp Llama-2-7B Q4_0 H100 tg128', 'llama.cpp CUDA scoreboard #15013', 268, sb('h100_80').perf.tps, 20);
  bench('llama.cpp Llama-2-7B Q4_0 A100 tg128', 'llama.cpp CUDA scoreboard #15013', 191, sb('a100_80').perf.tps, 20);
  bench('llama.cpp Llama-2-7B Q4_0 H100 pp512', 'llama.cpp CUDA scoreboard #15013', 9918, sb('h100_80').perf.prefillTps, 30);
  bench('llama.cpp Llama-2-7B Q4_0 A100 pp512', 'llama.cpp CUDA scoreboard #15013', 4850, sb('a100_80').perf.prefillTps, 30);
  bench('gpt-oss-20b MXFP4 RTX 4090 llama.cpp', 'llama.cpp discussion #15396', 222, run(INF, 'gpt-oss-20b', '4090_24', { quant: 'mxfp4', runtime: 'llamacpp', seq: 4096 }).perf.tps, 15);
  bench('gpt-oss-120b MXFP4 M2 Ultra llama.cpp', 'llama.cpp discussion #15396', 80, run(INF, 'gpt-oss-120b', 'm2_ultra_192', { quant: 'mxfp4', runtime: 'llamacpp', seq: 4096 }).perf.tps, 25);
  bench('DeepSeek-R1 FP8 8×H200 SGLang batch 1', 'Verda blog', 47.7, run(INF, 'deepseek-r1-671b', 'h200_141', { quant: 'fp8', numGpus: 8, runtime: 'sglang', seq: 2048 }).perf.tps, 30);
  bench('gpt-oss-120b 8×B200 TensorRT-LLM batch 1', 'TensorRT-LLM tech blog 9', 420, run(INF, 'gpt-oss-120b', 'b200_192', { quant: 'mxfp4', numGpus: 8, runtime: 'trtllm', interconnect: 'nvlink_gen5', seq: 2048 }).perf.tps, 45);
  const big = (par) => run(INF, 'llama-3-1-405b', 'h200_141', { quant: 'fp8', kvQuant: 'fp8', runtime: 'trtllm', numGpus: 8, parallelism: par, users: 1024, seq: 2176, outputTokens: 128 });
  const tpT = big('tensor'), ppT = big('pipeline');
  bench('405B 8×H200 TP max throughput 2048/128', 'NVIDIA H200 NVLink Switch blog', 506, tpT.perf.throughput, 30);
  bench('405B 8×H200 PP max throughput 2048/128', 'NVIDIA H200 NVLink Switch blog', 764, ppT.perf.throughput, 30);
  bench('405B PP ÷ TP at max throughput', 'NVIDIA: PP 1.5× TP', 1.51, ppT.perf.throughput / tpT.perf.throughput, 20);
  bench('FLUX.1-dev BF16 H100 28 steps (s)', 'Modal FLUX benchmark', 6.75, run(DIF, 'flux-1-dev', 'h100_80').perf.seconds, 15);
  bench('FLUX.1-dev BF16 L20 28 steps (s)', 'flux-fast README baseline', 24.94, run(DIF, 'flux-1-dev', 'l20_48').perf.seconds, 15);
  bench('FLUX.1-dev BF16 RTX 4090 28 steps (s)', 'community 15–20 s', 17.5, run(DIF, 'flux-1-dev', '4090_24').perf.seconds, 20);
  bench('Wan2.1-14B 720p×81 50 steps H100 (s)', 'arXiv 2509.19222', 1875, run(DIF, 'wan-2-1-t2v-14b', 'h100_80', { imageWidth: 1280, imageHeight: 720, numFrames: 81, inferenceSteps: 50, guidanceScale: 5 }).perf.seconds, 15);
  bench('Wan2.1-1.3B 720p×81 50 steps H100 (s)', 'arXiv 2509.19222', 410, run(DIF, 'wan-2-1-t2v-1-3b', 'h100_80', { imageWidth: 1280, imageHeight: 720, numFrames: 81, inferenceSteps: 50, guidanceScale: 5 }).perf.seconds, 15);
  bench('SDXL A100 30 steps (s)', 'diffusion-fast baseline', 3.31, run(DIF, 'sdxl-1-0-base', 'a100_80', { inferenceSteps: 30, guidanceScale: 5 }).perf.seconds, 15);
  check('Wan2.2-TI2V-5B 720p×121 on RTX 4090 under 9 min (Wan2.2 README)', run(DIF, 'wan-2-2-ti2v-5b', '4090_24', { imageWidth: 1280, imageHeight: 704, numFrames: 121, inferenceSteps: 50, guidanceScale: 5 }).perf.seconds < 540);
  const f5 = (q) => run(DIF, 'flux-1-dev', '5090_32', { quant: q, runtime: 'tensorrt', inferenceSteps: 30 }).perf.seconds;
  bench('FLUX TensorRT 5090: FP16 ÷ FP8 time', 'NVIDIA TensorRT FP4 blog (10.93 / 6.68)', 10.93 / 6.68, f5('fp16') / f5('fp8'), 15);
  bench('FLUX TensorRT 5090: FP16 ÷ FP4 time', 'NVIDIA TensorRT FP4 blog (10.93 / 3.85)', 10.93 / 3.85, f5('fp16') / f5('nvfp4'), 20);
  check('Wan VAE decode 720p×81 frames ≈ 4–8 s on H100', (() => { const r = run(DIF, 'wan-2-1-t2v-14b', 'h100_80', { imageWidth: 1280, imageHeight: 720, numFrames: 81, inferenceSteps: 50, guidanceScale: 5 }); const m = r.trace.find((t) => t.k === 'Generation time').f.match(/VAE decode ([\d.]+) (ms|s)/); const s = Number(m[1]) / (m[2] === 'ms' ? 1000 : 1); return s > 4 && s < 8; })());
}

/* =========================================================================================================== */
section('G. Fine-tuning');
{
  const f = (o, slug = 'llama-3-1-8b', gpu = 'h100_80') => run(FT, slug, gpu, o);
  const sum = (r, ...k) => k.reduce((a, x) => a + seg(r, x), 0);
  exact('FP32 full FT: 4 + 4 + 8 = 16 B/param', 'P × 16', LL8.P * 16 / GiB, sum(f({ ftQuant: 'fp32' }), 'weights', 'grads', 'optim'), 0.01);
  exact('FP8 full FT: BF16+FP8 cache 3 + grads 2 + master 4 + Adam 8 = 17 B', 'P × 17', LL8.P * 17 / GiB, sum(f({ ftQuant: 'fp8' }), 'weights', 'grads', 'optim'), 0.01);
  exact('8-bit Adam: 2 + 2 + 4 + 2 = 10 B/param', 'P × 10', LL8.P * 10 / GiB, sum(f({ opt: { use_8bit_optimizer: true } }), 'weights', 'grads', 'optim'), 0.01);
  exact('SGD momentum: 2 + 2 + 4 + 4 = 12 B/param', 'P × 12', LL8.P * 12 / GiB, sum(f({ opt: { optimizer_type: 'sgd' } }), 'weights', 'grads', 'optim'), 0.01);
  exact('Pure BF16 AdamW: 2 + 2 + 0 + 4 = 8 B/param', 'P × 8', LL8.P * 8 / GiB, sum(f({ opt: { use_bf16_true: true } }), 'weights', 'grads', 'optim'), 0.01);
  exact('ZeRO-1 ×4: optimizer+master sharded', '2P + 2P + 12P/4', (LL8.P * 4 + LL8.P * 12 / 4) / GiB, sum(f({ numGpus: 4, opt: { zero_stage: 1 } }), 'weights', 'grads', 'optim'), 0.01);
  exact('ZeRO-2 ×4: + gradients sharded', '2P + (2P + 12P)/4', (LL8.P * 2 + LL8.P * 14 / 4) / GiB, sum(f({ numGpus: 4, opt: { zero_stage: 2 } }), 'weights', 'grads', 'optim'), 0.01);
  exact('ZeRO-3 ×4: everything sharded', '16P/4', LL8.P * 16 / 4 / GiB, sum(f({ numGpus: 4, opt: { zero_stage: 3 } }), 'weights', 'grads', 'optim'), 0.01);
  const z3 = f({ numGpus: 8, opt: { zero_stage: 3 } }, 'llama-3-1-70b');
  exact('ZeRO-3 gather buffer = 2 layers of BF16 + 0.25 GiB NCCL (independent of N)', '2·(P/L)·2 + 0.25 GiB', (2 * LL70.P / 80 * 2) / GiB + 0.25, seg(z3, 'comm'), 0.5);
  const off1 = f({ opt: { offload_optimizer: true } });
  check('Single-GPU optimizer offload removes optimizer states from the GPU', seg(off1, 'optim') === 0 && off1.memory.offloaded.length === 1);
  const qwenLora = f({ ftMethod: 'lora' }, 'qwen3-8b');
  const qPer = 16 * ((4096 + 32 * 128) * 2 + (4096 + 8 * 128) * 2 + 3 * (4096 + 12288));
  exact('LoRA r16 Qwen3-8B adapters (fp32)', 'r·Σ(d_in+d_out)·36 × 4 B', qPer * 36 * 4 / GiB, seg(qwenLora, 'lora'), 0.01);
  const r64 = f({ ftMethod: 'lora', loraRank: 64 });
  exact('LoRA r64 Llama-8B = 4 × r16 adapters', '4 × 41.94M × 4 B', 41943040 * 4 * 4 / GiB, seg(r64, 'lora'), 0.01);
  exact('Logits naive: 2048 × 128256 × 10 B', '10 B/elem', 2048 * 128256 * 10 / GiB, seg(f({ ftMethod: 'lora' }), 'logits'), 0.01);
  exact('Logits fused CE: 256-token chunk × 8 B', '256 × V × 8', 256 * 128256 * 8 / GiB, seg(f({ ftMethod: 'lora', opt: { fused_cross_entropy: true } }), 'logits'), 0.01);
  const ql = f({ ftMethod: 'qlora' });
  exact('QLoRA NF4 base 4.127 bpw, embeddings BF16', '(P − 2Vh)·4.127/8 + 2Vh·2', ((LL8.P - 2 * LL8.V * LL8.h) * 4.127 / 8 + 2 * LL8.V * LL8.h * 2) / GiB, seg(ql, 'weights'), 0.01);
  check('QLoRA uses less memory than LoRA', ql.memory.perGpu < f({ ftMethod: 'lora' }).memory.perGpu);
  const ck0 = f({ ftMethod: 'lora', opt: { gradient_checkpointing: false } }), ck1 = f({ ftMethod: 'lora' });
  check('Gradient checkpointing cuts activations', seg(ck1, 'act') < seg(ck0, 'act') / 3);
  check('Gradient checkpointing costs throughput (extra forward)', ck1.perf.trainTps < ck0.perf.trainTps);
  const eager = f({ ftMethod: 'lora', opt: { flash_attention: false } });
  check('Eager attention (no flash) stores T² scores: more activation memory', seg(eager, 'act') > seg(ck1, 'act'));
  const d1 = f({ numGpus: 4, opt: { zero_stage: 2 } }, 'llama-3-1-8b'), d3 = f({ numGpus: 4, opt: { zero_stage: 3 } });
  check('ZeRO-3 trains slower than ZeRO-2 (param all-gathers)', d3.perf && d1.perf && d3.perf.trainTps < d1.perf.trainTps);
  const ga1 = f({ numGpus: 4, opt: { zero_stage: 2 }, gradSteps: 1 }), ga8 = f({ numGpus: 4, opt: { zero_stage: 2 }, gradSteps: 8 });
  check('Gradient accumulation does not reduce ZeRO-2 tok/s below 1-step', ga8.perf.trainTps >= ga1.perf.trainTps * 0.95);
  const pc = f({ numGpus: 4, interconnect: 'pcie4', opt: { zero_stage: 3 } }), nvl = f({ numGpus: 4, interconnect: 'nvlink_gen4', opt: { zero_stage: 3 } });
  check('ZeRO-3 over PCIe slower than over NVLink', pc.perf.trainTps < nvl.perf.trainTps);
  exact('Training time = data tokens ÷ tok/s', 'tokens / tps / 3600', ck1.perf.dataTok / ck1.perf.trainTps / 3600, ck1.perf.hours, 1);
  const fp8t = f({ ftQuant: 'fp8' }, 'llama-3-2-3b'), bf16t = f({}, 'llama-3-2-3b');
  check('FP8 training faster than BF16 on H100 (Llama-3.2-3B full FT)', fp8t.perf.trainTps > bf16t.perf.trainTps);
  check('70B full FT on 8×H100 ZeRO-3 does not fit (needs offload / more GPUs)', z3.memory.pct > 100);
  const z3o = f({ numGpus: 8, opt: { zero_stage: 3, offload_optimizer: true } }, 'llama-3-1-70b');
  check('…and fits with optimizer offload', z3o.memory.pct < 100);
}

/* =========================================================================================================== */
section('H. Diffusion');
{
  const d = (slug, gpu, o) => run(DIF, slug, gpu, o);
  exact('FLUX 1536×1024 latent tokens', '(1536/16)·(1024/16)', 96 * 64, d('flux-1-dev', 'h100_80', { imageWidth: 1536 }).perf.latentTokens, 0.01);
  exact('SDXL 1024² UNet tokens (8× VAE, ÷4 at the bottleneck)', '(1024/32)²', 1024, d('sdxl-1-0-base', 'h100_80').perf.latentTokens, 0.01);
  exact('Wan 832×480×81 tokens', '21 · 52 · 30', 21 * 52 * 30, d('wan-2-1-t2v-1-3b', 'h100_80', { imageWidth: 832, imageHeight: 480, numFrames: 81 }).perf.latentTokens, 0.01);
  exact('Wan 1 frame = 1 latent frame', '1 · 80 · 45', 80 * 45, d('wan-2-1-t2v-1-3b', 'h100_80', { imageWidth: 1280, imageHeight: 720, numFrames: 1 }).perf.latentTokens, 0.01);
  exact('Qwen-Image-2.1 (16× VAE, patch 1) 1024²', '(1024/16)²', 4096, d('qwen-image-2-1', 'h100_80').perf.latentTokens, 0.01);
  exact('FLUX.2-dev FP8 weights', '32e9 × 1', 32e9 / GiB, seg(d('flux-2-dev', 'h100_80', { quant: 'fp8' }), 'weights'), 0.01);
  const no = d('flux-1-dev', 'h100_80', { encoderOffload: false });
  exact('Encoders resident when offload is off: 4.9B × 2', '4.9e9 × 2', 4.9e9 * 2 / GiB, seg(no, 'enc'), 0.01);
  const yes = d('flux-1-dev', '4090_24');
  check('With model offload, FLUX.1-dev BF16 fits a 24 GB card', yes.memory.pct <= 100, `${yes.memory.perGpu.toFixed(2)} GiB`);
  check('…and is "at the limit" (>90%), matching mixed OOM reports', yes.memory.pct > 90);
  const v = d('wan-2-1-t2v-14b', 'h100_80', { imageWidth: 1280, imageHeight: 720, numFrames: 81, encoderOffload: false }), vt = d('wan-2-1-t2v-14b', 'h100_80', { imageWidth: 1280, imageHeight: 720, numFrames: 81, encoderOffload: false, vaeTiling: true });
  check('VAE tiling lowers peak memory when VAE decode is the peak', vt.memory.perGpu <= v.memory.perGpu);
  const cfg1 = d('stable-diffusion-3-5-large', 'h100_80', { guidanceScale: 1 }), cfg2 = d('stable-diffusion-3-5-large', 'h100_80', { guidanceScale: 5 });
  exact('CFG doubles FLOPs per step (SD3.5-L)', '2×', 2 * tr(cfg1, 'FLOPs per step'), tr(cfg2, 'FLOPs per step'), 1);
  const fd = d('flux-1-dev', 'h100_80', { guidanceScale: 5 }), fd1 = d('flux-1-dev', 'h100_80', { guidanceScale: 1 });
  exact('FLUX.1-dev is guidance-distilled: guidance scale does not add a pass', '1×', tr(fd1, 'FLOPs per step'), tr(fd, 'FLOPs per step'), 0.01);
  const w1 = d('wan-2-1-t2v-1-3b', 'h100_80', { guidanceScale: 1, imageWidth: 832, imageHeight: 480, numFrames: 81 }), w5 = d('wan-2-1-t2v-1-3b', 'h100_80', { guidanceScale: 5, imageWidth: 832, imageHeight: 480, numFrames: 81 });
  check('Wan CFG passes run sequentially: activation memory unchanged by CFG', Math.abs(w1.memory.perGpu - w5.memory.perGpu) < 0.01);
  const s1 = d('stable-diffusion-3-5-large', 'h100_80', { guidanceScale: 1, encoderOffload: false }), s5 = d('stable-diffusion-3-5-large', 'h100_80', { guidanceScale: 5, encoderOffload: false });
  check('SD3 CFG is batched: more activation memory with CFG', s5.memory.perGpu >= s1.memory.perGpu);
  const bb1 = d('flux-1-dev', 'h100_80'), bb2 = d('flux-1-dev', 'h100_80', { batch: 2 });
  exact('Batch 2 doubles FLOPs per step', '2×', 2 * tr(bb1, 'FLOPs per step'), tr(bb2, 'FLOPs per step'), 1);
  const st20 = d('flux-1-dev', 'h100_80', { inferenceSteps: 20 }), st40 = d('flux-1-dev', 'h100_80', { inferenceSteps: 40 });
  exact('Time scales with steps (minus fixed VAE + encode)', 'Δ20 steps = 20 × step', 20 * st20.perf.stepMs / 1000, st40.perf.seconds - st20.perf.seconds, 0.5);
  check('FP8 diffusion faster on H100 than BF16', d('flux-1-dev', 'h100_80', { quant: 'fp8' }).perf.seconds < bb1.perf.seconds);
  check('FP8 diffusion on A100 is not faster (no FP8 cores)', Math.abs(d('flux-1-dev', 'a100_80', { quant: 'fp8' }).perf.stepMs - d('flux-1-dev', 'a100_80').perf.stepMs) < 1e-6);
  check('Higher-TFLOPS GPU is never slower (B200 < H100 < A100 < 4090 step time)', (() => { const t = ['b200_192', 'h100_80', 'a100_80', '4090_24'].map((g) => d('flux-1-dev', g).perf.stepMs); return t[0] < t[1] && t[1] < t[2] && t[2] < t[3]; })());
}

/* =========================================================================================================== */
section('I. Inputs, warnings, runtime / format rules');
{
  const r0 = run(INF, 'llama-3-1-8b', 'h100_80', { batch: 0, users: 0, numGpus: 2.6, seq: 4096, outputTokens: 99999 });
  check('batch 0 / users 0 sanitized to 1 (finite TTFT)', r0.perf && Number.isFinite(r0.perf.ttft));
  check('numGpus 2.6 rounds to 3', r0.layout.n === 3);
  exact('outputTokens ≥ seq clamps to seq − 1', 'seq − 1', 4095, r0.perf.outTok, 0.01);
  check('Context overflow warns', warned(run(INF, 'glm-4-9b', 'h100_80', { seq: 16384 }), /exceeds the model/));
  check('vLLM + Q8_0 KV warns (vLLM: FP16/FP8 only)', warned(run(INF, 'llama-3-1-8b', 'h100_80', { kvQuant: 'q8_0' }), /vLLM has no/));
  check('vLLM + FP8 KV is fine', !warned(run(INF, 'llama-3-1-8b', 'h100_80', { kvQuant: 'fp8' }), /has no/));
  check('Ollama + FP8 KV warns (f16/q8_0/q4_0 only)', warned(run(INF, 'llama-3-1-8b', '4090_24', { runtime: 'ollama', quant: 'q4_k_m', kvQuant: 'fp8' }), /Ollama has no/));
  check('llama.cpp + Q4_0 KV is fine', !warned(run(INF, 'llama-3-1-8b', '4090_24', { runtime: 'llamacpp', quant: 'q4_k_m', kvQuant: 'q4_0' }), /has no/));
  check('TensorRT-LLM NVFP4 KV on H100 warns (Blackwell only)', warned(run(INF, 'llama-3-1-8b', 'h100_80', { runtime: 'trtllm', kvQuant: 'nvfp4' }), /needs a Blackwell/));
  check('llama.cpp cannot load AWQ INT4', warned(run(INF, 'llama-3-1-8b', '4090_24', { runtime: 'llamacpp', quant: 'int4' }), /cannot load/));
  check('MLX cannot load FP8', warned(run(INF, 'llama-3-1-8b', 'm4_max_128', { runtime: 'mlx', quant: 'fp8' }), /cannot load/));
  check('GGUF on vLLM warns', warned(run(INF, 'llama-3-1-8b', 'h100_80', { quant: 'q4_k_m' }), /GGUF/));
  check('NVFP4 on H100 warns (dequantized)', warned(run(INF, 'llama-3-1-8b', 'h100_80', { quant: 'nvfp4' }), /NVFP4 is accelerated only on Blackwell/));
  check('MXFP4 on A100 warns', warned(run(INF, 'gpt-oss-20b', 'a100_80', { quant: 'mxfp4' }), /MXFP4 is accelerated only/));
  check('MLX on NVIDIA warns', warned(run(INF, 'llama-3-1-8b', 'h100_80', { runtime: 'mlx' }), /MLX only runs on Apple/));
  check('TP 8 > 4 KV heads warns about replication', warned(run(INF, 'qwen3-30b-a3b', 'h100_80', { numGpus: 8 }), /exceeds the model's 4 KV heads/));
  check('Auto runtime: GGUF on a 4090 → Ollama', run(INF, 'llama-3-1-8b', '4090_24', { runtime: 'auto', quant: 'q4_k_m' }).runtime === 'ollama');
  check('Auto runtime: Apple → MLX', run(INF, 'llama-3-1-8b', 'm4_max_128', { runtime: 'auto', quant: 'int4' }).runtime === 'mlx');
  check('Auto runtime: AMD APU + AWQ → not llama.cpp', run(INF, 'llama-3-1-8b', D.gpus.find((g) => g.type === 'apu' && g.vendor === 'amd'), { runtime: 'auto', quant: 'int4' }).runtime !== 'llamacpp');
  check('Does-not-fit returns no speed numbers', run(INF, 'llama-3-1-405b', '4090_24').perf === null);
}

/* =========================================================================================================== */
section('J. Dataset & hardware integrity');
{
  const textModels = D.models.filter((m) => m.task_type === 'text_generation');
  check('Every model computes on B300 ×8 with finite numbers', D.models.every((m) => { const r = run(m.task_type === 'text_generation' ? INF : DIF, m, 'b300_288', { numGpus: m.task_type === 'text_generation' ? 8 : 1, quant: 'fp8' }); return Number.isFinite(r.memory.perGpu) && r.memory.perGpu > 0; }));
  check('Every GPU computes Llama-3.2-1B Q4_K_M with finite tok/s', D.gpus.filter((g) => !g.key.startsWith('custom')).every((g) => { const r = run(INF, 'llama-3-2-1b', g, { quant: 'q4_k_m', runtime: 'auto', seq: 2048 }); return !r.perf || Number.isFinite(r.perf.tps); }));
  check('MoE shared ≥ embeddings + softmax attention (no attention in the routed pool)', textModels.filter((m) => C.geometry(m).moe).every((m) => { const g = C.geometry(m); return g.shared + 1 >= Math.min(g.P, g.embedParams + g.softL * g.attnParams); }));
  check('MoE matmul-active params ≤ total', textModels.every((m) => { const g = C.geometry(m); return g.Amm <= g.P + 1; }));
  check('Active matmul params within 15% of published active count (MoE)', textModels.filter((m) => C.geometry(m).moe).every((m) => { const g = C.geometry(m); return Math.abs(g.Amm / g.A - 1) < 0.15; }), textModels.filter((m) => C.geometry(m).moe).filter((m) => { const g = C.geometry(m); return Math.abs(g.Amm / g.A - 1) >= 0.15; }).map((m) => m.slug).join(', '));
  check('No GPU with zero/negative memory, bandwidth or TFLOPS', D.gpus.every((g) => g.memory > 0 && g.bandwidth_gbs > 0 && g.fp16_tflops > 0));
  check('Usable memory never exceeds raw memory', D.gpus.every((g) => !g.usable_memory || g.usable_memory <= g.memory));
  check('No model dated after 2026-11-22', D.models.every((m) => !m.release_date || m.release_date <= '2026-11-22'));
  check('DSA/CSA models all have a top-k (default 2048)', textModels.filter((m) => /dsa|csa2/.test(m.attention_structure)).every((m) => C.kvLayout(m, C.geometry(m)).topk > 0));
  check('Sliding-window ratios are in (0, 1]', textModels.filter((m) => m.uses_sliding_window).every((m) => (m.sliding_window_ratio ?? 1) > 0 && (m.sliding_window_ratio ?? 1) <= 1));
}

/* ---------- report ---------- */
const byArea = {};
for (const r of results) (byArea[r.area] = byArea[r.area] || []).push(r);
let pass = 0;
for (const [a, rs] of Object.entries(byArea)) {
  console.log(`\n${a}  (${rs.filter((r) => r.pass).length}/${rs.length})`);
  for (const r of rs) { if (r.pass) pass++; console.log(`  ${r.pass ? '✓' : '✗'} ${r.name}${r.detail && (!r.pass || process.env.VERBOSE) ? `  — ${r.detail}` : ''}`); }
}
console.log(`\n${results.length} scenario tests, ${pass} passed, ${results.length - pass} failed`);
if (pass !== results.length) process.exitCode = 1;
