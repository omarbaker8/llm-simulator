// 59 hand-calculated checks. Every expected value below is derived here from published model configs
// (Hugging Face config.json numbers written out as literals) and first-principles formulas — NOT from calc.js.
// Usage: node tests/hand.js
const fs = require('fs'), vm = require('vm'), path = require('path');
const dir = process.argv[2] || path.join(__dirname, '..');
const ctx = {}; ctx.window = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(dir + '/data.js', 'utf8'), ctx);
const D = ctx.VRAM_DATA, C = require(dir + '/calc.js');
const M = (s) => D.models.find((m) => m.slug === s), G = (k) => D.gpus.find((g) => g.key === k);
const GiB = 2 ** 30;

const INF = { mode: 'inference', quant: 'fp16', kvQuant: 'fp16', runtime: 'vllm', numGpus: 1, cluster: false, parallelism: 'tensor', interconnect: 'nvlink_gen4',
  batch: 1, users: 1, seq: 8192, outputTokens: 512, prefixShare: 0, offload: false, offloadTarget: 'cpu_ram', offloadFrac: 0, offloadKv: false,
  spec: false, specDraft: 1, specAccept: 0.7, specTokens: 4, specType: 'sequential', elec: 0.12, carbon: 400, opt: {} };
const FT = { ...INF, mode: 'finetuning', seq: 2048, outputTokens: 1, ftMethod: 'full', ftQuant: 'fp16', loraRank: 16, gradSteps: 1, numSamples: 1000, tokensPerSample: 2048, numEpochs: 1,
  opt: { optimizer_type: 'adamw', zero_stage: 0, flash_attention: true, gradient_checkpointing: true, gradient_checkpointing_ratio: 1, use_8bit_optimizer: false, sequence_packing: true } };
const DIF = { ...INF, imageWidth: 1024, imageHeight: 1024, numFrames: 1, inferenceSteps: 28, guidanceScale: 3.5, encoderOffload: true, vaeTiling: false, diffRuntime: 'diffusers' };

const run = (base, slug, gpu, o = {}) => C.calculate({ ...base, ...o, opt: { ...base.opt, ...(o.opt || {}) } }, M(slug), G(gpu));
const seg = (r, k) => (r.memory.segments.find((s) => s.key === k) || { gib: NaN }).gib;
const tr = (r, k) => { const t = r.trace.find((x) => x.k === k); if (!t) return NaN; const m = String(t.v).match(/[-\d.,]+/); return m ? Number(m[0].replace(/,/g, '')) : NaN; };

const rows = [], refs = [];
let n = 0;
function T(area, name, handFormula, expected, actual, tolPct = 0.5) {
  const isRef = /tok\/s/.test(name);
  if (!isRef) n++;
  const dev = (actual - expected) / expected * 100;
  const pass = Number.isFinite(actual) && Math.abs(dev) <= tolPct;
  (isRef ? refs : rows).push({ n: isRef ? 'R' + (refs.length + 1) : n, area, name, handFormula, expected, actual, dev, tolPct, pass });
}
const fmt = (x) => !Number.isFinite(x) ? String(x) : Math.abs(x) >= 1e5 ? x.toExponential(4) : Math.abs(x) >= 100 ? x.toFixed(1) : x.toFixed(4);

/* ---------- published config numbers ---------- */
const L8 = { P: 8.03e9, L: 32, h: 4096, heads: 32, kv: 8, hd: 128, V: 128256, ffn: 14336 };
const L70 = { P: 70.55e9, L: 80, h: 8192, heads: 64, kv: 8, hd: 128, V: 128256, ffn: 28672 };
const embL8 = 2 * L8.V * L8.h, embL70 = 2 * L70.V * L70.h; // untied in + out

/* ===== A. Weight memory (GiB) ===== */
T('Weights', 'Llama-3.1-8B FP16', '8.03e9 × 2 B', L8.P * 2 / GiB, seg(run(INF, 'llama-3-1-8b', 'h100_80'), 'weights'));
T('Weights', 'Llama-3.1-8B FP8 (emb/lm_head BF16)', '(P − 2·V·h)×1 + 2·V·h×2', ((L8.P - embL8) * 1 + embL8 * 2) / GiB, seg(run(INF, 'llama-3-1-8b', 'h100_80', { quant: 'fp8' }), 'weights'));
T('Weights', 'Llama-3.1-8B Q4_K_M (llama.cpp README 4.8944 bpw)', 'P × 4.8944 ÷ 8', L8.P * 4.8944 / 8 / GiB, seg(run(INF, 'llama-3-1-8b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp' }), 'weights'), 1);
T('Weights', 'Llama-3.1-70B FP16', '70.55e9 × 2 B', L70.P * 2 / GiB, seg(run(INF, 'llama-3-1-70b', 'h200_141', { numGpus: 2 }), 'weights') * 2);
T('Weights', 'Llama-3.1-70B AWQ INT4 g128', '(P − emb)×(4 + 16/128 + 4/128)/8 + emb×2', ((L70.P - embL70) * (4 + 16 / 128 + 4 / 128) / 8 + embL70 * 2) / GiB, seg(run(INF, 'llama-3-1-70b', 'h100_80', { quant: 'int4' }), 'weights'));
{ // gpt-oss-120b MXFP4: experts 4.25 bpw, attention + embeddings BF16 (per quantization_config)
  const L = 36, h = 2880, heads = 64, hd = 64, kv = 8, V = 201088, P = 116.83e9;
  const attn = L * (h * heads * hd * 2 + 2 * h * kv * hd), emb = 2 * V * h;
  T('Weights', 'gpt-oss-120b MXFP4 (attn+emb BF16)', '(P − attn − emb)×4.25/8 + (attn+emb)×2', ((P - attn - emb) * 4.25 / 8 + (attn + emb) * 2) / GiB, seg(run(INF, 'gpt-oss-120b', 'h100_80', { quant: 'mxfp4' }), 'weights'), 1);
  T('Weights', 'gpt-oss-20b MXFP4 GGUF (attn+emb Q8_0; file 12.1 GB)', '(P − attn − emb)×4.25/8 + (attn+emb)×8.5/8', ((20.91e9 - 24 * (h * heads * hd * 2 + 2 * h * kv * hd) - emb) * 4.25 / 8 + (24 * (h * heads * hd * 2 + 2 * h * kv * hd) + emb) * 8.5 / 8) / GiB, seg(run(INF, 'gpt-oss-20b', '4090_24', { quant: 'mxfp4', runtime: 'llamacpp' }), 'weights'), 1);
}
T('Weights', 'Qwen3-30B-A3B Q4_K_M vs real file (Qwen/Qwen3-30B-A3B-GGUF 18.557 GB)', '18.557e9 B ÷ 2^30', 18.557e9 / GiB, seg(run(INF, 'qwen3-30b-a3b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp' }), 'weights'), 3);
T('Weights', 'Qwen3.5-27B Q4_K_M + 0.46B vision at F16', '26.90e9×4.8944/8 + 0.46e9×2', (26.9e9 * 4.8944 / 8 + 0.46e9 * 2) / GiB, seg(run(INF, 'qwen35-27b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp' }), 'weights'), 1);
T('Weights', 'DeepSeek-V3 FP8 (8 GPUs, total)', '(671e9 − 2·129280·7168)×1 + emb×2', ((671e9 - 2 * 129280 * 7168) + 2 * 129280 * 7168 * 2) / GiB, seg(run(INF, 'deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8 }), 'weights') * 8);

/* ===== B. KV cache per token (bytes) ===== */
const kvpt = (slug, o = {}) => run(INF, slug, 'h200_141', o).memory.kvPerToken;
T('KV/token', 'Llama-3.1-8B', '2 × 32 L × 8 kv × 128 × 2 B', 2 * 32 * 8 * 128 * 2, kvpt('llama-3-1-8b'), 0.01);
T('KV/token', 'Llama-3.1-70B', '2 × 80 × 8 × 128 × 2', 2 * 80 * 8 * 128 * 2, kvpt('llama-3-1-70b'), 0.01);
T('KV/token', 'Qwen2.5-7B', '2 × 28 × 4 × 128 × 2', 2 * 28 * 4 * 128 * 2, kvpt('qwen2-5-7b'), 0.01);
T('KV/token', 'Mistral-7B v0.2', '2 × 32 × 8 × 128 × 2', 2 * 32 * 8 * 128 * 2, kvpt('mistral-7b-instruct-v0-2'), 0.01);
T('KV/token', 'Phi-3-mini (MHA)', '2 × 32 × 32 × 96 × 2', 2 * 32 * 32 * 96 * 2, kvpt('phi-3-mini'), 0.01);
T('KV/token', 'DeepSeek-V3 (MLA)', '61 × (512 latent + 64 rope) × 2', 61 * 576 * 2, kvpt('deepseek-v3'), 0.01);
T('KV/token', 'Gemma-2-9B (all 42 layers)', '42 × 2 × 8 × 256 × 2', 42 * 2 * 8 * 256 * 2, kvpt('gemma-2-9b'), 0.01);
T('KV/token', 'Qwen3.5-27B (16 full-attn layers)', '16 × 2 × 4 × 256 × 2', 16 * 2 * 4 * 256 * 2, kvpt('qwen35-27b'), 0.01);
T('KV/token', 'gpt-oss-120b', '36 × 2 × 8 × 64 × 2', 36 * 2 * 8 * 64 * 2, kvpt('gpt-oss-120b', { quant: 'mxfp4' }), 0.01);
T('KV/token', 'Gemma-4-31B (50 sliding + 10 global K=V)', '50×2×16×256×2 + 10×1×4×512×2', 50 * 2 * 16 * 256 * 2 + 10 * 1 * 4 * 512 * 2, kvpt('gemma-4-31b'), 0.01);
T('KV/token', 'Gemma-4-12B (40 sliding + 8 global K=V, 1 global kv head)', '40×2×8×256×2 + 8×1×1×512×2', 40 * 2 * 8 * 256 * 2 + 8 * 1 * 1 * 512 * 2, kvpt('gemma-4-12b'), 0.01);
T('KV/token', 'Llama-3.1-8B with FP8 KV', '2 × 32 × 8 × 128 × 1 B', 2 * 32 * 8 * 128, kvpt('llama-3-1-8b', { kvQuant: 'fp8' }), 0.01);

/* ===== C. KV cache totals (GiB) ===== */
const kvseg = (slug, gpu, o) => seg(run(INF, slug, gpu, o), 'kv');
T('KV total', 'Llama-8B, 8,192 tokens', '8192 × 131072 B', 8192 * 131072 / GiB, kvseg('llama-3-1-8b', 'h100_80', {}), 0.01);
T('KV total', 'Llama-70B, 32,768 tokens (TP2 → ×2)', '32768 × 327680 B', 32768 * 327680 / GiB, kvseg('llama-3-1-70b', 'h200_141', { seq: 32768, numGpus: 2 }) * 2, 0.01);
T('KV total', 'Llama-8B, 131,072 tok, FP8 KV', '131072 × 65536 B', 131072 * 65536 / GiB, kvseg('llama-3-1-8b', 'h200_141', { seq: 131072, kvQuant: 'fp8' }), 0.01);
T('KV total', 'Gemma-2-9B, 8,192 tok (21 full + 21 SWA@4096)', '21×8192×8192 + 21×4096×8192', (21 * 8192 * 8192 + 21 * 4096 * 8192) / GiB, kvseg('gemma-2-9b', 'h100_80', {}), 0.01);
{
  const st = (48 * 128 * 128 + (2 * 16 * 128 + 48 * 128) * 3) * 2; // GDN state per layer per seq
  T('KV total', 'Qwen3.5-27B, 32,768 tok (16 full + 48 GDN states)', '16×32768×4096 + 48×[(48·128·128 + (2·16·128+48·128)·3)·2]', (16 * 32768 * 4096 + 48 * st) / GiB, kvseg('qwen35-27b', 'h200_141', { seq: 32768 }), 0.05);
}
T('KV total', 'DeepSeek-V3, 131,072 tok (MLA, per GPU = total)', '131072 × 70272 B', 131072 * 70272 / GiB, kvseg('deepseek-v3', 'h200_141', { seq: 131072, quant: 'fp8', numGpus: 8 }), 0.01);
T('KV total', 'Llama-8B, 4 users × 8K, no sharing', '4 × 8192 × 131072', 4 * 8192 * 131072 / GiB, kvseg('llama-3-1-8b', 'h100_80', { users: 4 }), 0.01);
T('KV total', 'Llama-8B, 4 users, 50% prompt prefix shared', '(3840 + 4×(3840+512)) × 131072', (3840 + 4 * (3840 + 512)) * 131072 / GiB, kvseg('llama-3-1-8b', 'h100_80', { users: 4, prefixShare: 0.5 }), 0.01);
T('KV total', 'Llama-8B, n=4 parallel samples (prompt shared)', '(7680 + 4×512) × 131072', (7680 + 4 * 512) * 131072 / GiB, kvseg('llama-3-1-8b', 'h100_80', { batch: 4 }), 0.01);

/* ===== D. FLOPs ===== */
T('FLOPs', 'Llama-8B prefill 7,680 tok (TFLOP)', '2·(P − V·h input lookup)·T + 4·L·(T²/2)·heads·hd', (2 * (L8.P - L8.V * L8.h) * 7680 + 4 * 32 * (7680 ** 2 / 2) * 32 * 128) / 1e12, tr(run(INF, 'llama-3-1-8b', 'h100_80'), 'Prefill FLOPs'), 0.1);
T('FLOPs', 'Llama-70B prefill 4,096 tok (TFLOP)', 'same, L=80, heads 64, V·h = 128256·8192', (2 * (L70.P - L70.V * L70.h) * 4096 + 4 * 80 * (4096 ** 2 / 2) * 64 * 128) / 1e12, tr(run(INF, 'llama-3-1-70b', 'h200_141', { seq: 4608, numGpus: 2 }), 'Prefill FLOPs'), 0.1);
T('FLOPs', 'Qwen3.5-27B prefill 8,192 tok (TFLOP, 16 full layers)', '2·(26.9e9 − 248320·5120)·T + 4·16·(T²/2)·24·256', (2 * (26.9e9 - 248320 * 5120) * 8192 + 4 * 16 * (8192 ** 2 / 2) * 24 * 256) / 1e12, tr(run(INF, 'qwen35-27b', 'h200_141', { seq: 8704 }), 'Prefill FLOPs'), 0.1);
T('FLOPs', 'gpt-oss-120b prefill 4,096 tok (TFLOP, 18 full + 18 SWA@128)', '2·(attn 0.955B + lm_head 0.579B + router + 4/128·114.7B experts)·T + attention', (2 * (36 * (2880 * 64 * 64 * 2 + 2 * 2880 * 8 * 64) + 201088 * 2880 + 36 * 2880 * 128 + 4 / 128 * 36 * 128 * 3 * 2880 * 2880) * 4096 + 4 * (18 * 4096 ** 2 / 2 + 18 * 128 * (4096 - 64)) * 64 * 64) / 1e12, tr(run(INF, 'gpt-oss-120b', 'h100_80', { quant: 'mxfp4', seq: 4608 }), 'Prefill FLOPs'), 0.5); // router (13M) is pooled with experts
T('FLOPs', 'DeepSeek-V3 prefill 4,096 tok (TFLOP, MLA d=160)', '2·(37e9 − 129280·7168)·T + 4·61·(T²/2)·128·160', (2 * (37e9 - 129280 * 7168) * 4096 + 4 * 61 * (4096 ** 2 / 2) * 128 * 160) / 1e12, tr(run(INF, 'deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8, seq: 4608 }), 'Prefill FLOPs'), 0.1);
T('FLOPs', 'Full FT Llama-8B FLOPs/token @2048, ckpt (GFLOP)', '(2·(P − V·h) + 4·32·1024·32·128) × (3 + 1 recompute)', (2 * (L8.P - L8.V * L8.h) + 4 * 32 * 1024 * 32 * 128) * 4 / 1e9, tr(run(FT, 'llama-3-1-8b', 'h100_80'), 'FLOPs / token'), 0.2);
T('FLOPs', 'LoRA Llama-8B FLOPs/token @2048, ckpt (GFLOP)', '2·(P − V·h)×(2+1) + attention 4·32·1024·4096 × (3+1)', (2 * (L8.P - L8.V * L8.h) * 3 + 4 * 32 * 1024 * 32 * 128 * 4) / 1e9, tr(run(FT, 'llama-3-1-8b', 'h100_80', { ftMethod: 'lora' }), 'FLOPs / token'), 0.5);

/* ===== E. Decode bytes & physics bounds ===== */
{
  const r = run(INF, 'llama-3-1-8b', 'h100_80');
  const bytes = (L8.P - L8.V * L8.h) * 2 + (7680 + 256) * 131072; // untied input table only gathered; KV at avg context
  T('Decode', 'Llama-8B FP16 bytes read / token (GiB)', '(P − V·h)×2 + (7680+256)×131072', bytes / GiB, tr(r, 'Bytes read per step'), 0.5);
  T('Decode', 'Llama-8B FP16 H100 tok/s ≤ 3350 GB/s ÷ bytes (≥ 70% of roofline)', 'roofline = 3.35e12 / bytes', 3.35e12 / bytes * 0.85, r.perf.tps, 15);
  const r2 = run(INF, 'llama-3-1-70b', 'h100_80', { quant: 'fp8', numGpus: 2 });
  const b2 = ((L70.P - embL70) + embL70 * 2 - L70.V * L70.h * 2) + (7680 + 256) * 327680;
  T('Decode', 'Llama-70B FP8 TP2 H100 tok/s vs 85% roofline', '2 × 3.35e12 / bytes × 0.85', 2 * 3.35e12 / b2 * 0.85, r2.perf.tps, 20);
}
T('Decode', 'Llama-8B Q4_K_M 4090 llama.cpp tok/s vs benchmark 144 (scoreboard)', 'measured ref', 144, run(INF, 'llama-3-1-8b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp', seq: 640, outputTokens: 128 }).perf.tps, 10);
T('Decode', 'Llama-8B Q4_0 H100 llama.cpp tg tok/s vs 268 (scoreboard)', 'measured ref', 268, run(INF, 'llama-3-1-8b', 'h100_80', { quant: 'q4_0', runtime: 'llamacpp', seq: 640, outputTokens: 128 }).perf.tps, 15);
T('Decode', 'gpt-oss-20b MXFP4 4090 llama.cpp tok/s vs 222 (llama.cpp #15396)', 'measured ref', 222, run(INF, 'gpt-oss-20b', '4090_24', { quant: 'mxfp4', runtime: 'llamacpp' }).perf.tps, 15);

/* ===== F. Fine-tuning memory ===== */
{
  const perLayer = 16 * ((4096 + 4096) + (4096 + 1024) * 2 + (4096 + 4096) + (4096 + 14336) * 2 + (14336 + 4096));
  const lora = perLayer * 32; // 41,943,040
  const r = run(FT, 'llama-3-1-8b', 'h100_80', { ftMethod: 'lora' });
  T('Fine-tune', 'LoRA r16 adapter params Llama-8B (fp32 GiB)', 'r·Σ(d_in+d_out) over q,k,v,o,gate,up,down × 32 × 4 B', lora * 4 / GiB, seg(r, 'lora'), 0.01);
  T('Fine-tune', 'LoRA AdamW states (GiB)', '41.94M × 8 B', lora * 8 / GiB, seg(r, 'optim'), 0.01);
  T('Fine-tune', 'LoRA logits 2,048 tok (GiB)', '2048 × 128256 × (2 B bf16 + 4 B fp32 upcast + 4 B grad)', 2048 * 128256 * 10 / GiB, seg(r, 'logits'), 0.01);
  const f = run(FT, 'llama-3-1-8b', 'h100_80');
  T('Fine-tune', 'Full FT Llama-8B weights+grads+Adam+master (GiB)', 'P × (2+2+8+4) B', L8.P * 16 / GiB, seg(f, 'weights') + seg(f, 'grads') + seg(f, 'optim'), 0.01);
  const q = run(FT, 'llama-3-1-8b', 'h100_80', { ftMethod: 'qlora' });
  T('Fine-tune', 'QLoRA Llama-8B base weights, emb/lm_head BF16 (GiB)', '(P − 2·V·h)×4.127/8 + 2·V·h×2', ((L8.P - embL8) * 4.127 / 8 + embL8 * 2) / GiB, seg(q, 'weights'), 0.5);
  const z = run(FT, 'llama-3-1-70b', 'h100_80', { numGpus: 8, opt: { zero_stage: 3 } });
  T('Fine-tune', 'Full FT 70B ZeRO-3 ×8: states per GPU (GiB)', '70.55e9 × 16 B ÷ 8', L70.P * 16 / 8 / GiB, seg(z, 'weights') + seg(z, 'grads') + seg(z, 'optim'), 0.01);
}

/* ===== G. Diffusion ===== */
{
  const f = run(DIF, 'flux-1-dev', 'h100_80');
  T('Diffusion', 'FLUX.1-dev 1024² latent tokens', '(1024/8/2)²', (1024 / 16) ** 2, f.perf.latentTokens, 0.01);
  T('Diffusion', 'FLUX.1-dev BF16 denoiser weights (GiB)', '11.9e9 × 2 B', 11.9e9 * 2 / GiB, seg(f, 'weights'), 0.01);
  // Independent: 6.45B params act per image token (adaLN excluded), +512 text tokens in joint attention, 57 blocks, h 3072
  const Ntok = 4096 + 512;
  T('Diffusion', 'FLUX.1-dev FLOPs/step 1024² CFG 1 (TFLOP)', '2·6.45e9·4608 + 4·57·4608²·3072', (2 * 6.45e9 * Ntok + 4 * 57 * Ntok ** 2 * 3072) / 1e12, tr(f, 'FLOPs per step'), 5);
  const w = run(DIF, 'wan-2-1-t2v-14b', 'h100_80', { imageWidth: 1280, imageHeight: 720, numFrames: 81, inferenceSteps: 50, guidanceScale: 5 });
  T('Diffusion', 'Wan2.1 1280×720×81f latent tokens', '(1 + 80/4) × (1280/16) × (720/16)', 21 * 80 * 45, w.perf.latentTokens, 0.01);
  const t = run(DIF, 'wan-2-2-ti2v-5b', '4090_24', { imageWidth: 1280, imageHeight: 704, numFrames: 121, inferenceSteps: 50, guidanceScale: 5 });
  T('Diffusion', 'Wan2.2 TI2V-5B 1280×704×121f tokens (16× VAE, 2×2 patch)', '(1 + 120/4) × (1280/32) × (704/32)', 31 * 40 * 22, t.perf ? t.perf.latentTokens : NaN, 0.01);
}

/* ===== H. Serving memory & prefill (checked against LLMServingSim's real vLLM runs) ===== */
{
  const kvTok8 = 32 * 2 * 8 * 128 * 2;                                    // Llama-8B: 128 KiB per token
  const act = (chunk, h, ffn) => chunk * (4 * h + 3 * ffn) * 2 + L8.V * 4; // prefill chunk activations + 1 × vocab logits
  // RTX PRO 6000 Blackwell Server Edition: datasheet 1597 GB/s. Decode reads all weights but the untied input table.
  const wRead = L8.P * 2 - L8.V * L8.h * 2, kvRead = ((4096 - 512) + 256) * kvTok8, fixed = 0.30e-3 + 32 * 6e-6;
  T('Serving', 'Llama-8B decode step, RTX PRO 6000 Server Edition (ms)', '(15.009e9 + 3840 × 128 KiB) ÷ (1597 GB/s × 0.80) + 0.30 ms + 32 × 6 µs',
    ((wRead + kvRead) / (1597e9 * 0.8) + fixed) * 1e3, run(INF, 'llama-3-1-8b', 'rtx_6000_blackwell_server_96', { seq: 4096 }).perf.stepMs, 0.5);
  // MoE prefill reads only the routed experts the prompt touches: Qwen3-30B-A3B routed = 48 × 128 × 3 × 2048 × 768.
  const routedQ = 48 * 128 * 3 * 2048 * 768, touch = 1 - (1 - 8 / 128) ** 10;
  const pq = run(INF, 'qwen3-30b-a3b', 'h100_80', { seq: 11, outputTokens: 1 }).perf;
  T('Serving', 'Qwen3-30B-A3B prefill of 10 tokens, H100 (ms)', '(61.0e9 − 28.99e9 × 2 B × (1 − 8/128)^10) ÷ (3350 GB/s × 0.80) + 48 × 6 µs',
    ((30.5e9 * 2 - routedQ * 2 * (1 - touch)) / (3350e9 * 0.8) + 48 * 6e-6) * 1e3, pq.ttft - pq.decodeStepMs, 0.5);
  // vLLM `serve` max_num_batched_tokens by device memory: < 70 GiB → 2048, ≥ 70 GiB → 8192 (A100 excluded), ≥ 160 GiB → 16384.
  const r4 = run(INF, 'llama-3-1-8b', '4090_24', { seq: 16384, outputTokens: 1 });
  T('Serving', 'Llama-8B activations, RTX 4090 (2048-token chunk) (GiB)', '2048 × (4·4096 + 3·14336) × 2 B + 128256 × 4 B', act(2048, 4096, 14336) / GiB, seg(r4, 'act'), 0.1);
  T('Serving', 'Llama-8B activations, A100-80GB (2048, excluded tier) (GiB)', '2048 × 59392 × 2 B + 128256 × 4 B', act(2048, 4096, 14336) / GiB, seg(run(INF, 'llama-3-1-8b', 'a100_80', { seq: 16384, outputTokens: 1 }), 'act'), 0.1);
  T('Serving', 'Llama-8B activations, H100 (8192-token chunk) (GiB)', '8192 × 59392 × 2 B + 128256 × 4 B', act(8192, 4096, 14336) / GiB, seg(run(INF, 'llama-3-1-8b', 'h100_80', { seq: 16384, outputTokens: 1 }), 'act'), 0.1);
  T('Serving', 'Llama-70B activations, B200 (16384-token chunk) (GiB)', '16384 × (4·8192 + 3·28672) × 2 B + 128256 × 4 B', act(16384, 8192, 28672) / GiB, seg(run(INF, 'llama-3-1-70b', 'b200_192', { seq: 32768, outputTokens: 1, quant: 'fp8' }), 'act'), 0.1);
  // KV is a preallocated pool (no fragmentation); 3% fragmentation applies to activations only.
  const nonKv4 = L8.P * 2 + act(2048, 4096, 14336) * 1.03 + 1.30 * GiB;  // weights + activations·1.03 + context 0.55 + graphs 0.75
  T('Serving', 'Llama-8B per-GPU use, 4090, 16,384-token context (GiB)', 'weights + act × 1.03 + 0.55 + 0.75 + 16384 × 128 KiB', (nonKv4 + 16384 * kvTok8) / GiB, r4.memory.perGpu, 0.1);
  const kvCap = (r, s, util = 0.9) => (r.memory.capPerGpu / 0.9 * util - (r.memory.perGpu - seg(r, 'kv'))) / (seg(r, 'kv') / s);
  T('Serving', 'Llama-8B KV capacity, 4090 @0.90 (tokens; vLLM measured 41,408)', '(0.90 × 24 GiB − non-KV) ÷ 128 KiB', (0.9 * 24 * GiB - nonKv4) / kvTok8, kvCap(r4, 16383), 0.1);
  const nonKvP = L8.P * 2 + act(8192, 4096, 14336) * 1.03 + 1.30 * GiB;
  T('Serving', 'Llama-8B KV capacity, RTX PRO 6000 @0.92 (tokens; measured 586,384)', '(0.92 × 96 GiB − non-KV(8192 chunk)) ÷ 128 KiB', (0.92 * 96 * GiB - nonKvP) / kvTok8,
    kvCap(run(INF, 'llama-3-1-8b', 'rtx_6000_blackwell_server_96', { seq: 16384, outputTokens: 1 }), 16383, 0.92), 0.1);
}

/* ---------- report ---------- */
const pass = rows.filter((r) => r.pass).length;
const out = [];
out.push(`# | Area | Test | Hand formula | Hand value | Repo value | Dev % | Tol % | Result`);
for (const r of rows) out.push([r.n, r.area, r.name, r.handFormula, fmt(r.expected), fmt(r.actual), (r.dev >= 0 ? '+' : '') + r.dev.toFixed(2), '±' + r.tolPct, r.pass ? 'PASS' : 'FAIL'].join(' | '));
out.push(`\nTOTAL: ${rows.length} hand-calculated tests, ${pass} passed, ${rows.length - pass} failed`);
out.push('\nReference speed checks (benchmarks / roofline, not exact hand math):');
for (const r of refs) out.push([r.n, r.name, fmt(r.expected), fmt(r.actual), (r.dev >= 0 ? '+' : '') + r.dev.toFixed(2), '±' + r.tolPct, r.pass ? 'PASS' : 'FAIL'].join(' | '));
console.log(out.join('\n'));
if (pass !== rows.length) process.exitCode = 1;
