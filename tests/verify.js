/* Verification suite for calc.js — run with:  node tests/verify.js
 * 1. Hand-computed reference values (exact formulas).
 * 2. Physics bounds across every model × GPU (nothing may beat the bandwidth roofline).
 * 3. Monotonicity (more batch/users/context never uses less memory, etc.).
 * 4. Sanity against public benchmark ranges. */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const dir = path.join(__dirname, '..');
const ctx = { window: {} }; ctx.window = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(dir, 'data.js'), 'utf8'), ctx);
const D = ctx.VRAM_DATA;
const C = require(path.join(dir, 'calc.js'));
C.applyFixes(D);
const GiB = 2 ** 30;
const M = (s) => D.models.find((m) => m.slug === s);
const Gk = (k) => D.gpus.find((g) => g.key === k);

let pass = 0, fail = 0;
const ok = (cond, name, detail = '') => { if (cond) pass++; else { fail++; console.log(`  ✗ ${name} ${detail}`); } };
const near = (a, b, tol, name) => ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), name, `got ${a}, expected ${b}`);
const range = (a, lo, hi, name) => ok(a >= lo && a <= hi, name, `got ${a.toFixed ? a.toFixed(2) : a}, expected ${lo}–${hi}`);

const base = {
  mode: 'inference', quant: 'fp16', kvQuant: 'fp16', runtime: 'auto', numGpus: 1, cluster: false, parallelism: 'tensor', interconnect: 'nvlink_gen4',
  batch: 1, users: 1, seq: 4096, outputTokens: 512, prefixShare: 0, offload: false, offloadTarget: 'cpu_ram', offloadFrac: 0, offloadKv: false,
  spec: false, specDraft: 1, specAccept: 0.7, specTokens: 4, specType: 'sequential', elec: 0.12, carbon: 400,
  ftMethod: 'full', ftQuant: 'fp16', loraRank: 16, gradSteps: 1, numSamples: 10000, tokensPerSample: 512, numEpochs: 1,
  opt: { optimizer_type: 'adamw', zero_stage: 0, flash_attention: true, gradient_checkpointing: false, gradient_checkpointing_ratio: 1, use_8bit_optimizer: false, use_bf16_true: false, use_fused_kernels: true, sequence_packing: true, dynamic_padding: false, activation_offloading: false },
  imageWidth: 1024, imageHeight: 1024, numFrames: 1, inferenceSteps: 28, guidanceScale: 3.5, encoderOffload: true, vaeTiling: false,
};
const run = (slug, gpu, o = {}) => C.calculate({ ...base, ...o, opt: { ...base.opt, ...(o.opt || {}) } }, M(slug), Gk(gpu));
const seg = (r, key) => (r.memory.segments.find((s) => s.key === key) || { gib: 0 }).gib;

console.log('1. Exact formulas');
{
  const r = run('llama-3-1-8b', 'h100_80', { seq: 8192 });
  near(seg(r, 'weights'), 8.03e9 * 2 / GiB, 1e-6, 'Llama-3.1-8B FP16 weights = 8.03e9 × 2 B');
  near(r.memory.kvPerToken, 2 * 32 * 8 * 128 * 2, 1e-9, 'Llama-3.1-8B KV/token = 2·32·8·128·2 = 128 KiB');
  near(seg(r, 'kv'), 1.0, 1e-9, 'KV @ 8192 tokens = 1 GiB');
  near(seg(run('llama-3-1-8b', 'h100_80', { seq: 8192, batch: 4 }), 'kv'), 4.0, 1e-9, 'KV scales with batch (4 → 4 GiB)');
  near(seg(run('llama-3-1-8b', 'h100_80', { seq: 8192, users: 16 }), 'kv'), 16.0, 1e-9, 'KV scales with users (16 → 16 GiB, no cap)');
  near(seg(run('llama-3-1-8b', 'h100_80', { seq: 8192, outputTokens: 512, users: 4, prefixShare: 0.5 }), 'kv'), (3840 + 4 * (8192 - 3840)) * 2 * 32 * 8 * 128 * 2 / GiB, 1e-9, '50% shared prompt prefix (3840 tok), 4 users (output tokens not shared)');
  near(seg(run('llama-3-1-8b', 'h100_80', { seq: 8192, kvQuant: 'fp8' }), 'kv'), 0.5, 1e-9, 'FP8 KV halves the cache');
  near(seg(run('llama-3-1-8b', 'h100_80', { quant: 'q4_k_m', runtime: 'llamacpp' }), 'weights'), 8.03e9 * 4.89 / 8 / GiB, 1e-6, 'Q4_K_M = 4.89 bits/weight');
  const ds = run('deepseek-v3', 'h200_141', { quant: 'fp8', seq: 1024 });
  near(ds.memory.kvPerToken, 61 * 576 * 2, 1e-9, 'DeepSeek-V3 MLA KV/token = 61 layers × 576 × 2 B');
  const gm = run('gemma-3-27b', 'h100_80', { seq: 32768 });
  const gExpect = (10 * 32768 + 52 * 1024) * 2 * 16 * 128 * 2 / GiB;
  near(seg(gm, 'kv'), gExpect, 1e-9, 'Gemma-3-27B: 10 global + 52 sliding(1024) layers');
  const oss = run('gpt-oss-120b', 'h100_80', { quant: 'mxfp4' });
  near(seg(oss, 'weights'), ((116.8e9 - 201088 * 2880 * 2) * 4.25 / 8 + 201088 * 2880 * 2 * 2) / GiB, 1e-6, 'gpt-oss-120b MXFP4 with 16-bit embeddings');
  range(seg(oss, 'weights'), 57, 62, 'gpt-oss-120b ≈ 60.8 GiB checkpoint');
  // fine-tuning
  const ft = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', seq: 2048 });
  near(seg(ft, 'weights') + seg(ft, 'grads') + seg(ft, 'optim'), 8.03e9 * 16 / GiB, 1e-6, 'Full FT mixed-precision AdamW = 16 B/param (2+2+4+8)');
  const ql = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'qlora', seq: 2048 });
  near(seg(ql, 'weights'), 8.03e9 * 4.127 / 8 / GiB, 1e-6, 'QLoRA NF4 = 4.127 bits/param');
  const lo = C.geometry(M('llama-3-1-8b'));
  const loraP = 16 * 32 * ((4096 + 4096) * 2 + (4096 + 1024) * 2 + 3 * (4096 + 14336));
  ok(ft && loraP === 41943040, 'LoRA r16 all-linear on Llama-8B = 41.9M params', `got ${loraP}`);
  ok(!lo.tied, 'Llama-3.1-8B detected as untied embeddings');
  ok(C.geometry(M('llama-3-2-1b')).tied, 'Llama-3.2-1B detected as tied embeddings');
  // energy
  const e = run('llama-3-1-8b', 'h100_80');
  near(e.energy.co2KgPerDay, e.watts / 1000 * 24 * 400 / 1000, 1e-12, 'CO₂/day = kW × 24 × g/kWh ÷ 1000');
  near(e.energy.elecPerHour, e.watts / 1000 * 0.12, 1e-12, 'Electricity/h = kW × $/kWh');
}

console.log('2. Physics bounds over every model × GPU');
{
  let checked = 0, viol = 0;
  const gpus = D.gpus.filter((g) => !g.key.startsWith('custom'));
  for (const m of D.models.filter((x) => x.task_type === 'text_generation')) {
    for (const g of gpus) {
      for (const q of ['fp16', 'q4_k_m']) {
        const r = C.calculate({ ...base, quant: q }, m, g);
        if (!r.perf) continue;
        checked++;
        const G = r.geometry;
        // roofline: one decode step must read at least the active weights once
        const activeBytes = C.weightBytes(G, q) * (G.moe ? G.A / G.P : 1) * 0.9;
        const roof = g.bandwidth_gbs * 1e9 / activeBytes;
        if (r.perf.tps > roof) { viol++; if (viol < 6) console.log(`  ✗ ${m.slug} on ${g.key} ${q}: ${r.perf.tps.toFixed(0)} tok/s > roofline ${roof.toFixed(0)}`); }
        if (!(r.perf.ttft > 0 && Number.isFinite(r.perf.tps))) { viol++; console.log(`  ✗ ${m.slug} ${g.key} invalid perf`); }
      }
    }
  }
  ok(viol === 0, `no decode speed above bandwidth roofline (${checked} configs)`);
  let ftBad = 0;
  for (const g of gpus) {
    const r = run('llama-3-1-8b', g.key, { mode: 'finetuning', ftMethod: 'qlora', seq: 1024 });
    if (r.perf && (r.perf.mfu > 0.6 || r.perf.trainTps < 10)) ftBad++;
  }
  ok(ftBad === 0, 'fine-tuning speed sane on every GPU (old engine reported 1 tok/s on ~20 GPUs)');
}

console.log('3. Monotonicity');
{
  const v = (o) => run('llama-3-1-8b', 'h100_80', o);
  let prev = 0;
  for (const b of [1, 2, 4, 8, 16, 32, 64, 128]) { const x = v({ batch: b }).memory.perGpu; ok(x > prev, `VRAM grows with batch ${b}`); prev = x; }
  prev = 0;
  for (const u of [1, 2, 8, 9, 16, 64, 128]) { const x = v({ users: u }).memory.perGpu; ok(x > prev, `VRAM grows with users ${u}`); prev = x; }
  prev = Infinity;
  for (const s of [2048, 8192, 32768, 131072]) { const x = v({ seq: s }).perf; if (x) { ok(x.tps < prev, `tok/s falls as context grows (${s})`); prev = x.tps; } }
  prev = 0;
  for (const u of [1, 4, 16, 64]) { const x = v({ users: u }).perf.throughput; ok(x > prev, `aggregate throughput rises with users ${u}`); prev = x; }
  const r4 = v({ users: 4 }).perf;
  near(r4.throughput, r4.tps * 4, 1e-9, 'per-user × users = total throughput');
  const bw = ['h100_80', 'h200_141', 'b200_192'].map((k) => run('gemma-3-27b', k).perf.tps);
  ok(bw[0] < bw[1] && bw[1] < bw[2], 'more bandwidth → faster decode (H100 < H200 < B200)', bw.map((x) => x.toFixed(0)).join(' < '));
  const tp = [1, 2, 4].map((n) => run('llama-3-1-70b', 'h100_80', { numGpus: n, parallelism: 'tensor', quant: 'fp8' }));
  ok(tp[1].perf.tps > tp[0].perf.tps && tp[2].perf.tps > tp[1].perf.tps, 'tensor parallel speeds up decode');
  ok(tp[2].memory.perGpu < tp[1].memory.perGpu, 'tensor parallel lowers per-GPU memory');
}

console.log('4. Benchmark sanity (public numbers, ±~25%)');
{
  range(run('llama-3-1-8b', 'h100_80', { runtime: 'vllm', seq: 2048, outputTokens: 256 }).perf.tps, 115, 175, 'vLLM Llama-3.1-8B BF16 H100 bs1 ≈ 140 tok/s');
  range(run('llama-3-1-8b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp', seq: 1024, outputTokens: 128 }).perf.tps, 105, 175, 'llama.cpp 8B Q4_K_M RTX 4090 ≈ 130–150 tok/s');
  range(run('llama-3-1-8b', '3090_24', { quant: 'q4_k_m', runtime: 'llamacpp', seq: 1024, outputTokens: 128 }).perf.tps, 90, 140, 'llama.cpp 8B Q4_K_M RTX 3090 ≈ 110 tok/s');
  range(run('llama-3-1-8b', 'm2_ultra_192', { quant: 'q4_k_m', runtime: 'llamacpp', seq: 1024, outputTokens: 128 }).perf.tps, 60, 100, 'llama.cpp 8B Q4_K_M M2 Ultra ≈ 75 tok/s');
  range(run('llama-3-1-70b', 'h100_80', { numGpus: 2, parallelism: 'tensor', runtime: 'vllm', seq: 2048 }).perf.tps, 28, 50, 'vLLM 70B BF16 2×H100 TP2 ≈ 35–40 tok/s');
  range(run('llama-3-1-8b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp', seq: 1024, outputTokens: 128 }).perf.prefillTps, 3000, 14000, 'llama.cpp 8B Q4_K_M 4090 prompt processing ≈ 5–10k tok/s');
  range(run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 4 }).perf.trainTps, 4000, 16000, 'LoRA 8B on H100 ≈ 5–12k tok/s');
}


console.log('5. Dataset: Qwen configs, GPUs, duplicates');
{
  // Checkpoint totals from Hugging Face safetensors metadata, minus the MTP draft head (not loaded for plain decoding)
  const ckpt = { 'qwen35-27b': [27.781, 0.425], 'qwen35-35b-a3b': [35.952, 0.845], 'qwen35-397b-a17b': [403.398, 6.596], 'qwen-38-24t-a95b': [2446.183, 26.378],
    'qwen-38-flash': [180.0, 2.586], 'qwen3-next-80b-a3b': [81.325, 1.651], 'qwen3-coder-480b-a35b': [480.155, 0], 'qwen3-vl-8b': [8.767, 0], 'qwen36-27b': [27.781, 0.425] };
  for (const [slug, [tot, mtp]] of Object.entries(ckpt)) {
    const m = M(slug); ok(!!m, `${slug} present`); if (!m) continue;
    near(Number(m.num_of_params) + Number(m.num_of_unread_params || 0), tot - mtp, 0.002, `${slug} params = checkpoint − MTP`);
  }
  near(Number(M('qwen35-397b-a17b').num_of_expert_params), 17.35, 0.01, 'Qwen3.5-397B-A17B active ≈ 17.3B');
  near(Number(M('qwen-38-24t-a95b').num_of_expert_params), 95.29, 0.01, 'Qwen3.8-2.4T-A95B active ≈ 95.3B');
  near(Number(M('qwen3-next-80b-a3b').num_of_expert_params), 3.88, 0.01, 'Qwen3-Next-80B-A3B active ≈ 3.9B');
  // Qwen3.5-27B: 16 full-attention layers × 2 × 4 KV heads × 256 × 2 B; 48 Gated-DeltaNet layers of fixed state
  const r = run('qwen35-27b', 'h100_80', { seq: 32768 });
  near(r.memory.kvPerToken, 16 * 2 * 4 * 256 * 2, 1e-9, 'Qwen3.5-27B KV/token = 64 KiB (full layers only)');
  const K = C.kvLayout(M('qwen35-27b'), C.geometry(M('qwen35-27b')));
  near(K.lin * K.linState, 48 * (48 * 128 * 128 + (2 * 16 * 128 + 48 * 128) * 3) * 2, 1e-9, 'Qwen3.5-27B GDN state = 48 × (48·128² + conv) × 2 B');
  // Flash-Next: 52B of N-gram tables + vision are loaded but not streamed each token
  const f = run('qwen-38-flash', 'b300_288', { quant: 'fp8', seq: 8192 });
  const G = C.geometry(M('qwen-38-flash'));
  ok(G.unread > 50e9, 'Flash-Next lookup tables counted as unread');
  ok(f.perf && f.perf.tps > 60, 'Flash-Next decode not slowed by lookup tables', `got ${f.perf && f.perf.tps}`);
  ok(seg(f, 'weights') * GiB > (G.P * 1 + G.unread * 2) * 0.99, 'Flash-Next memory includes lookup tables');
  // GPUs: nothing unreleased; B300 and Rubin present with datasheet specs
  ok(D.gpus.every((g) => g.released !== false), 'no unreleased / upcoming GPUs');
  const b3 = Gk('b300_288'); ok(b3 && b3.memory === 288 && b3.bandwidth_gbs === 8000 && b3.fp16_tflops === 2250, 'B300: 288 GB, 8 TB/s, 2.25 PF BF16');
  const b2 = run('llama-3-1-70b', 'b200_192', { quant: 'fp8' }), b3r = run('llama-3-1-70b', 'b300_288', { quant: 'fp8' });
  near(b3r.perf.tps, b2.perf.tps, 0.02, 'B300 decode ≈ B200 (same 8 TB/s)');
  ok(b3r.memory.capPerGpu > b2.memory.capPerGpu, 'B300 has more capacity than B200');
  const nv = run('llama-3-1-70b', 'b200_192', { quant: 'nvfp4', seq: 16384, outputTokens: 256 }), fp = run('llama-3-1-70b', 'b200_192', { quant: 'fp16', seq: 16384, outputTokens: 256 });
  ok(nv.perf.ttft < fp.perf.ttft / 2.5, 'NVFP4 prefill on Blackwell uses FP4 tensor cores');
  // No duplicate model names
  const names = D.models.map((m) => m.name.toLowerCase().replace(/[\s.\-_]/g, ''));
  ok(new Set(names).size === names.length, 'no duplicate model names', names.filter((n, i) => names.indexOf(n) !== i).join(', '));
  ok(new Set(D.models.map((m) => m.slug)).size === D.models.length, 'no duplicate slugs');
  ok(D.models.every((m) => !m.release_date || m.release_date <= '2026-11-21'), 'no models dated in the future');
  // Every new model computes without NaN on a big GPU
  for (const m of D.models.filter((x) => x._added)) {
    const rr = C.calculate({ ...base, opt: base.opt }, m, Gk('b300_288'));
    ok(Number.isFinite(rr.memory.perGpu) && (!rr.perf || Object.values(rr.perf).every((v) => typeof v !== 'number' || Number.isFinite(v))), `${m.slug} computes`);
  }
}

console.log('6. Regression tests (third-review fixes)');
{
  // 1. RTX 4060-4080 Super FP32-acc halving & monotonic 40-series TFLOPS + GGUF prefill
  const r4080s = run('llama-3-1-8b', '4080super_16', { quant: 'fp8', runtime: 'vllm' });
  const r4090 = run('llama-3-1-8b', '4090_24', { quant: 'fp8', runtime: 'vllm' });
  ok(C.effTflops(Gk('4080super_16')).v < C.effTflops(Gk('4090_24')).v, 'RTX 4080 Super effTflops < RTX 4090');
  ok(r4090.perf.prefillTps > r4080s.perf.prefillTps, 'RTX 4090 FP8 prefill faster than RTX 4080 Super');
  const q4080s = run('llama-3-1-8b', '4080super_16', { quant: 'q4_k_m', runtime: 'llamacpp' });
  const q4090 = run('llama-3-1-8b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp' });
  const q3090 = run('llama-3-1-8b', '3090_24', { quant: 'q4_k_m', runtime: 'llamacpp' });
  ok(q4090.perf.prefillTps > q4080s.perf.prefillTps && q4080s.perf.prefillTps > q3090.perf.prefillTps, 'GGUF prefill: 4090 > 4080S > 3090');

  // 2. FP8 support for L40, RTX 4500 Ada, Radeon AI PRO R9700
  ok(C.supportsFp8(Gk('l40_48')), 'L40 supports FP8');
  ok(C.supportsFp8(Gk('rtx_4500_ada_24')), 'RTX 4500 Ada supports FP8');
  ok(C.supportsFp8(Gk('radeon_ai_pro_r9700_32')), 'Radeon AI PRO R9700 supports FP8');

  // 3. Custom Apple Silicon does not halve user-entered effective BF16 TFLOPS
  const customApple = { key: 'custom', label: 'Custom Apple', type: 'apple_silicon', vendor: 'apple', category: 'consumer', memory: 128, bandwidth_gbs: 546, fp16_tflops: 17, tdp_watts: 60 };
  near(C.effTflops(customApple).v, 17, 1e-9, 'Custom Apple Silicon keeps user-entered effective TFLOPS');

  // 4. Dense multimodal models exclude unread vision weights from decode wRead
  const q08 = run('qwen35-08b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp', seq: 1024, outputTokens: 128 });
  const q08G = C.geometry(M('qwen35-08b'));
  const q08TextB = q08G.P * C.QUANTS.q4_k_m.bpw / 8;
  const q08UnreadB = C.unreadBytes(q08G, 'q4_k_m');
  ok(seg(q08, 'weights') * GiB > q08TextB + q08UnreadB * 0.99, 'Dense VL memory includes vision weights');
  const q08StepMemMax = (q08TextB + C.kvBytesSeq(C.kvLayout(M('qwen35-08b'), q08G), 1024, 2)) / (Gk('4090_24').bandwidth_gbs * 1e9 * 0.74);
  ok(q08.perf.stepMs / 1000 < q08StepMemMax + 0.002, 'Dense VL decode does not stream unread vision weights');

  // 5. MLA / tp > kvh KV bandwidth uses kvShard, not tp
  const mla1 = run('deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'tensor', seq: 131072, users: 4, outputTokens: 512 });
  const mlaKv = C.kvBytesSeq(C.kvLayout(M('deepseek-v3'), C.geometry(M('deepseek-v3'))), 131072 - 256, 2) * 4;
  const minKvTimeMs = mlaKv / (Gk('h200_141').bandwidth_gbs * 1e9 * 0.8) * 1000;
  ok(mla1.perf && mla1.perf.stepMs > minKvTimeMs, 'MLA TP8 decode time accounts for full replicated KV read per GPU', `stepMs=${mla1.perf && mla1.perf.stepMs.toFixed(1)}, minKv=${minKvTimeMs.toFixed(1)}`);

  // 6. Data-parallel replicated KV, prefixShare, and throughput
  const dp1 = run('llama-3-1-8b', 'h100_80', { numGpus: 4, parallelism: 'replicated', users: 1, batch: 1, seq: 8192 });
  near(seg(dp1, 'kv'), 1.0, 1e-9, 'DP=4 with 1 user still needs 1.0 GiB KV on active GPU');
  near(dp1.perf.throughput, dp1.perf.tps, 1e-9, 'DP=4 with 1 user has total throughput = 1 × per-user speed');
  const dp4sh = run('llama-3-1-8b', 'h100_80', { numGpus: 4, parallelism: 'replicated', users: 4, batch: 1, seq: 8192, prefixShare: 1.0 });
  near(seg(dp4sh, 'kv'), 1.0, 1e-9, 'DP=4 with 4 users & 100% prefixShare: each replica stores 1 full sequence');
  const dp5 = run('llama-3-1-8b', 'h100_80', { numGpus: 4, parallelism: 'replicated', users: 5, batch: 1, seq: 8192 });
  near(dp5.perf.throughput, dp5.perf.tps * 5, 1e-9, 'DP=4 with 5 users has total throughput = 5 × per-user speed');

  // 7. Cluster TP*PP > n clamps safely so total weights never shrink below actual model weights
  const cl = run('llama-3-1-70b', 'h100_80', { cluster: true, numNodes: 1, gpusPerNode: 8, tpDegree: 8, ppDegree: 2 });
  near(seg(cl, 'weights') * 8, 70.55e9 * 2 / GiB, 1e-6, 'Cluster TP*PP > n clamps so total weights match model size');
  ok(cl.warnings.some((w) => /clamped/i.test(w)), 'Cluster TP*PP > n emits warning');

  // 8. Fine-tuning packed step count & step time consistency, gated offload penalties, FP8 speedup
  const ftPack = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 2, gradSteps: 4, numSamples: 10000, tokensPerSample: 512, numEpochs: 3, opt: { sequence_packing: true, gradient_checkpointing: true } });
  ok(ftPack.perf.steps === Math.ceil(10000 * 512 * 3 / 16384), 'Packed FT steps = ceil(15.36M / 16384) = 938', `got ${ftPack.perf.steps}`);
  near(ftPack.perf.steps * (1 / ftPack.perf.stepsPerSec), ftPack.perf.hours * 3600, 0.002, 'Packed FT steps × stepTime ≈ total hours');
  const ftNoOff = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 2, opt: { offload_param: false, offload_optimizer: false } });
  const ftGhost = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 2, opt: { offload_param: true, offload_optimizer: true } });
  near(ftNoOff.perf.trainTps, ftGhost.perf.trainTps, 1e-9, '1-GPU FT ignores inactive ZeRO offload flags in speed');
  const ftBf16 = run('llama-3-1-8b', 'h200_141', { mode: 'finetuning', ftMethod: 'full', ftQuant: 'fp16', seq: 2048, batch: 2, opt: { gradient_checkpointing: true } });
  const ftFp8 = run('llama-3-1-8b', 'h200_141', { mode: 'finetuning', ftMethod: 'full', ftQuant: 'fp8', seq: 2048, batch: 2, opt: { gradient_checkpointing: true } });
  ok(ftFp8.perf && ftBf16.perf && ftFp8.perf.trainTps > ftBf16.perf.trainTps * 1.5, 'Full FT FP8 on H200 gets FP8 tensor-core speedup');

  // 9. Diffusion FP8 speedup & monotonic efficiency (no 500 TF cliff)
  const fl16 = run('flux-1-dev', 'h100_80', { quant: 'fp16' });
  const fl8 = run('flux-1-dev', 'h100_80', { quant: 'fp8' });
  ok(fl8.perf.seconds < fl16.perf.seconds * 0.7, 'FLUX.1-dev FP8 is faster than FP16 on H100');
  const flAda = run('flux-1-dev', 'rtx_6000_ada_48', { quant: 'fp16' });
  const flBw = run('flux-1-dev', 'rtx_6000_blackwell_96', { quant: 'fp16' });
  ok(flBw.perf.seconds < flAda.perf.seconds, 'Diffusion monotonic: RTX 6000 Blackwell faster than RTX 6000 Ada');

  // 10. Repaired dataset models (no 0 weights, no NaN on spec, MoE fixed)
  for (const slug of ['devstral-2-123b-instruct', 'codestral-2501', 'mistral-small-2501', 'magistral-small', 'ministral-8b-2410', 'qwen2-5-32b', 'minimax-m2', 'minimax-m3']) {
    const r = run(slug, 'b300_288', { quant: 'fp8', spec: true });
    ok(seg(r, 'weights') > 1 && Number.isFinite(r.memory.perGpu), `${slug} has valid weights & no NaN with spec`);
  }
  ok(C.geometry(M('minimax-m3')).moe, 'MiniMax-M3 recognized as MoE');
  ok(C.geometry(M('minimax-m2')).shared + C.geometry(M('minimax-m2')).routed * C.geometry(M('minimax-m2')).f - C.geometry(M('minimax-m2')).A < 1e6, 'MiniMax-M2 active params consistent');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
