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
  const b1 = run('llama-3-1-8b', 'h100_80', { seq: 8192, outputTokens: 512, batch: 1 });
  const b4 = run('llama-3-1-8b', 'h100_80', { seq: 8192, outputTokens: 512, batch: 4 });
  near(seg(b4, 'kv'), (7680 + 4 * 512) / 8192, 1e-9, 'Batch n=4 shares prompt KV (7680 + 4×512 = 9728 tok)');
  near(b4.perf.prefillTps, b1.perf.prefillTps, 1e-9, 'Batch n=4 prefills the shared prompt once');
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
  const ossG = C.geometry(M('gpt-oss-120b'));
  const ossFull16 = 201088 * 2880 * 2 + ossG.L * ossG.attnParams;
  near(seg(oss, 'weights'), ((116.8e9 - ossFull16) * 4.25 / 8 + ossFull16 * 2) / GiB, 1e-6, 'gpt-oss-120b MXFP4 with 16-bit embeddings + attention');
  range(seg(oss, 'weights'), 59.5, 62, 'gpt-oss-120b ≈ 60.7 GiB checkpoint');
  // fine-tuning
  const ft = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', seq: 2048 });
  near(seg(ft, 'weights') + seg(ft, 'grads') + seg(ft, 'optim'), 8.03e9 * 16 / GiB, 1e-6, 'Full FT mixed-precision AdamW = 16 B/param (2+2+4+8)');
  const ql = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'qlora', seq: 2048 });
  const lo = C.geometry(M('llama-3-1-8b'));
  near(seg(ql, 'weights'), ((8.03e9 - lo.embedParams) * 4.127 / 8 + lo.embedParams * 2) / GiB, 1e-6, 'QLoRA NF4 (4.127 bpw) with BF16 embeddings');
  const loraP = Math.round(seg(ql, 'lora') * GiB / 4);
  ok(loraP === 41943040, 'calc.js LoRA r16 all-linear on Llama-8B = 41.94M params', `got ${loraP}`);
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
        // roofline: one decode step must read at least the active (streamed) weights once
        const activeBytes = (C.weightBytes(G, q) - C.unreadBytes(G, q)) * (G.moe ? G.A / G.P : 1) * 0.9;
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
  // Sweep across every architecture family: dense GQA, MoE, MLA, sliding window, and hybrid linear attention.
  for (const [slug, gpuKey, extra] of [
    ['llama-3-1-8b', 'h100_80', {}],
    ['qwen3-30b-a3b', 'h100_80', { quant: 'fp8' }],
    ['deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'tensor' }],
    ['gemma-3-27b', 'h100_80', { quant: 'fp8' }],
    ['qwen35-27b', 'h100_80', { quant: 'fp8' }],
  ]) {
    const v = (o) => run(slug, gpuKey, { ...extra, ...o });
    let prev = 0;
    for (const b of [1, 2, 4, 8, 16, 32]) { const x = v({ batch: b }).memory.perGpu; ok(x > prev, `${slug}: VRAM grows with batch ${b}`); prev = x; }
    prev = 0;
    for (const u of [1, 2, 8, 16, 64]) { const x = v({ users: u }).memory.perGpu; ok(x > prev, `${slug}: VRAM grows with users ${u}`); prev = x; }
    prev = Infinity;
    for (const s of [2048, 8192, 32768, 131072]) { const x = v({ seq: s }).perf; if (x) { ok(x.tps < prev, `${slug}: tok/s falls as context grows (${s})`); prev = x.tps; } }
    prev = 0;
    for (const u of [1, 4, 16, 64]) { const x = v({ users: u }).perf; if (x) { ok(x.throughput > prev, `${slug}: aggregate throughput rises with users ${u}`); prev = x.throughput; } }
  }
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
  // Every model in the curated dataset has complete geometry, sane params, and computes without NaN
  for (const m of D.models) {
    const rr = C.calculate({ ...base, opt: base.opt }, m, Gk('b300_288'));
    ok(Number.isFinite(rr.memory.perGpu) && rr.memory.perGpu > 0 && (!rr.perf || Object.values(rr.perf).every((v) => typeof v !== 'number' || Number.isFinite(v))), `${m.slug} computes`);
    if (m.task_type === 'text_generation') {
      const G = C.geometry(m);
      ok(m.hidden_dim_size > 0 && m.num_of_layers > 0 && m.num_attention_heads > 0 && m.vocab_size > 0 && m.context_length > 0, `${m.slug} has complete core config`);
      if (G.moe) {
        ok(G.A > 0 && G.A < G.P, `${m.slug} MoE active < total`, `A=${G.A}, P=${G.P}`);
        ok(G.shared > 0, `${m.slug} MoE shared > 0 (catches wrong expert count)`, `shared=${G.shared}`);
        ok(m.num_of_experts > m.num_of_active_experts && m.num_of_active_experts > 0, `${m.slug} MoE E > k > 0`);
        near(G.f, m.num_of_active_experts / m.num_of_experts, 1e-9, `${m.slug} MoE f === k/E (no fallback clamp)`);
        ok(m.moe_ffn_intermediate_size > 0, `${m.slug} MoE has moe_ffn_intermediate_size`);
      }
      ok(m.num_of_auxiliary_params == null, `${m.slug} text model has no unsplit num_of_auxiliary_params`);
      ok(!('num_dense_layers' in m), `${m.slug} has no unused num_dense_layers field`);
    } else {
      // Non-UNet diffusion models must have hidden_dim_size = heads * head_dim (not joint_attention_dim)
      const isUnet = /stable-diffusion-(1|2)|sdxl|sd-2|kandinsky|svd/.test(m.slug);
      if (!isUnet) ok(m.hidden_dim_size === m.num_attention_heads * m.attention_head_dim, `${m.slug} diffusion h === heads*hd`, `h=${m.hidden_dim_size}, heads*hd=${m.num_attention_heads * m.attention_head_dim}`);
    }
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

  // 4. Dense multimodal models exclude unread vision weights from decode wRead (tight comparison against a no-unread clone)
  const q08m = M('qwen35-08b');
  const q08 = run('qwen35-08b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp', seq: 1024, outputTokens: 128 });
  const q08NoVis = C.calculate({ ...base, quant: 'q4_k_m', runtime: 'llamacpp', seq: 1024, outputTokens: 128 }, { ...q08m, num_of_unread_params: 0 }, Gk('4090_24'));
  const q08G = C.geometry(q08m);
  const q08UnreadB = C.unreadBytes(q08G, 'q4_k_m');
  ok(seg(q08, 'weights') * GiB > seg(q08NoVis, 'weights') * GiB + q08UnreadB * 0.99, 'Dense VL memory includes vision weights');
  near(q08.perf.tps, q08NoVis.perf.tps, 1e-9, 'Dense VL decode speed is identical with or without unread vision weights');

  // 5. MLA / tp > kvh KV bandwidth uses kvShard, not tp
  const mla1 = run('deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'tensor', seq: 131072, users: 4, outputTokens: 512 });
  const mlaKv = C.kvBytesSeq(C.kvLayout(M('deepseek-v3'), C.geometry(M('deepseek-v3'))), 131072 - 256, 2) * 4;
  const minKvTimeMs = mlaKv / (Gk('h200_141').bandwidth_gbs * 1e9 * 0.8) * 1000;
  ok(mla1.perf && mla1.perf.stepMs > minKvTimeMs, 'MLA TP8 decode time accounts for full replicated KV read per GPU', `stepMs=${mla1.perf && mla1.perf.stepMs.toFixed(1)}, minKv=${minKvTimeMs.toFixed(1)}`);

  // 6. Data-parallel replicated KV, prefixShare, and throughput (independent of TP/single-GPU baseline)
  const dp1 = run('llama-3-1-8b', 'h100_80', { numGpus: 4, parallelism: 'replicated', users: 1, batch: 1, seq: 8192 });
  const solo8 = run('llama-3-1-8b', 'h100_80', { numGpus: 1, users: 1, batch: 1, seq: 8192 });
  near(seg(dp1, 'kv'), 1.0, 1e-9, 'DP=4 with 1 user still needs 1.0 GiB KV on active GPU');
  near(dp1.perf.throughput, solo8.perf.tps, 1e-6, 'DP=4 with 1 user matches 1-GPU solo throughput (idle replicas add 0)');
  const dp4sh = run('llama-3-1-8b', 'h100_80', { numGpus: 4, parallelism: 'replicated', users: 4, batch: 1, seq: 8192, prefixShare: 1.0 });
  near(seg(dp4sh, 'kv'), 1.0, 1e-9, 'DP=4 with 4 users & 100% prefixShare: each replica stores 1 full sequence');
  const dp4 = run('llama-3-1-8b', 'h100_80', { numGpus: 4, parallelism: 'replicated', users: 4, batch: 1, seq: 8192 });
  near(dp4.perf.throughput, solo8.perf.tps * 4, 1e-6, 'DP=4 with 4 users scales 4× over 1-GPU solo speed');
  const dp5 = run('llama-3-1-8b', 'h100_80', { numGpus: 4, parallelism: 'replicated', users: 5, batch: 1, seq: 8192 });
  ok(dp5.perf.throughput > dp4.perf.throughput && dp5.perf.tps < dp4.perf.tps, 'DP=4 with 5 users has higher total throughput than 4 users, lower per-user speed');

  // 7. Cluster TP*PP > n clamps safely so total weights never shrink below actual model weights
  const cl = run('llama-3-1-70b', 'h100_80', { cluster: true, numNodes: 1, gpusPerNode: 8, tpDegree: 8, ppDegree: 2 });
  near(seg(cl, 'weights') * 8, 70.55e9 * 2 / GiB, 1e-6, 'Cluster TP*PP > n clamps so total weights match model size');
  ok(cl.warnings.some((w) => /clamped/i.test(w)), 'Cluster TP*PP > n emits warning');

  // 8. Fine-tuning packed step count & step time consistency, 1-GPU & multi-GPU offload gating, FP8 speedup
  const ftPack = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 2, gradSteps: 4, numSamples: 10000, tokensPerSample: 512, numEpochs: 3, opt: { sequence_packing: true, gradient_checkpointing: true } });
  ok(ftPack.perf.steps === Math.ceil(10000 * 512 * 3 / 16384), 'Packed FT steps = ceil(15.36M / 16384) = 938', `got ${ftPack.perf.steps}`);
  near(ftPack.perf.steps * (1 / ftPack.perf.stepsPerSec), ftPack.perf.hours * 3600, 0.002, 'Packed FT steps × stepTime ≈ total hours');
  const ft1NoOff = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 2, opt: { offload_param: false, offload_optimizer: false } });
  const ft1Off = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 2, opt: { offload_param: true, offload_optimizer: true } });
  ok(ft1Off.memory.perGpu < ft1NoOff.memory.perGpu && ft1Off.perf.trainTps < ft1NoOff.perf.trainTps, '1-GPU FT CPU offload reduces VRAM and applies offload speed penalty');
  const ftDdpNoOff = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', numGpus: 2, ftMethod: 'lora', seq: 2048, batch: 2, opt: { zero_stage: 0, offload_param: false, offload_optimizer: false } });
  const ftDdpGhost = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', numGpus: 2, ftMethod: 'lora', seq: 2048, batch: 2, opt: { zero_stage: 0, offload_param: true, offload_optimizer: true } });
  near(ftDdpNoOff.perf.trainTps, ftDdpGhost.perf.trainTps, 1e-9, 'Multi-GPU plain DDP (ZeRO-0) ignores inactive ZeRO offload flags');
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
  ok(C.geometry(M('minimax-m2')).shared > 1e9, 'MiniMax-M2 shared params > 1B');

  // 11. Fourth-review dataset & GPU datasheet fixes
  const klG = C.geometry(M('kimi-linear-48b-a3b-instruct'));
  ok(M('kimi-linear-48b-a3b-instruct').num_of_experts === 256 && klG.shared > 1e9, 'Kimi-Linear 48B has 256 experts & shared > 1B');
  ok(M('phi-3-mini').link_weights.includes('128k'), 'Phi-3-mini link matches 128K context');
  ok(M('mistral-large-3').moe_ffn_intermediate_size === 4096 && M('mistral-large-3').attention_head_dim === 160, 'Mistral Large 3 has moe_ffn=4096 & hd=160');
  near(Number(M('gemma-3-4b').num_of_params) + Number(M('gemma-3-4b').num_of_unread_params), 4.30, 0.01, 'Gemma 3 4B splits vision encoder into unread params');
  const g4e2K = C.kvLayout(M('gemma-4-e2b'), C.geometry(M('gemma-4-e2b')));
  ok(g4e2K.full + g4e2K.swa === 15 && g4e2K.fullAttn + g4e2K.swaAttn === 35, 'Gemma 4 E2B: 15 KV source layers, 35 compute layers');
  const nemK = C.kvLayout(M('nvidia-nemotron-3-nano-30b-a3b'), C.geometry(M('nvidia-nemotron-3-nano-30b-a3b')));
  ok(nemK.full === 6 && nemK.lin === 23, 'Nemotron-3-Nano: 6 GQA layers + 23 Mamba-2 state layers');
  ok(M('flux-2-dev').hidden_dim_size === 6144 && M('stable-diffusion-3-5-large').hidden_dim_size === 2432 && M('stable-diffusion-3-5-medium').hidden_dim_size === 1536, 'Diffusion hidden dims match heads × head_dim');
  ok(Gk('h100nvl_94').tdp_watts === 400 && Gk('h200nvl_141').fp16_tflops === 835.5 && Gk('mi350x_288').fp16_tflops === 2309.6 && Gk('gb10_128').fp16_tflops === 125, 'Datacenter/GB10 GPU datasheet specs verified');
}

console.log('7. Specialist audit regression tests');
{
  // 1. DeepSeek V4 / V4.1 KV compression & Engram unread tables
  // Model card: "FP4 main KV caching (E2M1, one E4M3 scale per 16 channels) … 890 bytes per token".
  const v41 = run('deepseek-v4-1-flash', 'b300_288', { quant: 'fp8', kvQuant: 'nvfp4', seq: 131072 });
  near(v41.memory.kvPerToken, 890, 1e-9, 'DeepSeek-V4.1-Flash FP4 KV = 890 B/token (model card)');
  const v41f8 = run('deepseek-v4-1-flash', 'b300_288', { quant: 'fp8', kvQuant: 'fp8', seq: 131072 });
  near(v41f8.memory.kvPerToken, (3 * 576 / 2 + 576) * 1 + (3 * 32 / 2 + 32), 1e-9, 'DeepSeek-V4.1-Flash FP8 KV = 4 source layers (3 ÷2 + 1 full)');
  ok(C.geometry(M('deepseek-v4-1-flash')).unread === 196e9, 'DeepSeek-V4.1-Flash includes 196B unread Engram tables');
  const v4f = run('deepseek-v4-flash', 'b300_288', { quant: 'fp8', kvQuant: 'fp8', seq: 131072 });
  // config.json compress_ratios: 21 CSA layers ÷4 + 20 HCA layers ÷128 (+2 SWA-only); 576-element latent, 132 B indexer keys.
  near(v4f.memory.kvPerToken, (21 / 4 + 20 / 128) * 576 * 1 + (21 / 4) * 132, 1e-9, 'DeepSeek-V4-Flash KV: 21 CSA layers ÷4 (+ indexer keys) + 20 HCA layers ÷128');
  ok(v4f.memory.kvPerToken < 0.1 * run('deepseek-v32', 'b300_288', { quant: 'fp8', kvQuant: 'fp8', seq: 131072 }).memory.kvPerToken, 'DeepSeek-V4-Flash KV < 10% of V3.2 (card: V4-Pro 10% at 1M)');

  // 2. Gemma 4 heterogeneous global vs sliding KV
  const g431 = run('gemma-4-31b', 'h100_80', { quant: 'fp8', kvQuant: 'fp8', seq: 131072 });
  near(seg(g431, 'kv'), (10 * 131072 * (1 * 4 * 512) + 50 * 1024 * (2 * 16 * 256)) / GiB, 1e-6, 'Gemma-4-31B: 10 global (k=v, 4×512) + 50 swa (2×16×256)');
  const g426 = run('gemma-4-26b-a4b', 'h100_80', { quant: 'fp8', kvQuant: 'fp8', seq: 131072 });
  near(seg(g426, 'kv'), (5 * 131072 * (1 * 2 * 512) + 25 * 1024 * (2 * 8 * 256)) / GiB, 1e-6, 'Gemma-4-26B-A4B: 5 global (k=v, 2×512) + 25 swa');
  const g4e2 = run('gemma-4-e2b', 'h100_80', { quant: 'fp16', kvQuant: 'fp16', seq: 32768 });
  const g4e2K = C.kvLayout(M('gemma-4-e2b'), C.geometry(M('gemma-4-e2b')));
  ok(g4e2K.fullElems === 2 * 1 * 512 && g4e2K.elems === 2 * 1 * 256, 'Gemma-4-E2B: global head_dim=512 (2× wider than sliding 256)');
  ok(seg(g4e2, 'kv') > 0.1, 'Gemma-4-E2B KV computed with heterogeneous global dims');

  // 3. Multi-user server throughput includes amortized prefill + PP8 beats TP8 at high concurrency on 405B 8×H200
  const tp1 = run('llama-3-1-405b', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'tensor', seq: 2048, outputTokens: 256, users: 1 });
  const pp1 = run('llama-3-1-405b', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'pipeline', seq: 2048, outputTokens: 256, users: 1 });
  ok(tp1.perf.tps > pp1.perf.tps * 4, 'Llama-405B 8×H200 u=1: TP8 much faster than PP8');
  const tpHi = run('llama-3-1-405b', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'tensor', seq: 2048, outputTokens: 256, users: 256 });
  const ppHi = run('llama-3-1-405b', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'pipeline', seq: 2048, outputTokens: 256, users: 256 });
  range(tpHi.perf.throughput, 400, 1100, 'Llama-405B 8×H200 TP8 u=256 prefill-amortized throughput (realistic, not 7000)');
  ok(ppHi.perf.throughput > tpHi.perf.throughput, 'Llama-405B 8×H200 u=256: PP8 beats TP8 in aggregate throughput', `PP=${ppHi.perf.throughput.toFixed(0)} vs TP=${tpHi.perf.throughput.toFixed(0)}`);

  // 4. llama.cpp on datacenter H100 realistic prefill & decode
  const lcH100 = run('llama-3-1-8b', 'h100_80', { quant: 'q4_k_m', runtime: 'llamacpp', seq: 1024, outputTokens: 128 });
  range(lcH100.perf.prefillTps, 7000, 16000, 'llama.cpp 8B Q4_K_M on H100 prefill ≈ 10k tok/s (not 49k)');
  range(lcH100.perf.tps, 230, 320, 'llama.cpp 8B Q4_K_M on H100 decode ≈ 270 tok/s (not 406)');

  // 5. Multi-GPU MoE bs=1 realistic speeds + consumer MoE preserved
  const ds8 = run('deepseek-v3', 'h200_141', { quant: 'fp8', numGpus: 8, parallelism: 'tensor', seq: 4096, outputTokens: 512, users: 1 });
  range(ds8.perf.tps, 80, 140, 'DeepSeek-V3 FP8 8×H200 TP8 u=1 ≈ 100–120 tok/s (InferenceX 2026: R1 SGLang 9.70 ms at c4)');
  // Reference is TensorRT-LLM (tech blog 9: TP8/EP1, batch 1, 1k/2k), so pin the runtime.
  const oss8 = run('gpt-oss-120b', 'b200_192', { quant: 'mxfp4', runtime: 'trtllm', numGpus: 8, parallelism: 'tensor', interconnect: 'nvlink_gen5', seq: 3072, outputTokens: 2048, users: 1 });
  range(oss8.perf.tps, 360, 480, 'gpt-oss-120b MXFP4 8×B200 TP8 TensorRT-LLM u=1 ≈ 420 tok/s');
  // Reference is llama.cpp (discussion #15396), so pin the runtime: the GGUF keeps attention/embeddings in Q8_0, vLLM in BF16.
  const oss20 = run('gpt-oss-20b', '4090_24', { quant: 'mxfp4', runtime: 'llamacpp', seq: 4096, outputTokens: 512, users: 1 });
  range(oss20.perf.tps, 175, 240, 'gpt-oss-20b MXFP4 RTX 4090 llama.cpp u=1 ≈ 200–222 tok/s');

  // 6. Diffusion: Wan 2.2 TI2V-5B 32× spatial, FLUX MMDiT ratio, sequential peak VRAM
  const wan = run('wan-2-2-ti2v-5b', '4090_24', { quant: 'fp16', runtime: 'diffusers', imageWidth: 1280, imageHeight: 704, numFrames: 121, inferenceSteps: 50, guidanceScale: 5 });
  ok(wan.perf && wan.memory.pct <= 100, 'Wan 2.2 TI2V-5B 720p 121f fits on RTX 4090', `pct=${wan.memory.pct.toFixed(1)}%`);
  range(wan.perf.seconds, 300, 700, 'Wan 2.2 TI2V-5B 720p 50-step on 4090 ≈ 8–10 min');
  const fl4090 = run('flux-1-dev', '4090_24', { quant: 'fp8', runtime: 'diffusers', inferenceSteps: 28 });
  range(fl4090.perf.seconds, 8, 16, 'FLUX.1-dev FP8 28-step on RTX 4090 ≈ 10–13 s');
  const segSum = fl4090.memory.segments.reduce((a, s) => a + s.gib, 0);
  near(segSum, fl4090.memory.perGpu, 1e-9, 'Diffusion segments sum identically to perGpu');

  // 7. Hardware & precision helpers (RTX 5090 FP4 8×, B200 180GB usable, MI355X, MLX 4-bit, GGUF large-vocab, no_rope_kv)
  ok(C.lowPrecisionX(C.QUANTS.nvfp4, Gk('5090_32'), 'vllm').x === 8, 'RTX 5090 NVFP4 = 8× halved FP32-acc BF16');
  ok(C.lowPrecisionX(C.QUANTS.mxfp4, Gk('mi355x_288'), 'vllm').x === 4, 'MI355X MXFP4 = 4× BF16');
  ok(C.lowPrecisionX(C.QUANTS.fp8, Gk('h100_80'), 'hf').x === 1, 'HF Transformers does not get W8A8 FP8 tensor-core boost');
  ok(C.supportsFp8(Gk('mi355x_288')) && Gk('r100_288').bandwidth_gbs === 19200, 'MI355X added & Rubin R100 bandwidth 19200 GB/s');
  near(C.usableGiB(Gk('b200_192')).v, 178.4, 1e-9, 'B200 usable HBM = 178.4 GiB (vLLM: 178.35 GiB total)');
  near(run('llama-3-1-8b', 'm4_max_64', { quant: 'int4', runtime: 'mlx' }).memory.weightsGiB, 8.03e9 * 4.5 / 8 / GiB, 1e-6, 'MLX 4-bit = 4.5 bpw across all weights');
  ok(run('gemma-3-1b', '4090_24', { quant: 'q4_k_m', runtime: 'llamacpp' }).memory.weightsGiB > 1.0e9 * 4.89 / 8 / GiB * 1.1, 'Gemma-3-1B Q4_K_M reflects Q8_0 large-vocab embedding overhead');
  ok(C.kvLayout(M('glm-53-flash'), C.geometry(M('glm-53-flash'))).elems === 512, 'GLM-5.3-Flash no_rope_kv uses 512 latent elems');

  // 8. Fine-tuning fused cross-entropy & ZeRO-3 gather buffer
  const ftStd = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 4, opt: { fused_cross_entropy: false } });
  const ftFce = run('llama-3-1-8b', 'h100_80', { mode: 'finetuning', ftMethod: 'lora', seq: 2048, batch: 4, opt: { fused_cross_entropy: true } });
  ok(seg(ftFce, 'logits') < seg(ftStd, 'logits') / 10, 'Fused cross-entropy cuts logits VRAM > 10×');
  const z3_2 = run('llama-3-1-70b', 'h100_80', { mode: 'finetuning', numGpus: 2, opt: { zero_stage: 3 } });
  const z3_8 = run('llama-3-1-70b', 'h100_80', { mode: 'finetuning', numGpus: 8, opt: { zero_stage: 3 } });
  near(seg(z3_8, 'comm'), seg(z3_2, 'comm'), 1e-9, 'ZeRO-3 gather buffer does not shrink with N');

  // 9. Malformed / zero / fractional inputs never produce NaN
  for (const bad of [
    { batch: 0, users: 0, numGpus: 2.5, seq: 1 },
    { mode: 'finetuning', batch: 0, gradSteps: 0, numSamples: 0, tokensPerSample: 0, numEpochs: 0, loraRank: 0, opt: undefined },
    { mode: 'inference', imageWidth: 0, imageHeight: 0, numFrames: 0, inferenceSteps: 0 },
  ]) {
    const mBad = bad.imageWidth === 0 ? M('flux-1-dev') : M('llama-3-1-8b');
    const rb = C.calculate({ ...base, ...bad }, mBad, Gk('h100_80'));
    ok(Number.isFinite(rb.memory.perGpu) && rb.memory.perGpu > 0 && (!rb.perf || Object.values(rb.perf).every((v) => typeof v !== 'number' || Number.isFinite(v))), 'Malformed input produces finite non-NaN output');
  }

  // 10. Datacenter llama.cpp long-context prefill applies MMQ dequant penalty to matmuls only, not FP16 FlashAttention
  const nem1m = run('nvidia-nemotron-3-nano-30b-a3b', 'b300_288', { quant: 'iq2_m', runtime: 'llamacpp', seq: 1048576, outputTokens: 511, batch: 8 });
  range(nem1m.perf.ttft / 1000, 75, 115, 'Nemotron-3-Nano 1M prompt on B300 llama.cpp TTFT ≈ 90–95 s (not 300 s)');
  range(nem1m.perf.prefillTps, 9500, 14000, 'Nemotron-3-Nano 1M prompt on B300 llama.cpp prefill ≈ 11k tok/s');
  const lcApple = run('codestral-2501', 'm3_max_36', { quant: 'iq2_m', runtime: 'ollama', seq: 49152, outputTokens: 511 });
  ok(!lcApple.memory.segments.some((s) => s.key === 'graph'), 'Apple Silicon does not reserve CUDA graphs');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
