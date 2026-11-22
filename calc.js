/* GPU Simulator — calculation engine.
 *
 * Plain, readable JavaScript. Every number the page shows is produced here, and every
 * step is recorded in `result.trace` so the page can print the formula next to the value.
 *
 * Units: memory is in GiB (2^30 bytes) — the same unit nvidia-smi, Activity Monitor and
 * GPU spec sheets use for "24 GB", "80 GB", etc. Bandwidth is GB/s (10^9 bytes/s) and
 * compute is TFLOPS (10^12 FLOP/s), as on vendor datasheets.
 */
(function (root) {
  'use strict';
  const GiB = 2 ** 30;
  const num = (x, d = 0) => { const v = Number.parseFloat(x); return Number.isFinite(v) ? v : d; };
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

  /* ------------------------------------------------------------------ *
   * Weight formats. bpw = effective bits per weight INCLUDING scales.
   * GGUF values are measured from real Llama-3-8B GGUF file sizes
   * (file bytes × 8 / 8.03e9 params), so they include llama.cpp's mixed
   * per-tensor types (e.g. Q4_K_M stores some tensors as Q6_K).
   * ------------------------------------------------------------------ */
  const QUANTS = {
    fp32:   { label: 'FP32', bpw: 32, group: 'Standard' },
    fp16:   { label: 'FP16 / BF16', bpw: 16, group: 'Standard' },
    fp8:    { label: 'FP8 (W8A8)', bpw: 8, group: 'Standard', emb16: true, fp8: true },
    int8:   { label: 'INT8 (W8A16)', bpw: 8.0625, group: 'Standard', emb16: true },      // + per-group fp16 scale / 256
    mxfp4:  { label: 'MXFP4', bpw: 4.25, group: 'Standard', emb16: true },               // 4-bit + 8-bit scale / 32
    nvfp4:  { label: 'NVFP4', bpw: 4.5, group: 'Standard', emb16: true },                // 4-bit + FP8 scale / 16
    int4:   { label: 'INT4 (AWQ / GPTQ g128)', bpw: 4.156, group: 'Standard', emb16: true }, // 4 + 16-bit scale/128 + 4-bit zero/128
    q8_0:   { label: 'Q8_0', bpw: 8.50, group: 'GGUF', gguf: true },
    q6_k:   { label: 'Q6_K', bpw: 6.56, group: 'GGUF', gguf: true },
    q5_k_m: { label: 'Q5_K_M', bpw: 5.70, group: 'GGUF', gguf: true },
    q5_k_s: { label: 'Q5_K_S', bpw: 5.57, group: 'GGUF', gguf: true },
    q4_k_m: { label: 'Q4_K_M', bpw: 4.89, group: 'GGUF', gguf: true },
    q4_k_s: { label: 'Q4_K_S', bpw: 4.66, group: 'GGUF', gguf: true },
    q4_0:   { label: 'Q4_0', bpw: 4.66, group: 'GGUF', gguf: true },
    iq4_xs: { label: 'IQ4_XS', bpw: 4.43, group: 'GGUF', gguf: true },
    q3_k_l: { label: 'Q3_K_L', bpw: 4.30, group: 'GGUF', gguf: true },
    q3_k_m: { label: 'Q3_K_M', bpw: 4.00, group: 'GGUF', gguf: true },
    q3_k_s: { label: 'Q3_K_S', bpw: 3.64, group: 'GGUF', gguf: true },
    iq3_m:  { label: 'IQ3_M', bpw: 3.77, group: 'GGUF', gguf: true },
    q2_k:   { label: 'Q2_K', bpw: 3.17, group: 'GGUF', gguf: true },
    iq2_m:  { label: 'IQ2_M', bpw: 2.94, group: 'GGUF', gguf: true },
  };
  // KV-cache element sizes in bytes (int4/fp4 include one fp16 scale per 32 values)
  const KV_BYTES = { fp16: 2, fp8: 1, int4: 0.5625, nvfp4: 0.5625 };
  const KV_LABELS = { fp16: 'FP16 / BF16', fp8: 'FP8', nvfp4: 'FP4', int4: 'INT4' };

  /* ------------------------------------------------------------------ *
   * Inference runtimes. Every constant is visible here and in the page.
   *  bwEff     achieved fraction of peak memory bandwidth in decode GEMV
   *  pfEff     achieved fraction of peak tensor TFLOPS during prefill
   *  layerUs   per-layer launch/sync overhead per decode step (µs)
   *  stepMs    fixed per-step cost: scheduler, sampling, detokenise (ms)
   *  chunk     prefill chunk (tokens processed per forward)
   *  ctxGiB    driver context + library workspaces per GPU
   *  graphGiB  CUDA-graph / compiled-kernel pools per GPU
   *  frag      allocator fragmentation as a share of dynamic memory
   * ------------------------------------------------------------------ */
  const RUNTIMES = {
    vllm:     { label: 'vLLM', bwEff: 0.80, pfEff: 0.55, layerUs: 6, stepMs: 0.30, chunk: 8192, ctxGiB: 0.55, graphGiB: 0.75, frag: 0.03, desc: 'Paged KV cache, continuous batching, CUDA graphs, chunked prefill (8192).' },
    sglang:   { label: 'SGLang', bwEff: 0.81, pfEff: 0.56, layerUs: 6, stepMs: 0.25, chunk: 8192, ctxGiB: 0.55, graphGiB: 0.70, frag: 0.03, desc: 'Radix-tree prefix cache, continuous batching, CUDA graphs.' },
    trtllm:   { label: 'TensorRT-LLM', bwEff: 0.84, pfEff: 0.62, layerUs: 4, stepMs: 0.20, chunk: 8192, ctxGiB: 0.50, graphGiB: 0.35, frag: 0.02, desc: 'Compiled engines with fused kernels and in-flight batching.' },
    llamacpp: { label: 'llama.cpp', bwEff: 0.74, pfEff: 0.42, layerUs: 10, stepMs: 0.35, chunk: 512, ctxGiB: 0.35, graphGiB: 0.10, frag: 0.02, desc: 'GGUF weights, 512-token micro-batches (n_ubatch), parallel slots.' },
    ollama:   { label: 'Ollama', bwEff: 0.72, pfEff: 0.40, layerUs: 10, stepMs: 0.60, chunk: 512, ctxGiB: 0.40, graphGiB: 0.10, frag: 0.02, desc: 'llama.cpp backend behind an HTTP server (adds a little per-token overhead).' },
    mlx:      { label: 'MLX (Apple)', bwEff: 0.78, pfEff: 0.60, layerUs: 14, stepMs: 0.30, chunk: 2048, ctxGiB: 0.25, graphGiB: 0, frag: 0.03, desc: 'Apple MLX with Metal kernels on unified memory.' },
    hf:       { label: 'HF Transformers', bwEff: 0.55, pfEff: 0.40, layerUs: 80, stepMs: 1.00, chunk: 1e9, ctxGiB: 0.60, graphGiB: 0, frag: 0.08, desc: 'Eager PyTorch, static batching, no paged KV cache (reference baseline).' },
  };

  /* Interconnects: unidirectional GB/s per GPU and small-message all-reduce latency (µs, NCCL/RCCL incl. launch). */
  const LINKS = {
    nvlink_gen6: { label: 'NVLink Gen 6 (3.6 TB/s)', bw: 1800, lat: 8 }, nvlink_gen5: { label: 'NVLink Gen 5 (1.8 TB/s)', bw: 900, lat: 8 },
    nvlink_gen4: { label: 'NVLink Gen 4 (900 GB/s)', bw: 450, lat: 10 }, nvlink_gen3: { label: 'NVLink Gen 3 (600 GB/s)', bw: 300, lat: 10 },
    nvlink_gen2: { label: 'NVLink Gen 2 (300 GB/s)', bw: 150, lat: 12 }, nvlink_bridge: { label: 'NVLink Bridge (112 GB/s)', bw: 56, lat: 14 },
    amd_xgmi: { label: 'AMD Infinity Fabric / xGMI (896 GB/s)', bw: 448, lat: 12 },
    pcie5: { label: 'PCIe 5.0 x16 (128 GB/s)', bw: 52, lat: 18 }, pcie4: { label: 'PCIe 4.0 x16 (64 GB/s)', bw: 26, lat: 20 }, pcie3: { label: 'PCIe 3.0 x16 (32 GB/s)', bw: 13, lat: 24 },
    infiniband_xdr: { label: 'InfiniBand XDR (800 Gb/s)', bw: 90, lat: 22 }, infiniband_ndr: { label: 'InfiniBand NDR (400 Gb/s)', bw: 45, lat: 22 },
    infiniband_hdr: { label: 'InfiniBand HDR (200 Gb/s)', bw: 22, lat: 25 }, infiniband_edr: { label: 'InfiniBand EDR (100 Gb/s)', bw: 11, lat: 28 },
    thunderbolt5: { label: 'Thunderbolt 5 (80 Gb/s)', bw: 7, lat: 80 }, thunderbolt4: { label: 'Thunderbolt 4/3 (40 Gb/s)', bw: 3.5, lat: 100 },
    ethernet400g: { label: '400 Gigabit Ethernet', bw: 45, lat: 35 }, ethernet200g: { label: '200 Gigabit Ethernet', bw: 22, lat: 35 },
    ethernet100g: { label: '100 Gigabit Ethernet', bw: 11, lat: 40 }, ethernet25g: { label: '25 Gigabit Ethernet', bw: 2.8, lat: 60 },
    ethernet10g: { label: '10 Gigabit Ethernet', bw: 1.1, lat: 80 }, ethernet1g: { label: '1 Gigabit Ethernet', bw: 0.11, lat: 150 },
  };

  const OFFLOAD = { cpu_ram: { label: 'System RAM', bw: 80 }, nvme: { label: 'NVMe SSD', bw: 6 } };
  const PCIE_HOST_BW = 25; // GB/s achievable host→device copy on PCIe 4.0 x16

  /* ------------------------------------------------------------------ *
   * Hardware helpers
   * ------------------------------------------------------------------ */
  const isApple = (g) => g.type === 'apple_silicon';
  // RTX 30/50 and RTX 4060–4080 Super list FP16-accumulate tensor rates; RTX 4090/4090D already list FP32-accumulate.
  const isGeForceFp16Acc = (g) => g.vendor === 'nvidia' && /^(30|50|40[678])\d/.test(g.key);

  /** Tensor TFLOPS a PyTorch/CUDA kernel can actually use (bf16 with fp32 accumulate). */
  function effTflops(g) {
    // GeForce cards list FP16-accumulate tensor rates; FP32-accumulate (what bf16 kernels use) runs at half.
    if (isGeForceFp16Acc(g)) return { v: g.fp16_tflops / 2, note: 'GeForce RTX 30/40/50: FP32-accumulate rate = ½ the listed FP16-accumulate rate' };
    // Apple GPUs run FP16 at the same ALU rate as FP32; the list doubles it (skip for custom GPU where user enters effective rate).
    if (isApple(g) && g.key !== 'custom') return { v: g.fp16_tflops / 2, note: 'Apple GPU: FP16 runs at the FP32 rate (½ listed)' };
    return { v: g.fp16_tflops, note: 'dense FP16/BF16 tensor rate' };
  }
  const isBlackwell = (g) => /^(b100|b200|b300|r100|gb10|50\d\d|rtx_pro_\d+|rtx_\d+_blackwell|rtx_4\d00_blackwell)/.test(g.key);
  /** Speed-up of the matmuls over BF16 when weights AND activations are low precision. */
  function lowPrecisionX(q, g) {
    if (q.fp8 && supportsFp8(g)) return { x: 2, label: 'FP8 tensor cores' };
    if (q === QUANTS.nvfp4 && isBlackwell(g)) {
      // dense FP4 ÷ dense BF16: B200 9 / 2.25 PF, B300 15 / 2.25 PF, Rubin 35 / 4 PF (NVFP4 training rate)
      const x = /^b300/.test(g.key) ? 6.67 : /^r100/.test(g.key) ? 8.75 : 4;
      return { x, label: 'NVFP4 tensor cores' };
    }
    return { x: 1, label: '' };
  }
  const supportsFp8 = (g) => /^(h100|h200|h800|h20|b100|b200|b300|r100|gb10|l40|l4_|l20|40|50|rtx_\d+_ada|rtx_\d+_blackwell|mi300|mi325|mi350|rx_9|radeon_ai_pro_r9700)/.test(g.key);

  /** Share of spec-sheet bandwidth that decode kernels can reach on this device family
   *  (measured: llama.cpp Apple-Silicon benchmark thread, ROCm/Vulkan reports). */
  function deviceBw(g) {
    if (isApple(g)) return /ultra/.test(g.key) ? { v: 0.62, note: 'Apple Ultra (two fused dies) reaches ~45–50% of spec bandwidth' } : { v: 0.78, note: 'Apple GPUs reach ~55–60% of spec bandwidth' };
    if (g.vendor === 'amd' && g.type !== 'apu') return { v: g.category === 'datacenter' ? 0.88 : 0.82, note: 'ROCm kernels reach slightly less of peak than CUDA' };
    if (g.vendor === 'intel') return { v: 0.75, note: 'Intel Arc (SYCL/Vulkan) kernels' };
    return { v: 1, note: '' };
  }

  /** Memory a single device lets you allocate for compute. */
  function usableGiB(g) {
    if (isApple(g)) {
      // macOS caps GPU-wired memory (recommendedMaxWorkingSetSize): ~2/3 below 36 GB, ~3/4 at 36 GB and above.
      const f = g.memory >= 36 ? 0.75 : 2 / 3;
      return { v: g.memory * f, note: `macOS GPU wired-memory limit ≈ ${f === 0.75 ? '75' : '67'}% of unified memory` };
    }
    return { v: g.memory, note: g.type === 'apu' ? 'GPU-accessible share of system memory' : 'total device memory' };
  }

  function autoRuntime(g, quant) {
    if (QUANTS[quant]?.gguf) return g.category === 'datacenter' ? 'llamacpp' : 'ollama';
    if (isApple(g)) return 'mlx';
    if (g.type === 'apu') return 'llamacpp';
    return 'vllm';
  }

  /* ------------------------------------------------------------------ *
   * Model geometry (with documented fallbacks for missing fields)
   * ------------------------------------------------------------------ */
  function geometry(m) {
    let P = num(m.num_of_params) * 1e9;
    const h = m.hidden_dim_size || Math.round(Math.sqrt(Math.max(P, 1e9) / 12 / 32) / 128) * 128 || 4096;
    const L = m.num_of_layers || Math.max(1, Math.round(Math.max(P, 1e9) / (12 * h * h)));
    const heads = m.num_attention_heads || Math.max(1, Math.round(h / 128));
    const att = m.attention_structure || 'mha';
    const mla = att === 'mla' || att === 'dsa' || att === 'csa2';
    // MLA uses decoupled qk_nope(128)+qk_rope(64)=192 and v=128 (mean 160); some dataset rows leak kv_latent_dim (512/576) into attention_head_dim.
    const rawHd = m.attention_head_dim;
    const hd = mla ? (rawHd && rawHd < 512 ? rawHd : 160) : (rawHd || h / heads);
    const kvh = m.num_key_value_heads || (att === 'mqa' ? 1 : heads);
    const V = m.vocab_size || 0;
    const moe = (m.architecture === 'moe' || (m.num_of_experts > 1 && num(m.num_of_expert_params) > 0)) && num(m.num_of_expert_params) > 0;
    const attnParams = mla ? h * 1536 + 1536 * heads * 192 + 576 * h + 512 * heads * 256 + heads * 128 * h // DeepSeek-style MLA projections
      : h * heads * hd * 2 + 2 * h * kvh * hd;
    const ffnKnown = m.ffn_intermediate_size;
    // If num_of_params is missing/zero, infer from layer geometry so weights & ratios never divide by 0.
    if (P <= 0) {
      const ffnEst = ffnKnown || 4 * h;
      const embEst = V ? V * h * (m.tie_word_embeddings === true ? 1 : 2) : 0;
      P = Math.max(1e6, L * (attnParams + 3 * h * ffnEst) + embEst);
    }
    const A = moe ? Math.min(P, num(m.num_of_expert_params) * 1e9) : P;
    // Tied vs untied embeddings: pick whichever makes the architecture add up to the stated size.
    let tied = m.tie_word_embeddings;
    if (tied == null) {
      if (V && ffnKnown && !moe) {
        const body = L * (attnParams + 3 * h * ffnKnown);
        tied = Math.abs(body + V * h - P) < Math.abs(body + 2 * V * h - P);
      } else tied = V * h * 2 > 0.12 * P; // large-vocab small models are almost always tied
    }
    const embedParams = V ? V * h * (tied ? 1 : 2) : 0;
    const ffn = ffnKnown || (!moe ? Math.max(h, Math.round((P - embedParams - L * attnParams) / (3 * h * L))) : 4 * h);
    // MoE split: guarantee shared + routed = P and shared + routed × f = A at sRep = 1.
    let f = moe ? A / P : 1, routed = 0, shared = P;
    if (moe) {
      const E = m.num_of_experts, k = m.num_of_active_experts;
      const fCand = E && k && k < E ? k / E : 0;
      if (fCand > 0 && fCand <= A / P) {
        f = fCand;
        routed = clamp((P - A) / (1 - f), 0, P);
        shared = P - routed;
      } else {
        routed = P;
        shared = 0;
        f = A / P;
      }
    }
    const ffnActive = moe ? (m.moe_ffn_intermediate_size || ffn) * ((m.num_of_active_experts || 2) + (m.num_of_shared_experts || 0)) : ffn;
    // Loaded but never streamed per decoded token: vision encoders, N-gram lookup tables (Qwen3.8-Flash-Next).
    const unread = num(m.num_of_unread_params) * 1e9;
    return { P, A, h, L, heads, hd, kvh, V, att, mla, moe, f, routed, shared, tied, embedParams, ffn, ffnActive, attnParams, unread,
      inferred: { h: !m.hidden_dim_size, L: !m.num_of_layers, heads: !m.num_attention_heads, hd: !m.attention_head_dim, kvh: !m.num_key_value_heads, ffn: !ffnKnown, V: !m.vocab_size } };
  }

  /** Bytes of loaded-but-unread params (vision encoders, lookup tables), kept at 16-bit for low-bit quants. */
  function unreadBytes(G, quant) {
    const q = QUANTS[quant] || QUANTS.fp16;
    return G.unread ? G.unread * (q.bpw >= 16 ? q.bpw / 8 : 2) : 0;
  }

  /** Bytes of weights for a precision; embeddings stay 16-bit for non-GGUF low-bit formats. */
  function weightBytes(G, quant) {
    const q = QUANTS[quant] || QUANTS.fp16;
    const emb = q.emb16 ? Math.min(G.embedParams, G.P) : 0;
    // Vision towers / lookup tables are kept at 16-bit by FP8/INT4/FP4 checkpoints and shipped as F16 mmproj files for GGUF.
    const ub = unreadBytes(G, quant);
    return (G.P - emb) * q.bpw / 8 + emb * 2 + ub;
  }

  /** KV-cache layout: how many layers keep full / windowed / constant-size state. */
  function kvLayout(m, G) {
    let L = G.L;
    const srcLayers = m.kv_source_layer_count && m.kv_source_layer_count < L ? m.kv_source_layer_count : L; // cross-layer KV sharing
    const lin = m.linear_layer_count != null ? m.linear_layer_count
      : (m.uses_linear_attention ? Math.round(L * (m.linear_attention_ratio ?? 0.75)) : 0);
    const totalSoft = m.attention_layer_count != null ? m.attention_layer_count : Math.max(0, L - lin);
    const soft = m.attention_layer_count != null ? Math.min(srcLayers, totalSoft) : Math.max(0, srcLayers - Math.round(lin * srcLayers / L));
    const swaOk = m.uses_sliding_window && m.sliding_window_size && !G.mla;
    const swa = swaOk ? Math.round(soft * (m.sliding_window_ratio ?? 1)) : 0;
    const full = soft - swa;
    // Attention-compute layers (cross-layer KV sharing skips KV storage, not attention compute)
    const swaAttn = swaOk ? Math.round(totalSoft * (m.sliding_window_ratio ?? 1)) : 0;
    const fullAttn = totalSoft - swaAttn;
    let elems, kind;
    if (G.mla) {
      const lat = m.kv_latent_dim || 512;
      elems = lat >= 576 ? lat : lat + 64; // compressed latent + decoupled RoPE key, shared by K and V
      kind = `MLA latent ${elems}`;
    } else { elems = 2 * G.kvh * G.hd; kind = `2 × ${G.kvh} KV heads × ${G.hd} dim`; }
    const indexer = G.att === 'dsa' || G.att === 'csa2' ? (m.indexer_dim || 128) : (m.indexer_dim || 0); // lightning-indexer keys (FP8, 1 byte)
    // Per-layer recurrent state (bf16), independent of length. With published Gated-DeltaNet / Mamba-2 geometry:
    // value_heads × key_dim × value_dim, plus the short-conv state (2·key_heads·key_dim + value_heads·value_dim) × (kernel − 1).
    // Without it, fall back to attention heads × head_dim².
    let linState = 0, linKind = '';
    if (lin && m.linear_num_value_heads) {
      const vh = m.linear_num_value_heads, kh = m.linear_num_key_heads || vh, dk = m.linear_key_head_dim || 128, dv = m.linear_value_head_dim || 128, ck = m.linear_conv_kernel_dim || 4;
      linState = (vh * dk * dv + (2 * kh * dk + vh * dv) * (ck - 1)) * 2;
      linKind = `${vh} value heads × ${dk}×${dv} state + conv (${2 * kh * dk + vh * dv} ch × ${ck - 1}) × 2 B`;
    } else if (lin) { linState = G.heads * G.hd * G.hd * 2; linKind = `${G.heads} heads × ${G.hd}² × 2 B (estimated)`; }
    return { full, swa, fullAttn, swaAttn, lin, window: m.sliding_window_size || 0, elems, kind, indexer, linState, linKind };
  }
  /** KV bytes for ONE sequence holding `T` tokens. */
  function kvBytesSeq(K, T, kvB) {
    const swaT = K.window ? Math.min(T, K.window) : T;
    return K.full * T * K.elems * kvB + K.swa * swaT * K.elems * kvB
      + (K.full * T + K.swa * swaT) * K.indexer + K.lin * K.linState;
  }
  /** KV bytes for `seqs` concurrent sequences of length `T` sharing the first `sharedTok` prompt tokens.
   *  Generated tokens and linear-attention recurrent states are never shared across concurrent sequences. */
  function kvBytesGroup(K, T, sharedTok, seqs, kvB) {
    if (seqs <= 1 || sharedTok <= 0) return kvBytesSeq(K, T, kvB) * seqs;
    const sh = clamp(sharedTok, 0, T);
    const un = T - sh;
    const swaT = K.window ? Math.min(T, K.window) : T;
    const unWin = K.window ? Math.min(un, K.window) : un;
    const shWin = Math.max(0, swaT - unWin);
    const fullTok = sh + seqs * un;
    const swaTok = shWin + seqs * unWin;
    return (K.full * fullTok + K.swa * swaTok) * (K.elems * kvB + K.indexer) + seqs * K.lin * K.linState;
  }

  /* ------------------------------------------------------------------ *
   * Shared: parallel layout and GPU set
   * ------------------------------------------------------------------ */
  function layout(inp) {
    if (inp.cluster) {
      const n = Math.max(1, inp.numNodes * inp.gpusPerNode);
      const tp = clamp(inp.tpDegree, 1, n);
      const pp = clamp(inp.ppDegree, 1, Math.max(1, Math.floor(n / tp)));
      const dp = Math.max(1, Math.floor(n / (tp * pp)));
      return { n, tp, pp, dp, intra: inp.interconnect, inter: inp.interNode, nodes: inp.numNodes, perNode: inp.gpusPerNode, overAllocated: inp.tpDegree * inp.ppDegree > n };
    }
    const n = inp.numGpus;
    if (n <= 1) return { n: 1, tp: 1, pp: 1, dp: 1, intra: null };
    const p = inp.parallelism;
    return { n, tp: p === 'tensor' ? n : 1, pp: p === 'pipeline' ? n : 1, dp: p === 'replicated' ? n : 1, intra: inp.interconnect, nodes: 1, perNode: n };
  }

  function statusFor(pct) {
    return pct > 100 ? 'insufficient' : pct > 95 ? 'very_high' : pct > 85 ? 'high' : pct > 75 ? 'moderate' : pct > 60 ? 'okay' : 'good';
  }

  /* ================================================================== *
   * LLM INFERENCE
   * ================================================================== */
  function inference(inp, m, g) {
    const T = [];          // trace
    const W = [];          // warnings
    const G = geometry(m);
    const q = QUANTS[inp.quant] || QUANTS.fp16;
    const kvB = KV_BYTES[inp.kvQuant] ?? 2;
    const rtKey = inp.runtime === 'auto' || !RUNTIMES[inp.runtime] ? autoRuntime(g, inp.quant) : inp.runtime;
    const R = RUNTIMES[rtKey];
    const Lyt = layout(inp);
    const shards = Lyt.tp * Lyt.pp;           // GPUs holding one copy of the model
    const replicas = Lyt.dp;
    const tfl = effTflops(g);
    const usable = usableGiB(g);
    const dBw = deviceBw(g);
    const bwEff = R.bwEff * dBw.v;
    const bw = g.bandwidth_gbs * 1e9;

    const seq = inp.seq;
    const outTok = clamp(inp.outputTokens, 1, seq - 1);
    const promptTok = seq - outTok;
    const users = inp.users, batch = inp.batch;
    const S = users * batch;                  // concurrent sequences in flight
    const uRep = Math.max(1, Math.ceil(users / replicas));    // peak concurrent users on one replica
    const sRep = uRep * batch;                                // peak concurrent sequences on one replica
    const share = uRep > 1 ? clamp(inp.prefixShare, 0, 1) : 0;
    const sharedTok = Math.round(promptTok * share);

    if (Lyt.overAllocated) W.push(`TP (${inp.tpDegree}) × PP (${inp.ppDegree}) exceeds the ${Lyt.n} GPUs available — clamped to TP ${Lyt.tp} × PP ${Lyt.pp}.`);

    // ---- weights ----
    const ub = unreadBytes(G, inp.quant);
    const wB = weightBytes(G, inp.quant);
    T.push({ g: 'Memory', k: 'Weights', f: `${fmtP(G.P)} params × ${q.bpw} bits ÷ 8${q.emb16 && G.embedParams ? ` (embeddings ${fmtP(G.embedParams)} kept at 16-bit)` : ''}${G.unread ? ` + ${fmtP(G.unread)} vision / lookup params at ${q.bpw >= 16 ? q.bpw : 16}-bit (loaded, not read per token)` : ''}`, v: gib(wB) });

    // ---- KV cache ----
    const K = kvLayout(m, G);
    const kvPerTok = (K.full + K.swa) * K.elems * kvB + (K.full + K.swa) * K.indexer;
    const kvSeq = kvBytesSeq(K, seq, kvB);
    const kvRep = kvBytesGroup(K, seq, sharedTok, sRep, kvB);
    // Exact cluster total across replicas (handles uneven user splits across DP replicas)
    const qU = Math.floor(users / replicas), rU = users % replicas;
    const kvForU = (u) => u <= 0 ? 0 : kvBytesGroup(K, seq, u > 1 ? Math.round(promptTok * clamp(inp.prefixShare, 0, 1)) : 0, u * batch, kvB);
    const kvTotal = rU * kvForU(qU + 1) + (replicas - rU) * kvForU(qU);
    T.push({ g: 'Memory', k: 'KV cache / token', f: `${K.full + K.swa} attention layers × (${K.kind}) × ${kvB} B${K.indexer ? ` + indexer ${K.indexer} B/layer` : ''}`, v: `${fmtBytes(kvPerTok)}` });
    if (K.swa) T.push({ g: 'Memory', k: 'Sliding window', f: `${K.swa} of ${K.full + K.swa} layers keep only the last ${K.window.toLocaleString()} tokens`, v: '' });
    if (K.lin) T.push({ g: 'Memory', k: 'Linear attention', f: `${K.lin} layers keep a fixed state: ${K.linKind}`, v: fmtBytes(K.lin * K.linState) + ' / seq' });
    T.push({ g: 'Memory', k: 'KV cache', f: `${fmtBytes(kvSeq)} per ${seq.toLocaleString()}-token sequence × ${sRep} seq/replica${replicas > 1 ? ` (${replicas} replicas)` : ''}${sharedTok ? `, ${sharedTok.toLocaleString()} shared prompt tok cached once` : ''}`, v: gib(kvTotal) });

    // ---- activations (peak during chunked prefill) + logits ----
    const chunkTok = Math.min(promptTok * batch, R.chunk);
    const prefillActB = chunkTok * (4 * G.h + 3 * G.ffnActive) * 2;
    const logitsActB = sRep * (G.V || 32000) * 4;
    const actRep = prefillActB + logitsActB;
    T.push({ g: 'Memory', k: 'Activations', f: `prefill chunk ${chunkTok.toLocaleString()} tok × (4·h + 3·ffn) × 2 B + ${sRep} × vocab × 4 B logits (per replica)`, v: gib(actRep) });

    // ---- speculative decoding draft ----
    let draftB = 0, draftKvRep = 0;
    const draftP = inp.spec ? Math.min(inp.specDraft, G.P / 1e9) * 1e9 : 0;
    if (inp.spec) {
      draftB = draftP * q.bpw / 8;
      draftKvRep = kvRep * (draftP / G.P);
      T.push({ g: 'Memory', k: 'Draft model', f: `${fmtP(draftP)} × ${q.bpw} bits ÷ 8 + KV ≈ target KV × ${(draftP / G.P).toFixed(3)}`, v: gib(draftB + draftKvRep) });
    }

    // ---- offloading ----
    const offFrac = inp.offload ? clamp(inp.offloadFrac, 0, 1) : 0;
    const offW = wB * offFrac;
    const offKvFlag = Boolean(inp.offload && inp.offloadKv);
    const offKv = offKvFlag ? kvTotal : 0;

    // ---- per-GPU split ----
    // TP shards weights and KV heads (KV replicated when tp > kv heads; MLA latent is replicated on every TP rank).
    const kvShard = G.mla ? 1 : Math.min(Lyt.tp, G.kvh);
    const wGpu = (wB - offW) / shards;
    const kvGpu = offKvFlag ? 0 : kvRep / (kvShard * Lyt.pp);
    const actGpu = actRep / Lyt.tp;
    const draftGpu = inp.spec ? draftB / shards + (offKvFlag ? 0 : draftKvRep / (kvShard * Lyt.pp)) : 0;
    const dyn = kvGpu + actGpu;
    const fragGpu = dyn * R.frag;
    const ctxGpu = R.ctxGiB * GiB;
    const graphGpu = R.graphGiB * GiB;
    const commGpu = Lyt.n > 1 ? 0.25 * GiB : 0; // NCCL buffers
    if (Lyt.tp > G.kvh && !G.mla) W.push(`TP degree ${Lyt.tp} exceeds the model's ${G.kvh} KV heads — KV heads are replicated, so KV memory does not shrink past ${G.kvh}-way.`);
    if (Lyt.tp > 1 && G.mla) W.push('MLA latent KV is replicated on every tensor-parallel rank (it cannot be split by head).');

    const segsGpu = [
      ['weights', G.moe ? 'Model weights (all experts)' : 'Model weights', wGpu],
      ['kv', 'KV cache', kvGpu],
      ['act', 'Activations & logits', actGpu],
      ...(inp.spec ? [['draft', 'Draft model', draftGpu]] : []),
      ['ctx', `${g.vendor === 'amd' ? 'ROCm' : isApple(g) ? 'Metal' : 'CUDA'} context & workspaces`, ctxGpu],
      ...(graphGpu ? [['graph', 'CUDA graphs', graphGpu]] : []),
      ...(commGpu ? [['comm', 'Collective buffers', commGpu]] : []),
      ['frag', 'Allocator fragmentation', fragGpu],
    ];
    T.push({ g: 'Memory', k: 'Runtime overhead', f: `${R.label}: context ${R.ctxGiB} GiB + graphs ${R.graphGiB} GiB + ${Math.round(R.frag * 100)}% fragmentation of KV+activations${commGpu ? ' + 0.25 GiB NCCL' : ''} (per GPU)`, v: gib(ctxGpu + graphGpu + fragGpu + commGpu) });

    const perGpu = segsGpu.reduce((a, s) => a + s[2], 0);
    const totalUse = perGpu * Lyt.n;
    const cap = usable.v * GiB;
    const pct = (perGpu / cap) * 100;
    T.push({ g: 'Memory', k: 'Per-GPU usage', f: `sum of the rows above ÷ ${usable.v.toFixed(1)} GiB usable (${usable.note})`, v: `${gib(perGpu)} (${pct.toFixed(1)}%)` });

    // ---- decode step time ----
    const avgCtx = promptTok + outTok / 2;
    const embInShared = Math.min(G.embedParams, G.shared);
    const inEmb = G.tied ? 0 : Math.min(G.V * G.h, embInShared) * (q.emb16 ? 2 : q.bpw / 8); // untied input table: only looked up, never read in full
    // Weights read per step. Dense: text weights minus untied input embeddings (never unread vision / lookup tables).
    // MoE: shared part (attention, norms, router, shared experts, lm_head) + the routed experts that at least
    // one of the sRep tokens selected: E[touched] = routed × (1 − (1 − k/E)^sRep).
    const embB = embInShared * (q.emb16 ? 2 : q.bpw / 8);
    const sharedB = Math.max(0, G.shared - embInShared) * q.bpw / 8 + embB - inEmb;
    const touchFrac = G.moe ? 1 - Math.pow(1 - G.f, sRep) : 1;
    const routedB = G.routed * q.bpw / 8 * touchFrac;
    const MOE_KERNEL_EFF = 0.75;                               // fused-MoE GEMVs reach ~75% of dense-GEMV bandwidth
    const wRead = G.moe ? sharedB + routedB : Math.max(0, wB - ub - inEmb);
    const wReadEff = G.moe ? sharedB + routedB / MOE_KERNEL_EFF : wRead; // bandwidth-equivalent bytes
    const kvRead = kvBytesGroup(K, avgCtx, Math.min(sharedTok, avgCtx), sRep, kvB);
    const gpuW = wReadEff * (1 - offFrac);
    const tMem = (gpuW / Lyt.tp + (offKvFlag ? 0 : kvRead / kvShard)) / (bw * bwEff);
    const flopsTok = 2 * (G.moe ? G.A : G.P) + 4 * (K.fullAttn * avgCtx + K.swaAttn * Math.min(avgCtx, K.window || avgCtx)) * G.heads * G.hd;
    // W8A8 FP8 and W4A4 NVFP4 checkpoints run their GEMMs on low-precision tensor cores (dense rates vs BF16).
    const lowP = lowPrecisionX(q, g);
    const tComp = (flopsTok * sRep) / (tfl.v * lowP.x * 1e12 * Lyt.tp * 0.5);
    const tFixed = (R.stepMs + G.L * R.layerUs * (G.moe ? 2.5 : 1) / 1000) / 1000; // MoE layers launch ~2.5× more kernels (router, top-k, permute)
    const tpLink = LINKS[inp.cluster && Lyt.tp > Lyt.perNode ? Lyt.inter : Lyt.intra] || LINKS.pcie4;
    const ppLink = LINKS[inp.cluster && Lyt.tp * Lyt.pp > Lyt.perNode ? Lyt.inter : Lyt.intra] || LINKS.pcie4;
    const commFor = (tokensPerRank) => {
      let c = 0;
      const msg = tokensPerRank * G.h * 2;
      if (Lyt.tp > 1) c += 2 * G.L * (tpLink.lat * 1e-6 + 2 * (Lyt.tp - 1) / Lyt.tp * msg / (tpLink.bw * 1e9));
      if (Lyt.pp > 1) c += (Lyt.pp - 1) * (ppLink.lat * 1e-6 + msg / (ppLink.bw * 1e9));
      return c;
    };
    const tComm = commFor(sRep);
    const offTarget = OFFLOAD[inp.offloadTarget] || OFFLOAD.cpu_ram;
    // llama.cpp/Ollama run offloaded layers on the CPU at system-RAM speed; other runtimes stream them over PCIe.
    const cpuRun = (rtKey === 'llamacpp' || rtKey === 'ollama') && inp.offloadTarget === 'cpu_ram';
    const offBw = inp.offloadTarget === 'nvme' ? offTarget.bw : cpuRun ? offTarget.bw * 0.7 : isApple(g) ? offTarget.bw : PCIE_HOST_BW;
    let tOff = 0;
    if (offFrac > 0) {
      tOff = wRead * offFrac / (offBw * 1e9);
      T.push({ g: 'Speed', k: 'Offloaded layers', f: `${fmtBytes(wRead * offFrac)} read per token from ${offTarget.label} at ${offBw} GB/s${cpuRun ? ' (CPU executes these layers)' : ' (streamed to the GPU)'}`, v: ms(tOff) });
    }
    if (offKvFlag) tOff += kvRead / (PCIE_HOST_BW * 1e9);
    const tStep = Math.max(tMem, tComp) + tFixed + tComm + tOff;
    T.push({ g: 'Speed', k: 'Bytes read per step', f: `${G.moe ? `weights: shared ${fmtBytes(sharedB)} + routed ${fmtBytes(G.routed * q.bpw / 8)} × (1 − (1 − ${G.f.toFixed(3)})^${sRep}) [MoE kernels at ${0.75 * 100}% eff.] = ` : 'weights '}${fmtBytes(wRead)} + KV at avg context ${Math.round(avgCtx).toLocaleString()} × ${sRep} seq ${fmtBytes(kvRead)}`, v: fmtBytes(wRead + kvRead) });
    T.push({ g: 'Speed', k: 'Memory time', f: `(weights ÷ ${Lyt.tp} + KV ÷ ${kvShard}) ÷ (${g.bandwidth_gbs} GB/s × ${bwEff.toFixed(2)} efficiency (${R.label} ${R.bwEff}${dBw.v < 1 ? ` × device ${dBw.v}` : ''}))`, v: ms(tMem) });
    T.push({ g: 'Speed', k: 'Compute time', f: `${(flopsTok / 1e9).toFixed(1)} GFLOP/token × ${sRep} ÷ (${tfl.v.toFixed(1)} TFLOPS${lowP.x > 1 ? ` × ${lowP.x} ${lowP.label}` : ''} × 0.5)  — ${tfl.note}`, v: ms(tComp) });
    T.push({ g: 'Speed', k: 'Fixed overhead', f: `${R.stepMs} ms/step + ${G.L} layers × ${R.layerUs * (G.moe ? 2.5 : 1)} µs${tComm ? ` + interconnect ${ms(tComm)}` : ''}`, v: ms(tFixed + tComm) });
    T.push({ g: 'Speed', k: 'Decode step', f: 'max(memory, compute) + fixed + interconnect + offload', v: ms(tStep) });

    // speculative decoding
    let perSeqTps = 1 / tStep, specInfo = null;
    if (inp.spec) {
      const a = clamp(inp.specAccept, 0.01, 0.999), k = inp.specTokens;
      const E = (1 - Math.pow(a, k + 1)) / (1 - a);
      const draftKvRead = kvRead * (draftP / G.P);
      const tDraft = (draftB / Lyt.tp + (offKvFlag ? 0 : draftKvRead / kvShard)) / (bw * bwEff) + R.stepMs / 2000 + 0.0002;
      // Verifying k+1 tokens activates more MoE experts per step and sends (k+1)× larger TP messages.
      const vTouch = G.moe ? 1 - Math.pow(1 - G.f, sRep * (k + 1)) : 1;
      const vRoutedB = G.routed * q.bpw / 8 * vTouch;
      const vWRead = G.moe ? sharedB + vRoutedB : wRead;
      const vWReadEff = G.moe ? sharedB + vRoutedB / MOE_KERNEL_EFF : wRead;
      const vTMem = (vWReadEff * (1 - offFrac) / Lyt.tp + (offKvFlag ? 0 : kvRead / kvShard)) / (bw * bwEff);
      const vTOff = (offFrac > 0 ? vWRead * offFrac / (offBw * 1e9) : 0) + (offKvFlag ? kvRead / (PCIE_HOST_BW * 1e9) : 0);
      const tVerify = Math.max(vTMem, tComp * (k + 1)) + tFixed + commFor(sRep * (k + 1)) + vTOff;
      const tCycle = (inp.specType === 'parallel' ? tDraft : k * tDraft) + tVerify;
      perSeqTps = E / tCycle;
      specInfo = { E, tDraft, tVerify, tCycle, speedup: perSeqTps * tStep };
      T.push({ g: 'Speed', k: 'Speculative decoding', f: `E[tokens/cycle] = (1 − α^(k+1)) / (1 − α) = ${E.toFixed(2)}; cycle = ${inp.specType === 'parallel' ? '1' : k} × draft ${ms(tDraft)} + verify ${ms(tVerify)}`, v: `${specInfo.speedup.toFixed(2)}× speed-up` });
    }
    const throughput = perSeqTps * S;

    // ---- prefill / TTFT (single request: `batch` prompts of promptTok tokens) ----
    const pfTok = promptTok * batch;
    const fp8x = lowP.x;
    // llama.cpp/Ollama prefill on quantized weights uses INT8 / FP16-acc MMQ kernels (≈2× effective BF16 rate on NVIDIA/AMD DC).
    const ggufPath = q.gguf && (rtKey === 'llamacpp' || rtKey === 'ollama');
    const int8x = g.int8_tensor === false ? 1 : (g.vendor === 'nvidia' || g.category === 'datacenter' ? 2 : 1); // B300/Rubin cut INT8 tensor throughput
    const pfTfl = ggufPath && !isApple(g) ? tfl.v * int8x : tfl.v * fp8x;
    const pfEff = isApple(g) ? 0.68 : R.pfEff;
    const swaPairs = K.window && promptTok > K.window ? K.window * (promptTok - K.window / 2) : promptTok * promptTok / 2;
    const attnPf = 4 * (K.fullAttn * promptTok * promptTok / 2 + K.swaAttn * swaPairs) * G.heads * G.hd * batch;
    const pfFlops = 2 * (G.moe ? G.A : G.P) * pfTok + attnPf;
    const pfComp = pfFlops / (pfTfl * 1e12 * pfEff * Lyt.tp);
    const pfChunkTok = Math.min(R.chunk, pfTok);
    const chunks = Math.ceil(pfTok / pfChunkTok);
    const pfW = Math.max(0, wB - ub);
    const pfMem = chunks * (pfW * (1 - offFrac)) / Lyt.tp / (bw * bwEff);
    const pfOff = offFrac > 0 ? chunks * pfW * offFrac / (offBw * 1e9) : 0;
    const pfComm = commFor(pfChunkTok);
    const pfFixed = chunks * (G.L * R.layerUs * 1e-6 + pfComm);
    const tPrefill = Math.max(pfComp, pfMem) + pfOff + pfFixed;
    const ttft = tPrefill + tStep;
    T.push({ g: 'Speed', k: 'Prefill FLOPs', f: `2 × ${fmtP(G.moe ? G.A : G.P)} ${G.moe ? 'active ' : ''}params × ${pfTok.toLocaleString()} prompt tokens + causal attention`, v: `${(pfFlops / 1e12).toFixed(2)} TFLOP` });
    T.push({ g: 'Speed', k: 'Time to first token', f: `max(FLOPs ÷ (${pfTfl.toFixed(1)} TFLOPS${ggufPath && !isApple(g) ? (int8x > 1 ? ' (INT8 MMQ)' : ' (MMQ, no INT8 boost)') : fp8x > 1 ? ` (${lowP.label})` : ''} × ${pfEff}${Lyt.tp > 1 ? ` × ${Lyt.tp}` : ''}), ${chunks} chunk weight reads)${pfOff ? ' + offload streaming' : ''} + first decode step`, v: ms(ttft) });
    const ttftQueued = uRep > 1 ? tPrefill * uRep + tStep : ttft;

    // ---- power ----
    const uMem = Math.min(1, (tMem / tStep) * bwEff), uComp = Math.min(1, (tComp / tStep) * 0.5);
    const watts = g.tdp_watts * Math.min(1, 0.2 + 0.5 * uMem + 0.8 * uComp) * Lyt.n;
    T.push({ g: 'Power', k: 'Power draw', f: `${Lyt.n} × TDP ${g.tdp_watts} W × (0.2 idle + 0.5 × memory-busy ${uMem.toFixed(2)} + 0.8 × compute-busy ${uComp.toFixed(2)})`, v: `${Math.round(watts)} W` });

    if (pct > 100 && !inp.offload) W.push('The configuration does not fit. Lower the precision, shorten the context, add GPUs, or enable offloading.');
    if (seq > (m.context_length || Infinity)) W.push(`Sequence length exceeds the model's ${m.context_length.toLocaleString()}-token context window.`);
    if (q.gguf && !['llamacpp', 'ollama'].includes(rtKey)) W.push(`${q.label} is a GGUF format — normally served by llama.cpp or Ollama.`);
    if (rtKey === 'mlx' && !isApple(g)) W.push('MLX only runs on Apple Silicon.');
    if (q.fp8 && !supportsFp8(g)) W.push(`${g.label} has no FP8 tensor cores: FP8 weights save memory but compute runs at FP16 speed (weight-only).`);
    if (inp.quant === 'nvfp4' && !isBlackwell(g)) W.push('NVFP4 is accelerated only on Blackwell; elsewhere it is dequantized to FP16 on the fly.');

    const status = statusFor(pct);
    const fits = pct <= 100;
    const perUserKv = Math.max(0, kvBytesSeq(K, seq, kvB) - (sharedTok ? (kvBytesSeq(K, seq, kvB) * 2 - kvBytesGroup(K, seq, sharedTok, 2, kvB)) : 0)) * batch;
    return {
      mode: 'inference', runtime: rtKey, runtimeLabel: R.label, layout: Lyt, status,
      memory: {
        perGpu: perGpu / GiB, total: totalUse / GiB, capPerGpu: usable.v, capTotal: usable.v * Lyt.n, rawPerGpu: g.memory, pct,
        segments: segsGpu.map(([key, label, b]) => ({ key, label, gib: b / GiB })),
        offloaded: offW + offKv > 0 ? [['Weights', offW], ['KV cache', offKv]].filter((x) => x[1] > 0).map(([label, b]) => ({ label, gib: b / GiB })) : [],
        offloadTarget: inp.offloadTarget,
        sharedGiB: ((wB - offW) * replicas + (ctxGpu + graphGpu + commGpu) * Lyt.n) / GiB, perUserGiB: perUserKv / GiB,
        kvPerToken: kvPerTok, weightsGiB: wB / GiB,
      },
      perf: fits ? {
        tps: perSeqTps, msPerToken: 1000 / perSeqTps, ttft: ttft * 1000, ttftQueued: ttftQueued * 1000, throughput,
        prefillTps: pfTok / tPrefill, stepMs: tStep * 1000, bound: tComp > tMem ? 'compute' : 'memory-bandwidth', spec: specInfo,
        promptTok, outTok, genSeconds: ttft + Math.max(0, outTok - 1) / perSeqTps,
      } : null,
      watts, trace: T, warnings: W, geometry: G,
    };
  }

  /* ================================================================== *
   * FINE-TUNING (text models)
   * ================================================================== */
  const OPT_BYTES = { adamw: 8, adam: 8, sgd: 4, adafactor: 0 };
  function finetune(inp, m, g) {
    const T = [], W = [];
    const G = geometry(m);
    const o = inp.opt;
    const method = inp.ftMethod;
    const Lyt = layout(inp);
    const N = Lyt.n;
    const zero = N > 1 ? o.zero_stage : 0;
    const tfl = effTflops(g);
    const usable = usableGiB(g);
    const seqLen = inp.seq, mb = inp.batch;
    const P = G.P;

    // ---- base / trainable weights ----
    let wBpp, trainable, wLabel;
    if (method === 'full') {
      wBpp = { fp32: 4, fp16: 2, fp8: 1 }[inp.ftQuant] ?? 2;
      trainable = P; wLabel = `${fmtP(P)} × ${wBpp} B (${inp.ftQuant.toUpperCase()})`;
    } else {
      wBpp = method === 'qlora' ? 4.127 / 8 : 2; // NF4 + double-quantised constants ≈ 4.127 bits
      // LoRA on all linear layers: r × (d_in + d_out) per projection
      const attnLora = G.mla
        ? (G.h + 1536) + (1536 + G.heads * 192) + (G.h + 576) + (512 + G.heads * 256) + (G.heads * 128 + G.h)
        : (G.h + G.heads * G.hd) * 2 + (G.h + G.kvh * G.hd) * 2;
      const ffnLora = G.moe
        ? 3 * (G.h + (m.moe_ffn_intermediate_size || G.ffn)) * ((m.num_of_experts || 8) + (m.num_of_shared_experts || 0))
        : 3 * (G.h + G.ffn);
      const perLayer = method === 'qlora' || method === 'lora' ? attnLora + ffnLora : 0;
      trainable = inp.loraRank * perLayer * G.L;
      wLabel = `${fmtP(P)} × ${method === 'qlora' ? '4.127 bits ÷ 8 (NF4 + double-quant)' : '2 B (frozen BF16)'}`;
    }
    const weights = P * wBpp + G.unread * 2;                                        // vision tower / lookup tables stay frozen in BF16
    if (G.unread) wLabel += ` + ${fmtP(G.unread)} frozen vision / lookup params × 2 B`;
    T.push({ g: 'Memory', k: 'Model weights', f: wLabel, v: gib(weights) });
    if (method !== 'full') T.push({ g: 'Memory', k: 'LoRA parameters', f: `r ${inp.loraRank} × Σ(d_in + d_out) over q,k,v,o,gate,up,down × ${G.L} layers`, v: fmtP(trainable) });

    // ---- gradients, optimizer, master weights ----
    const pureBf16 = !!o.use_bf16_true && (method !== 'full' || inp.ftQuant !== 'fp32');
    const gradB = method === 'full' ? Math.max(2, wBpp === 1 ? 2 : wBpp) : 4;        // LoRA adapters train in fp32
    const masterB = method === 'full' ? (wBpp < 4 && !pureBf16 ? 4 : 0) : 0;       // fp32 master copy in mixed precision
    let optB = OPT_BYTES[o.optimizer_type] ?? 8;
    if (o.optimizer_type === 'adafactor') optB = 8 / G.h;                            // factored second moments ≈ 4·(m+n)/(m·n) ≈ 8/h
    if (o.use_8bit_optimizer && optB > 0) optB = optB / 4;                          // 8-bit states (bitsandbytes)
    if (pureBf16 && optB > 0 && !o.use_8bit_optimizer) optB = optB / 2;             // bf16 optimizer states
    const loraParamB = method === 'full' ? 0 : trainable * 4;                       // fp32 adapter weights
    const grads = trainable * gradB, optim = trainable * optB, master = trainable * masterB;
    T.push({ g: 'Memory', k: 'Gradients', f: `${fmtP(trainable)} trainable × ${gradB} B`, v: gib(grads) });
    T.push({ g: 'Memory', k: 'Optimizer states', f: `${fmtP(trainable)} × ${optB.toFixed(optB < 1 ? 4 : 0)} B (${o.optimizer_type}${o.use_8bit_optimizer ? ', 8-bit' : ''}${pureBf16 ? ', bf16' : ''})${masterB ? ` + fp32 master weights ${masterB} B` : ''}`, v: gib(optim + master) });

    // ---- ZeRO / offload sharding ----
    const shard = (x, stage) => (zero >= stage ? x / N : x);
    let wGpu = shard(weights, 3), gGpu = shard(grads, 2), oGpu = shard(optim + master, 1);
    const loraGpu = shard(loraParamB, 3);
    const offOpt = Boolean(o.offload_optimizer && zero >= 1);
    const offGrad = Boolean(o.offload_gradients && zero >= 2);
    const offParam = Boolean(o.offload_param && zero >= 3);
    const off = [];
    if (offOpt) { off.push(['Optimizer states', optim + master]); oGpu = 0; }
    if (offGrad) { off.push(['Gradients', grads]); gGpu = 0; }
    if (offParam) { off.push(['Weights', weights]); wGpu = 0; }
    if (zero) T.push({ g: 'Memory', k: `ZeRO-${zero}`, f: `shards ${['', 'optimizer', 'optimizer + gradients', 'optimizer + gradients + weights'][zero]} across ${N} GPUs`, v: '' });

    // ---- activations (per micro-batch, bf16, flash attention) ----
    // Stored per token per layer: 7·h + 2·kv_dim + 4·ffn elements (norms, QKV, attn-out, residuals, gate/up/act/product)
    const KL = kvLayout(m, G);
    const kvDim = G.mla ? 576 : G.kvh * G.hd;
    const perTokLayer = (7 * G.h + 2 * kvDim + 4 * G.ffnActive) * 2;
    const effSeqLen = o.sequence_packing ? seqLen : o.dynamic_padding ? Math.min(seqLen, Math.ceil(inp.tokensPerSample * 1.15)) : seqLen;
    const softFrac = G.L > 0 ? (KL.fullAttn + KL.swaAttn) / G.L : 1;
    const attnScores = o.flash_attention ? 0 : 2 * G.heads * effSeqLen * 2 * softFrac; // eager attention stores T×T softmax on softmax layers
    const tokens = mb * effSeqLen;
    const ck = o.gradient_checkpointing ? clamp(o.gradient_checkpointing_ratio ?? 1, 0, 1) : 0;
    const fullLayers = G.L * (1 - ck);
    const actFull = tokens * (perTokLayer + attnScores) * fullLayers;
    const actCk = tokens * G.h * 2 * G.L * ck + (ck > 0 ? tokens * (perTokLayer + attnScores) : 0);
    let act = actFull + actCk;
    const logits = tokens * (G.V || 32000) * 4 * 2;                                  // fp32 logits + their gradient
    if (o.activation_offloading) act = act * 0.15;
    T.push({ g: 'Memory', k: 'Activations', f: `${tokens.toLocaleString()} tokens × (7h + 2·kv + 4·ffn) × 2 B × ${fullLayers.toFixed(0)} layers${ck ? ` + checkpoints h × 2 B × ${Math.round(G.L * ck)} layers` : ''}${o.activation_offloading ? ' × 0.15 (offloaded)' : ''}`, v: gib(act) });
    T.push({ g: 'Memory', k: 'Logits', f: `${tokens.toLocaleString()} × vocab ${(G.V || 32000).toLocaleString()} × 4 B × 2 (value + grad)`, v: gib(logits) });

    const ctx = (isApple(g) ? 0.3 : 0.6) * GiB, frag = 0.06 * (act + logits), comm = N > 1 ? 0.25 * GiB + (zero ? 2 * P * 2 / N * 0.05 : 0) : 0;
    const qBuf = method === 'qlora' ? P * 0.02 : 0; // dequantisation scratch
    const segs = [
      ['weights', method === 'full' ? 'Model weights' : 'Base weights (frozen)', wGpu],
      ...(method !== 'full' ? [['lora', 'LoRA adapters', loraGpu]] : []),
      ['grads', 'Gradients', gGpu],
      ['optim', 'Optimizer states', oGpu],
      ['act', 'Activations', act],
      ['logits', 'Logits', logits],
      ...(qBuf ? [['qbuf', 'QLoRA dequant buffers', qBuf]] : []),
      ['ctx', 'CUDA context & workspaces', ctx],
      ...(comm ? [['comm', 'Collective buffers', comm]] : []),
      ['frag', 'Allocator fragmentation', frag],
    ];
    const perGpu = segs.reduce((a, s) => a + s[2], 0);
    const pct = perGpu / (usable.v * GiB) * 100;

    // ---- speed ----
    // Per token: 2 × active params + causal attention 4 × (average attended tokens) × heads × head_dim on softmax layers only
    // (full layers attend T/2 on average; sliding-window layers W·(1 − W/(2T)); linear-attention layers are O(1)).
    const swaAvg = KL.window && effSeqLen > KL.window ? KL.window * (1 - KL.window / (2 * effSeqLen)) : effSeqLen / 2;
    const attended = KL.fullAttn * effSeqLen / 2 + KL.swaAttn * swaAvg;
    const flopsFwd = 2 * G.A + 4 * attended * G.heads * G.hd;
    const mult = method === 'full' ? 3 : 2;                                          // fwd + bwd (LoRA skips weight grads)
    const recompute = ck * 1;                                                        // one extra forward for checkpointed layers
    const flopsTok = flopsFwd * (mult + recompute);
    const fp8Train = method === 'full' && inp.ftQuant === 'fp8';
    const fp8TrainX = fp8Train && supportsFp8(g) ? 2 : 1;
    let mfu = g.category === 'datacenter' ? 0.42 : isApple(g) ? 0.30 : 0.38;
    if (fp8TrainX > 1) mfu *= 0.85;                                                  // FP8 cast/scale overhead → ≈1.7× net speedup
    if (!o.flash_attention) mfu *= 0.8;
    if (!o.use_fused_kernels) mfu *= 0.92;
    if (method === 'qlora') mfu *= 0.75;                                            // NF4 dequant on every matmul
    if (tokens < 2048) mfu *= Math.max(0.35, tokens / 2048);                        // tiny micro-batches under-fill the GPU
    let tps = mfu * tfl.v * fp8TrainX * 1e12 * N / flopsTok;
    const tokensPerStep = tokens * N * inp.gradSteps;
    // gradient all-reduce once per optimizer step (overlap 70%)
    let commS = 0;
    if (N > 1) {
      const link = LINKS[inp.cluster ? inp.interNode : inp.interconnect] || LINKS.pcie4;
      commS = 0.3 * 2 * (N - 1) / N * trainable * 2 / (link.bw * 1e9);
      // ZeRO-3 also all-gathers the sharded weights in forward and backward of every micro-batch (70% overlapped)
      if (zero >= 3) commS += inp.gradSteps * 0.3 * 2 * (N - 1) / N * P * wBpp / (link.bw * 1e9);
      tps = tokensPerStep / (tokensPerStep / tps + commS);
    }
    const offloadPenalty = (offOpt ? 0.85 : 1) * (offGrad ? 0.90 : 1) * (offParam ? 0.6 : 1) * (o.activation_offloading ? 0.9 : 1);
    tps *= offloadPenalty;
    T.push({ g: 'Speed', k: 'FLOPs / token', f: `(2 × ${fmtP(G.A)} + attention 4 × ${KL.full + KL.swa} layers × avg attended × d) × ${mult}${recompute ? ` + ${ck.toFixed(2)} recompute` : ''}`, v: `${(flopsTok / 1e9).toFixed(1)} GFLOP` });
    T.push({ g: 'Speed', k: 'Training throughput', f: `MFU ${mfu.toFixed(2)} × ${tfl.v.toFixed(1)} TFLOPS${fp8TrainX > 1 ? ' × 2 (FP8)' : ''} × ${N} GPU ÷ FLOPs/token${commS ? ` − ${zero >= 3 ? 'ZeRO-3 gathers + ' : ''}gradient all-reduce` : ''}${offloadPenalty < 1 ? ` × ${offloadPenalty.toFixed(2)} offload` : ''}`, v: `${Math.round(tps).toLocaleString()} tok/s` });

    const effTokPerSample = o.sequence_packing ? inp.tokensPerSample : Math.min(seqLen, inp.tokensPerSample);
    const padFactor = o.sequence_packing ? 1 : effSeqLen / effTokPerSample;
    const dataTok = inp.numSamples * effTokPerSample * inp.numEpochs;
    const computeTok = dataTok * padFactor;
    const hours = computeTok / tps / 3600;
    const steps = Math.ceil(computeTok / tokensPerStep);
    T.push({ g: 'Speed', k: 'Training time', f: `${fmtTok(dataTok)} data tokens × padding ${padFactor.toFixed(2)} ÷ ${Math.round(tps).toLocaleString()} tok/s (${steps.toLocaleString()} steps × ${tokensPerStep.toLocaleString()} tok)`, v: `${hours.toFixed(2)} h` });
    const watts = g.tdp_watts * 0.85 * N;
    T.push({ g: 'Power', k: 'Power draw', f: `${N} × TDP ${g.tdp_watts} W × 0.85 (sustained training load)`, v: `${Math.round(watts)} W` });
    if (N > 1 && zero === 0) W.push('ZeRO is off: every GPU holds a full copy of weights, gradients and optimizer states.');
    if (fp8Train && !supportsFp8(g)) W.push(`${g.label} has no FP8 tensor cores — FP8 training falls back to BF16/FP16 compute speed.`);
    if (seqLen > 32768) W.push('Very long sequences need gradient checkpointing or context parallelism.');
    if (pct > 100) W.push('Does not fit. Try LoRA/QLoRA, gradient checkpointing, an 8-bit optimizer, ZeRO-3 or a smaller micro-batch.');
    const status = statusFor(pct);
    const fits = pct <= 100;
    return {
      mode: 'finetuning', layout: Lyt, status,
      memory: { perGpu: perGpu / GiB, total: perGpu * N / GiB, capPerGpu: usable.v, capTotal: usable.v * N, rawPerGpu: g.memory, pct, segments: segs.map(([key, label, b]) => ({ key, label, gib: b / GiB })), offloaded: off.map(([label, b]) => ({ label, gib: b / GiB })), offloadTarget: 'cpu_ram' },
      perf: fits ? { trainTps: tps, samplesPerSec: tps / (effTokPerSample * padFactor), stepsPerSec: tps / tokensPerStep, tokensPerStep, hours, steps, dataTok, mfu } : null,
      watts, trace: T, warnings: W, geometry: G,
    };
  }

  /* ================================================================== *
   * DIFFUSION (image / video) — inference
   * ================================================================== */
  function diffusion(inp, m, g) {
    const T = [], W = [];
    const q = QUANTS[inp.quant] || QUANTS.fp16;
    const P = num(m.num_of_params) * 1e9, A = num(m.num_of_expert_params) * 1e9 || P, aux = num(m.num_of_auxiliary_params) * 1e9;
    const h = m.hidden_dim_size || 3072, L = m.num_of_layers || 28;
    const isUnet = /stable-diffusion-(1|2)|sdxl|sd-2|kandinsky|svd/.test(m.slug);
    const video = m.task_type === 'video_generation';
    const frames = video ? inp.numFrames : 1;
    const latFrames = video ? 1 + Math.floor((frames - 1) / 4) : 1;                // 4× temporal VAE compression
    const tokens = latFrames * Math.ceil(inp.imageWidth / 16) * Math.ceil(inp.imageHeight / 16) * (isUnet ? 0.25 : 1);
    const distilledCfg = /flux|schnell|turbo|lightning/.test(m.slug);
    const cfg = inp.guidanceScale > 1 && !distilledCfg ? 2 : 1;
    const Lyt = layout(inp);
    const tfl = effTflops(g);
    const usable = usableGiB(g);
    const b = inp.batch;

    const wB = P * q.bpw / 8;
    const encB = aux * 2;
    const encGpu = inp.encoderOffload ? 0 : encB;
    T.push({ g: 'Memory', k: 'Denoiser weights', f: `${fmtP(P)} × ${q.bpw} bits ÷ 8`, v: gib(wB) });
    T.push({ g: 'Memory', k: 'Text encoders', f: `${fmtP(aux)} × 2 B${inp.encoderOffload ? ' — offloaded to CPU after encoding' : ''}`, v: gib(encB) });
    const actB = b * cfg * tokens * h * 2 * 12;
    T.push({ g: 'Memory', k: 'Latent activations', f: `batch ${b} × CFG ${cfg} × ${Math.round(tokens).toLocaleString()} latent tokens × h ${h} × 2 B × 12 live tensors`, v: gib(actB) });
    const px = inp.imageWidth * inp.imageHeight * Math.min(frames, 8);
    const vaeB = (inp.vaeTiling ? Math.min(px, 512 * 512) : px) * 2560;
    T.push({ g: 'Memory', k: 'VAE decode workspace', f: `${inp.vaeTiling ? 'tile 512×512' : `${inp.imageWidth}×${inp.imageHeight}${video ? ` × ${Math.min(frames, 8)} frames` : ''}`} px × 2.5 KiB (128-ch conv feature maps)`, v: gib(vaeB) });
    const ctx = (isApple(g) ? 0.3 : 0.6) * GiB;
    const shard = inp.cluster ? Math.min(Lyt.tp, Lyt.n) : Math.max(1, Lyt.n);
    const actGpu = actB / Math.max(1, Lyt.n);
    const segs = [['weights', 'Denoiser weights', wB / shard], ...(encGpu ? [['enc', 'Text encoders', encGpu / shard]] : []), ['act', 'Latent activations', actGpu], ['vae', 'VAE decode workspace', vaeB], ['ctx', 'Runtime context', ctx], ['frag', 'Allocator fragmentation', 0.05 * (actGpu + vaeB)]];
    const perGpu = segs.reduce((a, s) => a + s[2], 0);
    const pct = perGpu / (usable.v * GiB) * 100;

    const linearFlops = 2 * A * tokens * cfg * b;
    const attnFlops = 4 * L * tokens * tokens * h * cfg * b;
    const flopsStep = linearFlops + attnFlops;
    const lowP = lowPrecisionX(q, g);
    const attnLowP = q.fp8 && supportsFp8(g) ? 1.5 : 1;
    // Smooth, strictly monotonic efficiency curve calibrated to FLUX/SDXL timings (higher TFLOPS is never slower).
    const eff = (0.60 / (1 + Math.pow(tfl.v / 1400, 0.75))) * (/tensorrt/.test(inp.runtime) ? 1.35 : 1);
    const gpuScale = Math.max(1, Lyt.n === 1 ? 1 : Lyt.n * 0.85);
    const tStep = (linearFlops / lowP.x + attnFlops / attnLowP) / (tfl.v * 1e12 * eff * gpuScale);
    const tVae = (px * frames / Math.min(frames, 8)) * 2.6e6 * b / (tfl.v * 1e12 * 0.3);
    const tEnc = inp.encoderOffload ? 0.8 : 0.05;
    const total = inp.inferenceSteps * tStep + tVae + tEnc;
    T.push({ g: 'Speed', k: 'FLOPs per step', f: `(2 × params × tokens + 4·L·tokens²·h) × CFG ${cfg} × batch ${b}`, v: `${(flopsStep / 1e12).toFixed(1)} TFLOP` });
    T.push({ g: 'Speed', k: 'Step time', f: `FLOPs ÷ (${tfl.v.toFixed(1)} TFLOPS${lowP.x > 1 ? ` × ${lowP.x} ${lowP.label}` : ''} × ${eff.toFixed(2)} achieved)`, v: ms(tStep) });
    T.push({ g: 'Speed', k: 'Generation time', f: `${inp.inferenceSteps} steps × step + VAE decode ${ms(tVae)} + text encode ${ms(tEnc)}`, v: `${total.toFixed(1)} s` });
    const watts = g.tdp_watts * 0.85 * Lyt.n;
    if (q.fp8 && !supportsFp8(g)) W.push(`${g.label} has no FP8 tensor cores: FP8 saves memory but runs at FP16 speed.`);
    if (inp.quant === 'nvfp4' && !isBlackwell(g)) W.push('NVFP4 is accelerated only on Blackwell; elsewhere it is dequantized to FP16 on the fly.');
    if (pct > 100) W.push('Does not fit. Enable text-encoder offload or VAE tiling, lower the resolution, or use FP8 weights.');
    return {
      mode: 'inference', diffusion: true, layout: Lyt, status: statusFor(pct),
      memory: { perGpu: perGpu / GiB, total: perGpu * Lyt.n / GiB, capPerGpu: usable.v, capTotal: usable.v * Lyt.n, rawPerGpu: g.memory, pct, segments: segs.map(([key, label, bb]) => ({ key, label, gib: bb / GiB })), offloaded: inp.encoderOffload && encB ? [{ label: 'Text encoders', gib: encB / GiB }] : [], offloadTarget: 'cpu_ram' },
      perf: pct <= 100 ? { seconds: total, itPerSec: 1 / tStep, stepMs: tStep * 1000, latentTokens: Math.round(tokens) } : null,
      watts, trace: T, warnings: W,
    };
  }

  /* ------------------------------------------------------------------ */
  function calculate(inp, model, gpu) {
    const g = { ...gpu };
    let r;
    const isDiff = model.task_type === 'image_generation' || model.task_type === 'video_generation';
    if (isDiff) r = diffusion(inp, model, g);
    else if (inp.mode === 'finetuning') r = finetune(inp, model, g);
    else r = inference(inp, model, g);
    // energy & cost (same for every mode)
    const kw = r.watts / 1000;
    r.energy = {
      watts: r.watts, kwhPerDay: kw * 24,
      co2KgPerDay: kw * 24 * inp.carbon / 1000, co2KgPerYear: kw * 24 * 365 * inp.carbon / 1000,
      elecPerHour: kw * inp.elec,
    };
    r.trace.push({ g: 'Cost', k: 'Electricity', f: `${kw.toFixed(3)} kW × $${inp.elec}/kWh`, v: `$${(kw * inp.elec).toFixed(4)}/h` });
    r.trace.push({ g: 'Cost', k: 'CO₂', f: `${kw.toFixed(3)} kW × 24 h × ${inp.carbon} g/kWh`, v: `${(kw * 24 * inp.carbon / 1000).toFixed(2)} kg/day` });
    return r;
  }

  /* formatting helpers used in traces */
  function gib(b) { return `${(b / GiB).toFixed(2)} GiB`; }
  function fmtBytes(b) { return b >= GiB ? `${(b / GiB).toFixed(2)} GiB` : b >= 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MiB` : `${(b / 1024).toFixed(1)} KiB`; }
  function fmtP(p) { return p >= 1e9 ? `${(p / 1e9).toFixed(2)}B` : p >= 1e6 ? `${(p / 1e6).toFixed(1)}M` : `${Math.round(p / 1e3)}K`; }
  function fmtTok(t) { return t >= 1e9 ? `${(t / 1e9).toFixed(2)}B` : t >= 1e6 ? `${(t / 1e6).toFixed(1)}M` : `${Math.round(t / 1e3)}K`; }
  function ms(s) { return s >= 1 ? `${s.toFixed(2)} s` : `${(s * 1000).toFixed(2)} ms`; }


  const api = { calculate, geometry, weightBytes, unreadBytes, kvLayout, kvBytesSeq, kvBytesGroup, effTflops, deviceBw, usableGiB, autoRuntime, supportsFp8, QUANTS, KV_BYTES, KV_LABELS, RUNTIMES, LINKS, OFFLOAD, GiB };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.GpuCalc = api;
})(typeof window !== "undefined" ? window : globalThis);
