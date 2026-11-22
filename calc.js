/* LLM Simulator — calculation engine.
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
    q4_k_s: { label: 'Q4_K_S', bpw: 4.67, group: 'GGUF', gguf: true },
    q4_0:   { label: 'Q4_0', bpw: 4.64, group: 'GGUF', gguf: true },
    iq4_xs: { label: 'IQ4_XS', bpw: 4.46, group: 'GGUF', gguf: true },
    q3_k_l: { label: 'Q3_K_L', bpw: 4.30, group: 'GGUF', gguf: true },
    q3_k_m: { label: 'Q3_K_M', bpw: 4.00, group: 'GGUF', gguf: true },
    q3_k_s: { label: 'Q3_K_S', bpw: 3.64, group: 'GGUF', gguf: true },
    iq3_m:  { label: 'IQ3_M', bpw: 3.76, group: 'GGUF', gguf: true },
    q2_k:   { label: 'Q2_K', bpw: 3.16, group: 'GGUF', gguf: true },
    iq2_m:  { label: 'IQ2_M', bpw: 2.93, group: 'GGUF', gguf: true },
  };
  // KV-cache element sizes in bytes. q8_0 / q4_0: 32-value blocks with an fp16 scale (llama.cpp, Ollama; MLX 8/4-bit group-64
  // with scale+bias has the same 8.5 / 4.5 bits). nvfp4: E2M1 + one E4M3 scale per 16. int4: + fp16 scale per 32.
  const KV_BYTES = { fp16: 2, fp8: 1, q8_0: 1.0625, nvfp4: 0.5625, q4_0: 0.5625, int4: 0.5625 };
  const KV_LABELS = { fp16: 'FP16 / BF16', fp8: 'FP8', q8_0: 'Q8_0 / 8-bit (llama.cpp, Ollama, MLX)', nvfp4: 'FP4 (NVFP4)', q4_0: 'Q4_0 / 4-bit (llama.cpp, Ollama, MLX)', int4: 'INT4 (HF quantized cache)' };
  // What each runtime actually accepts (vLLM CacheConfig: auto/fp8; TensorRT-LLM: fp8/nvfp4; llama.cpp --cache-type-k/v;
  // Ollama OLLAMA_KV_CACHE_TYPE f16/q8_0/q4_0; mlx-lm --kv-bits; HF QuantizedCache). Weight formats likewise.
  const KV_SUPPORT = { vllm: ['fp16', 'fp8'], sglang: ['fp16', 'fp8'], trtllm: ['fp16', 'fp8', 'nvfp4'], llamacpp: ['fp16', 'q8_0', 'q4_0'], ollama: ['fp16', 'q8_0', 'q4_0'], mlx: ['fp16', 'q8_0', 'q4_0'], hf: ['fp16', 'int4'] };
  const WEIGHT_UNSUPPORTED = { llamacpp: ['fp8', 'int8', 'int4', 'nvfp4'], ollama: ['fp8', 'int8', 'int4', 'nvfp4'], mlx: ['fp8', 'nvfp4'], hf: ['nvfp4'] };

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
    nvlink_gen2: { label: 'NVLink Gen 2 (300 GB/s)', bw: 150, lat: 12 }, nvlink_bridge_a30: { label: 'NVLink Bridge, A30 (200 GB/s)', bw: 100, lat: 14 }, nvlink_bridge: { label: 'NVLink Bridge (112.5 GB/s)', bw: 56, lat: 14 },
    amd_xgmi: { label: 'AMD Infinity Fabric / xGMI (7 × 128 GB/s)', bw: 448, lat: 12, perPeer: 64 }, // fully connected: 64 GB/s each way per peer link
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
  function effTflops(g, rtKey) {
    // Apple M5 GPUs have per-core Neural Accelerators that MLX (macOS 26.2+) uses for matmuls: measured FP16 matmul on a 40-core
    // M5 Max is 55.6–65.6 TFLOPS (arXiv 2611.00486); Apple reports up to 4× M4 time-to-first-token with MLX.
    if (rtKey === 'mlx' && g.matmul_tflops) return { v: g.matmul_tflops, note: 'M5 GPU Neural Accelerators (measured FP16 matmul, used by MLX)' };
    // GeForce cards list FP16-accumulate tensor rates; FP32-accumulate (what bf16 kernels use) runs at half.
    if (isGeForceFp16Acc(g)) return { v: g.fp16_tflops / 2, note: 'GeForce RTX 30/40/50: FP32-accumulate rate = ½ the listed FP16-accumulate rate' };
    // Apple GPUs run FP16 at the same ALU rate as FP32; the list doubles it (skip for custom GPU where user enters effective rate).
    if (isApple(g) && !g.key.startsWith('custom')) return { v: g.fp16_tflops / 2, note: 'Apple GPU: FP16 runs at the FP32 rate (½ listed)' };
    return { v: g.fp16_tflops, note: 'dense FP16/BF16 tensor rate' };
  }
  const isBlackwell = (g) => /^(b100|b200|b300|r100|gb10|50\d\d|rtx_pro_\d+|rtx_\d+_blackwell|rtx_4\d00_blackwell)/.test(g.key);
  const supportsFp8 = (g) => /^(h100|h200|h800|h20|b100|b200|b300|r100|gb10|l40|l4_|l20|40|50|rtx_\d+_ada|rtx_\d+_blackwell|mi300|mi325|mi35[05]|rx_9|radeon_ai_pro_r9700)/.test(g.key);
  const supportsMxfp4 = (g) => isBlackwell(g) || /^mi35[05]/.test(g.key);
  /** Speed-up of the matmuls over effective BF16 when weights AND activations run on low-precision tensor cores. */
  function lowPrecisionX(q, g, rtKey) {
    // Only serving / compiled engines run native W8A8 FP8 and W4A4 FP4 tensor-core kernels (not llama.cpp/Ollama/HF/MLX).
    if (rtKey && !['vllm', 'sglang', 'trtllm', 'diffusers', 'tensorrt'].includes(rtKey)) return { x: 1, label: '' };
    if (q.fp8 && supportsFp8(g)) return { x: 2, label: 'FP8 tensor cores' };
    if (q === QUANTS.nvfp4 && isBlackwell(g)) {
      // dense FP4 ÷ effective BF16: RTX 50-series 4× FP16-acc = 8× halved FP32-acc; B200 9/2.25=4;
      // HGX B300 13.5/2.25=6.0 (108 PF per 8 GPUs; the 15 PF figure is the 1400 W GB300); R100 35/4=8.75
      const x = /^50\d\d/.test(g.key) ? 8 : /^b300/.test(g.key) ? 6.0 : /^r100/.test(g.key) ? 8.75 : 4;
      return { x, label: 'NVFP4 tensor cores' };
    }
    if (q === QUANTS.mxfp4 && supportsMxfp4(g)) {
      const x = /^50\d\d/.test(g.key) ? 8 : /^b300/.test(g.key) ? 6.0 : /^r100/.test(g.key) ? 8.75 : 4;
      return { x, label: 'MXFP4 tensor cores' };
    }
    return { x: 1, label: '' };
  }

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
    if (g.usable_memory && g.usable_memory < g.memory) {
      return { v: g.usable_memory, note: `usable HBM (${g.memory} GB raw, ${g.memory - g.usable_memory} GB reserved/ECC)` };
    }
    return { v: g.memory, note: g.type === 'apu' ? 'GPU-accessible share of system memory' : 'total device memory' };
  }

  function autoRuntime(g, quant) {
    if (QUANTS[quant]?.gguf) return g.category === 'datacenter' ? 'llamacpp' : 'ollama';
    if (isApple(g)) return 'mlx';
    // APUs: llama.cpp for GGUF / MXFP4 GGUF; safetensors formats (AWQ, FP8…) need a PyTorch engine.
    if (g.type === 'apu') return quant === 'mxfp4' || quant === 'fp16' ? 'llamacpp' : g.vendor === 'nvidia' ? 'vllm' : 'hf';
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
        // The shared part can never be smaller than embeddings + softmax-attention projections. Some published "active"
        // counts exclude the input embedding (gpt-oss, MiniMax-M2), which would otherwise push attention into the routed pool.
        const linL = m.linear_layer_count != null ? m.linear_layer_count : (m.uses_linear_attention ? Math.round(L * (m.linear_attention_ratio ?? 0.75)) : 0);
        const softLayers = m.attention_layer_count != null ? m.attention_layer_count : Math.max(0, L - linL);
        // Shared experts run on every token too, so they belong in the always-active bucket (MiniMax-M3's "23B active"
        // excludes both embedding tables; without this its shared part would be 10.0B instead of ≈13B).
        const sharedExperts = (m.num_of_shared_experts || 0) * 3 * h * (m.moe_ffn_intermediate_size || 0) * L;
        const sharedFloor = Math.min(P, embedParams + softLayers * attnParams + sharedExperts);
        if (shared < sharedFloor) { shared = sharedFloor; routed = P - shared; }
      } else {
        routed = P;
        shared = 0;
        f = A / P;
      }
    }
    const ffnActive = moe ? (m.moe_ffn_intermediate_size || ffn) * ((m.num_of_active_experts || 2) + (m.num_of_shared_experts || 0)) : ffn;
    // Loaded but never streamed per decoded token: vision encoders, N-gram lookup tables (Qwen3.8-Flash-Next).
    const unread = num(m.num_of_unread_params) * 1e9;
    // Parameter classes: output head (lm_head, = token table when tied), untied input table (a lookup, never a matmul),
    // softmax-attention projections, and the rest (FFN / experts / norms / linear-attention mixers).
    const linLayers = m.linear_layer_count != null ? m.linear_layer_count : (m.uses_linear_attention ? Math.round(L * (m.linear_attention_ratio ?? 0.75)) : 0);
    const softL = m.attention_layer_count != null ? m.attention_layer_count : Math.max(0, L - linLayers);
    const outP = V ? Math.min(V * h, P) : 0;
    const inP = V && !tied ? Math.min(V * h, Math.max(0, P - outP)) : 0;
    const attnIn = mla ? h * (1536 + 576) : h * heads * hd + 2 * h * kvh * hd; // projections whose rows have length h
    // Matmul FLOPs per token use active params minus the untied input lookup. For MoE, shared + routed·k/E is the
    // active set (it equals A when the published count includes the input table, and corrects it when it does not).
    const Amm = moe ? Math.max(0, shared - inP) + routed * f : Math.max(0, P - inP);
    return { P, A, Amm, h, L, heads, hd, kvh, V, att, mla, moe, f, routed, shared, tied, embedParams, ffn, ffnActive, attnParams, attnIn, softL, outP, inP, unread,
      inferred: { h: !m.hidden_dim_size, L: !m.num_of_layers, heads: !m.num_attention_heads, hd: !m.attention_head_dim, kvh: !m.num_key_value_heads, ffn: !ffnKnown, V: !m.vocab_size } };
  }

  /** Bytes of loaded-but-unread params (vision encoders, lookup tables), kept at 16-bit for low-bit quants. */
  function unreadBytes(G, quant) {
    const q = QUANTS[quant] || QUANTS.fp16;
    return G.unread ? G.unread * (q.bpw >= 16 ? q.bpw / 8 : 2) : 0;
  }

  /* GGUF tensor layout (llama.cpp llama_tensor_get_type):
   *  - output.weight (or the tied token table) is stored as Q6_K for every K/I-quant, Q8_0 for Q8_0;
   *  - K-quants pack 256-value super-blocks, so tensors whose rows (input dim) are not a multiple of 256 fall back to
   *    32-value legacy types (Q2_K…IQ4_XS → IQ4_NL, Q4_K → Q5_0, Q5_K → Q5_1, Q6_K → Q8_0). E.g. Gemma-3-1B (h = 1152).
   * The bpw table is the llama.cpp README file average on Llama-3.1-8B, so the body rate is backed out from it. */
  const GGUF_FALLBACK_BPW = { q6_k: 8.5, q5_k_m: 6.0, q5_k_s: 6.0, q4_k_m: 5.5, q4_k_s: 5.5, iq4_xs: 4.5, q3_k_l: 4.5, q3_k_m: 4.5, q3_k_s: 4.5, iq3_m: 4.5, q2_k: 4.5, iq2_m: 4.5 };
  const ggufOutBpw = (quant) => (quant === 'q8_0' ? 8.5 : 6.5625);
  function ggufBodyBpw(quant) {
    const q = QUANTS[quant], pRef = 8.03e9, outRef = 128256 * 4096;
    return (pRef * q.bpw - outRef * ggufOutBpw(quant)) / (pRef - outRef);
  }

  /** Byte rates per parameter class. Params split into output head (outP), untied input table (inP), softmax-attention
   *  projections (attnP) and the rest (FFN / experts / norms). One function drives both memory and decode reads. */
  function paramRates(G, quant, rtKey) {
    const q = QUANTS[quant] || QUANTS.fp16;
    const outP = G.outP, inP = G.inP;
    const attnP = Math.min(G.softL * G.attnParams, Math.max(0, G.P - outP - inP));
    const restP = Math.max(0, G.P - outP - inP - attnP);
    const ggufRt = rtKey === 'llamacpp' || rtKey === 'ollama';
    // MLX 4-bit: affine group-64 (4-bit + fp16 scale + fp16 bias) = 4.5 bpw on every weight, embeddings included.
    if (rtKey === 'mlx' && quant === 'int4') {
      const r = 4.5 / 8;
      return { outP, inP, attnP, restP, outRate: r, inRate: r, attnRate: r, restRate: r, note: 'MLX 4-bit (group 64) = 4.5 bpw on all weights' };
    }
    if (q.gguf) {
      const body = ggufBodyBpw(quant);
      const misaligned = G.h % 256 !== 0 && GGUF_FALLBACK_BPW[quant];
      const fb = misaligned ? GGUF_FALLBACK_BPW[quant] : body;
      const attnFrac = G.attnParams > 0 ? Math.min(1, G.attnIn / G.attnParams) : 0;
      const attnBpw = attnFrac * fb + (1 - attnFrac) * body;            // q/k/v rows have length h; o does not
      const restBpw = (2 / 3) * fb + (1 / 3) * body;                     // gate/up rows have length h; down does not
      const outBpw = G.h % 256 !== 0 ? 8.5 : ggufOutBpw(quant);                 // Q6_K output also needs 256-aligned rows
      return { outP, inP, attnP, restP, outRate: outBpw / 8, inRate: fb / 8, attnRate: attnBpw / 8, restRate: restBpw / 8,
        note: `GGUF: output tensor ${outBpw === 8.5 ? 'Q8_0' : 'Q6_K'}${misaligned ? `, rows of length ${G.h} (not a multiple of 256) fall back to ${fb}-bpw legacy blocks` : ''}` };
    }
    const r = q.bpw / 8;
    // Non-GGUF low-bit checkpoints keep embeddings / lm_head in 16-bit; the MXFP4 GGUF keeps them (and attention) in Q8_0.
    const keptRate = ggufRt && quant === 'mxfp4' ? 8.5 / 8 : 2;
    const keptLabel = keptRate === 2 ? '16-bit' : 'Q8_0';
    const embRate = q.emb16 ? keptRate : r;
    // Production MoE FP4 checkpoints (gpt-oss MXFP4, DeepSeek-R1 NVFP4) keep attention projections unquantized too.
    const attnKept = (quant === 'mxfp4' || quant === 'nvfp4') && G.moe;
    return { outP, inP, attnP, restP, outRate: embRate, inRate: embRate, attnRate: attnKept ? keptRate : r, restRate: r,
      note: q.emb16 ? `embeddings${attnKept ? ' + attention' : ''} kept at ${keptLabel}` : '' };
  }

  /** Bytes of weights for a precision (all parameter classes + loaded-but-unread vision / lookup params). */
  function weightBytes(G, quant, rtKey) {
    const p = paramRates(G, quant, rtKey);
    return p.outP * p.outRate + p.inP * p.inRate + p.attnP * p.attnRate + p.restP * p.restRate + unreadBytes(G, quant);
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
    // Sparse global attention (DeepSeek-V3.2 / GLM-5 DSA, DeepSeek-V4 CSA, V4.1 CSA2): a lightning indexer scores every
    // cached entry and the main attention reads only the top-k. V4 additionally stores global KV compressed per layer:
    // kv_compress_layers = { ratio: layers } (V4-Flash config compress_ratios: 21 layers ÷4 (CSA, sparse) + 20 ÷128 (HCA, dense)).
    // MiniMax-M3 style block-sparse GQA: sparse_attention_layers use top-k (blocks × block size) tokens, the rest stay dense.
    const sparseGqa = !G.mla && m.sparse_attention_layers > 0;
    const sparse = G.att === 'dsa' || G.att === 'csa2' || sparseGqa;
    let groups = null, full2 = full;
    if (m.kv_compress_layers) {
      groups = Object.entries(m.kv_compress_layers).map(([r, n]) => ({ r: Math.max(1, Number(r)), n: Number(n) })).filter((x) => x.n > 0);
      full2 = groups.reduce((a, x) => a + x.n, 0);                     // layers that store global KV
    } else if (sparseGqa) {
      const ns = Math.min(full, m.sparse_attention_layers);
      groups = [{ r: 1, n: ns }, ...(full > ns ? [{ r: 1, n: full - ns, dense: true }] : [])];
    } else if (sparse) groups = [{ r: 1, n: full }];
    const nAll = groups ? Math.max(1, groups.reduce((a, x) => a + x.n, 0)) : 1;
    const storeFrac = groups ? groups.reduce((a, x) => a + x.n / x.r, 0) / nAll : 1;   // stored entries per token per layer
    const isIdx = (x) => !x.dense && x.r < 64;                                           // CSA/DSA layers; HCA (÷128) and dense layers have no indexer
    const idxFrac = groups ? groups.reduce((a, x) => a + (isIdx(x) ? x.n / x.r : 0), 0) / nAll : 1;
    let elems, fullElems, kind, baseElems = 0, lat = 0;
    const globalKvh = m.global_kv_heads || G.kvh;
    const globalHd = m.global_head_dim || G.hd;
    if (G.mla) {
      lat = m.kv_latent_dim || 512;
      baseElems = m.no_rope_kv ? lat : (lat >= 576 ? lat : lat + 64); // compressed latent (+ decoupled RoPE key unless no_rope_kv)
      elems = baseElems;
      fullElems = baseElems * storeFrac;
      kind = storeFrac < 1 ? `MLA latent ${baseElems} at ${groups.map((x) => `${x.n} layers ÷${x.r}`).join(' + ')}` : m.no_rope_kv ? `MLA latent ${elems} (no RoPE key)` : `MLA latent ${elems}`;
    } else {
      elems = 2 * G.kvh * G.hd;
      fullElems = (m.attention_k_eq_v ? 1 : 2) * globalKvh * globalHd;
      kind = fullElems !== elems && swa > 0
        ? `sliding 2×${G.kvh}×${G.hd}, global ${m.attention_k_eq_v ? '1' : '2'}×${globalKvh}×${globalHd}`
        : `${m.attention_k_eq_v && !swa ? '1' : '2'} × ${G.kvh} KV heads × ${G.hd} dim`;
    }
    // Indexer keys per entry: 128 FP8 values + one FP32 scale = 132 B (DeepSeek-V3.2); stored for every compressed entry.
    const idxRaw = m.indexer_dim != null ? m.indexer_dim : (sparse ? 132 : 0);
    // GLM-5.2/5.3 run the indexer every index_topk_freq-th layer and reuse its top-k elsewhere (keys + FLOPs ÷ freq).
    const idxFreq = Math.max(1, m.index_topk_freq || 1);
    const indexer = idxRaw * idxFrac / idxFreq;
    const topk = m.index_topk || 2048, idxHeads = m.index_n_heads || 64, idxDim = m.index_head_dim || 128;
    // Per-layer recurrent state (bf16), independent of length. With published Gated-DeltaNet / Mamba-2 geometry:
    // value_heads × key_dim × value_dim, plus the short-conv state (2·key_heads·key_dim + value_heads·value_dim) × (kernel − 1).
    // Without it, fall back to attention heads × head_dim².
    let linState = 0, linKind = '';
    if (lin && m.linear_num_value_heads) {
      const vh = m.linear_num_value_heads, kh = m.linear_num_key_heads || vh, dk = m.linear_key_head_dim || 128, dv = m.linear_value_head_dim || 128, ck = m.linear_conv_kernel_dim || 4;
      linState = (vh * dk * dv + (2 * kh * dk + vh * dv) * (ck - 1)) * 2;
      linKind = `${vh} value heads × ${dk}×${dv} state + conv (${2 * kh * dk + vh * dv} ch × ${ck - 1}) × 2 B`;
    } else if (lin) { linState = G.heads * G.hd * G.hd * 2; linKind = `${G.heads} heads × ${G.hd}² × 2 B (estimated)`; }
    return { full: full2, swa, fullAttn, swaAttn, lin, window: m.sliding_window_size || 0, elems, fullElems, globalKvh, globalHd, kind, indexer, linState, linKind,
      sparse, groups, topk, idxRaw, idxHeads, idxDim, idxFreq, baseElems, lat, mla: G.mla };
  }

  /** Global-attention entries one query reads per full-attention layer at context T (decode). */
  function attendedAt(K, T) {
    if (!K.groups) return T;
    let num = 0, den = 0;
    for (const x of K.groups) { const tc = T / x.r; num += x.n * (K.sparse && !x.dense && x.r < 64 ? Math.min(K.topk, tc) : tc); den += x.n; }
    return den ? num / den : T;
  }
  /** Σ over a causal prompt of T tokens of the entries each query attends, per full-attention layer (prefill). */
  function attendedPairs(K, T) {
    if (!K.groups) return T * T / 2;
    let num = 0, den = 0;
    for (const x of K.groups) {
      const cap = K.sparse && !x.dense && x.r < 64 ? K.topk : Infinity, lim = cap * x.r;
      num += x.n * (T <= lim ? T * T / (2 * x.r) : cap * T - cap * cap * x.r / 2); den += x.n;
    }
    return den ? num / den : T * T / 2;
  }
  /** Indexer entries scored per query per full-attention layer (sparse attention only). */
  function indexedAt(K, T) {
    if (!K.sparse || !K.groups) return 0;
    let num = 0, den = 0;
    for (const x of K.groups) { num += x.n * (!x.dense && x.r < 64 ? T / x.r : 0); den += x.n; }
    return den ? num / den / (K.idxFreq || 1) : 0;
  }
  /** KV bytes for ONE sequence holding `T` tokens. */
  function kvBytesSeq(K, T, kvB) {
    const swaT = K.window ? Math.min(T, K.window) : T;
    const fEl = K.fullElems ?? K.elems;
    return K.full * T * (fEl * kvB + K.indexer) + K.swa * swaT * (K.elems * kvB + K.indexer) + K.lin * K.linState;
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
    const fEl = K.fullElems ?? K.elems;
    return K.full * fullTok * (fEl * kvB + K.indexer) + K.swa * swaTok * (K.elems * kvB + K.indexer) + seqs * K.lin * K.linState;
  }
  /** KV bytes on one replica serving `uRep` users × `batch` parallel completions each (sharing prompt within each request
   *  and `sharedPromptTok` across users). Returns { total, fullPart, swaPart, linPart } for exact per-head TP sharding. */
  function kvBytesReplica(K, promptTok, outTok, uRep, batch, sharedPromptTok, kvB) {
    const sh = uRep > 1 ? clamp(sharedPromptTok, 0, promptTok) : 0;
    const T = promptTok + outTok;
    const fullTok = sh + uRep * (promptTok - sh) + uRep * batch * outTok;
    const swaT = K.window ? Math.min(T, K.window) : T;
    const outW = K.window ? Math.min(outTok, K.window) : outTok;
    const remP = Math.max(0, swaT - outW);
    const privP = Math.min(promptTok - sh, remP);
    const shP = Math.max(0, remP - privP);
    const swaTok = shP + uRep * privP + uRep * batch * outW;
    const fEl = K.fullElems ?? K.elems;
    const fullPart = K.full * fullTok * (fEl * kvB + K.indexer);
    const swaPart = K.swa * swaTok * (K.elems * kvB + K.indexer);
    const linPart = uRep * batch * K.lin * K.linState;
    return { total: fullPart + swaPart + linPart, fullPart, swaPart, linPart };
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
    const tfl = effTflops(g, rtKey);
    const usable = usableGiB(g);
    const dBw = deviceBw(g);
    // llama.cpp/Ollama single-stream CUDA dequant kernels achieve ~68% of their consumer bwEff on datacenter HBM GPUs.
    const dcGgufBw = (rtKey === 'llamacpp' || rtKey === 'ollama') && g.category === 'datacenter' ? 0.68 : 1;
    const bwEff = R.bwEff * dBw.v * dcGgufBw;
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
    const PR = paramRates(G, inp.quant, rtKey);
    const wB = weightBytes(G, inp.quant, rtKey);
    const effBpw = PR.restRate * 8;
    T.push({ g: 'Memory', k: 'Weights', f: `${fmtP(G.P)} params: output ${fmtP(PR.outP)} × ${(PR.outRate * 8).toFixed(2)} + ${PR.inP ? `input table ${fmtP(PR.inP)} × ${(PR.inRate * 8).toFixed(2)} + ` : ''}attention ${fmtP(PR.attnP)} × ${(PR.attnRate * 8).toFixed(2)} + rest ${fmtP(PR.restP)} × ${(PR.restRate * 8).toFixed(2)} bits ÷ 8${PR.note ? ` (${PR.note})` : ''}${G.unread ? ` + ${fmtP(G.unread)} vision / lookup params at ${q.bpw >= 16 ? q.bpw : 16}-bit (loaded, not read per token)` : ''}`, v: gib(wB) });

    // ---- KV cache ----
    const K0 = kvLayout(m, G);
    // llama.cpp / Ollama allocate recurrent (Mamba / DeltaNet) states as F32 (llama-memory-recurrent: r/s caches), 2× bf16.
    const recF32 = (rtKey === 'llamacpp' || rtKey === 'ollama') && K0.lin > 0;
    const K = recF32 ? { ...K0, linState: K0.linState * 2, linKind: K0.linKind.replace('× 2 B', '× 4 B (F32 in llama.cpp)') } : K0;
    const kvPerTok = K.full * (K.fullElems * kvB + K.indexer) + K.swa * (K.elems * kvB + K.indexer);
    const kvSeq = kvBytesSeq(K, seq, kvB);
    const kvRepObj = kvBytesReplica(K, promptTok, outTok, uRep, batch, sharedTok, kvB);
    const kvRep = kvRepObj.total;
    // Exact cluster total across replicas (handles uneven user splits across DP replicas)
    const qU = Math.floor(users / replicas), rU = users % replicas;
    const kvForU = (u) => u <= 0 ? 0 : kvBytesReplica(K, promptTok, outTok, u, batch, u > 1 ? Math.round(promptTok * clamp(inp.prefixShare, 0, 1)) : 0, kvB).total;
    const kvTotal = rU * kvForU(qU + 1) + (replicas - rU) * kvForU(qU);
    T.push({ g: 'Memory', k: 'KV cache / token', f: `${K.full + K.swa} attention layers × (${K.kind}) × ${kvB} B${K.indexer ? ` + indexer ${K.indexer.toFixed(K.indexer < 10 ? 1 : 0)} B/layer` : ''}`, v: `${fmtBytes(kvPerTok)}` });
    if (K.swa) T.push({ g: 'Memory', k: 'Sliding window', f: `${K.swa} of ${K.full + K.swa} layers keep only the last ${K.window.toLocaleString()} tokens`, v: '' });
    if (K.lin) T.push({ g: 'Memory', k: 'Linear attention', f: `${K.lin} layers keep a fixed state: ${K.linKind}`, v: fmtBytes(K.lin * K.linState) + ' / seq' });
    T.push({ g: 'Memory', k: 'KV cache', f: `${fmtBytes(kvSeq)} per ${seq.toLocaleString()}-token sequence × ${uRep} user${uRep > 1 ? 's' : ''}${batch > 1 ? ` × ${batch} samples (prompt shared per request)` : ''}${replicas > 1 ? ` (${replicas} replicas)` : ''}${sharedTok ? `, ${sharedTok.toLocaleString()} shared prompt tok cached once` : ''}`, v: gib(kvTotal) });

    // ---- activations (peak during chunked prefill) + logits ----
    // A request prefills its prompt once even when generating `batch` parallel completions (n).
    const chunkTok = Math.min(promptTok, R.chunk);
    const prefillActB = chunkTok * (4 * G.h + 3 * G.ffnActive) * 2;
    const logitsActB = sRep * (G.V || 32000) * 4;
    const actRep = prefillActB + logitsActB;
    T.push({ g: 'Memory', k: 'Activations', f: `prefill chunk ${chunkTok.toLocaleString()} tok × (4·h + 3·ffn) × 2 B + ${sRep} × vocab × 4 B logits (per replica)`, v: gib(actRep) });

    // ---- speculative decoding draft ----
    let draftB = 0, draftKvRep = 0;
    const draftP = inp.spec ? Math.min(inp.specDraft, G.P / 1e9) * 1e9 : 0;
    const draftKvRatio = draftP > 0 ? Math.sqrt(draftP / G.P) : 0;
    const draftLayers = draftP > 0 ? Math.max(1, Math.round(G.L * Math.cbrt(draftP / G.P))) : 0;
    if (inp.spec) {
      draftB = draftP * effBpw / 8;
      draftKvRep = kvRep * draftKvRatio;
      T.push({ g: 'Memory', k: 'Draft model', f: `${fmtP(draftP)} × ${effBpw.toFixed(2)} bits ÷ 8 + KV ≈ target KV × √(${fmtP(draftP)} ÷ ${fmtP(G.P)}) = × ${draftKvRatio.toFixed(3)} (KV scales with layers × width, not params)`, v: gib(draftB + draftKvRep) });
    }

    // ---- offloading ----
    const offFrac = inp.offload ? clamp(inp.offloadFrac, 0, 1) : 0;
    const offW = wB * offFrac;
    const offKvFlag = Boolean(inp.offload && inp.offloadKv);
    const offKv = offKvFlag ? kvTotal : 0;

    // ---- per-GPU split ----
    // TP shards weights and KV heads (KV replicated when tp > kv heads; MLA latent is replicated on every TP rank).
    const kvShardSwa = G.mla ? 1 : Math.min(Lyt.tp, G.kvh);
    const kvShardFull = G.mla ? 1 : Math.min(Lyt.tp, K.globalKvh || G.kvh);
    const kvShard = kvShardFull;
    const wGpu = (wB - offW) / shards;
    const kvGpuFrom = (obj) => (obj.fullPart / kvShardFull + obj.swaPart / kvShardSwa + obj.linPart / Lyt.tp) / Lyt.pp;
    const kvGpu = offKvFlag ? 0 : kvGpuFrom(kvRepObj);
    const actGpu = actRep / Lyt.tp;
    const draftGpu = inp.spec ? draftB / shards + (offKvFlag ? 0 : draftKvRep / (kvShardSwa * Lyt.pp)) : 0;
    const dyn = kvGpu + actGpu;
    const fragGpu = dyn * R.frag;
    const ctxGpu = R.ctxGiB * GiB;
    const graphGpu = isApple(g) ? 0 : R.graphGiB * GiB;
    const commGpu = Lyt.n > 1 ? 0.25 * GiB : 0; // NCCL buffers
    if (Lyt.tp > Math.min(G.kvh, K.globalKvh || G.kvh) && !G.mla) W.push(`TP degree ${Lyt.tp} exceeds the model's ${Math.min(G.kvh, K.globalKvh || G.kvh)} KV heads — KV heads are replicated on layers with fewer heads.`);
    if (Lyt.tp > 1 && G.mla) W.push('MLA latent KV is replicated on every tensor-parallel rank (it cannot be split by head).');

    const segsGpu = [
      ['weights', G.moe ? 'Model weights (all experts)' : 'Model weights', wGpu],
      ['kv', 'KV cache', kvGpu],
      ['act', 'Activations & logits', actGpu],
      ...(inp.spec ? [['draft', 'Draft model', draftGpu]] : []),
      ['ctx', `${g.vendor === 'amd' ? 'ROCm' : isApple(g) ? 'Metal' : 'CUDA'} context & workspaces`, ctxGpu],
      ...(graphGpu ? [['graph', g.vendor === 'amd' ? 'HIP graphs' : 'CUDA graphs', graphGpu]] : []),
      ...(commGpu ? [['comm', 'Collective buffers', commGpu]] : []),
      ['frag', 'Allocator fragmentation', fragGpu],
    ];
    T.push({ g: 'Memory', k: 'Runtime overhead', f: `${R.label}: context ${R.ctxGiB} GiB${graphGpu ? ` + graphs ${R.graphGiB} GiB` : ''} + ${Math.round(R.frag * 100)}% fragmentation of KV+activations${commGpu ? ' + 0.25 GiB NCCL' : ''} (per GPU)`, v: gib(ctxGpu + graphGpu + fragGpu + commGpu) });

    const perGpu = segsGpu.reduce((a, s) => a + s[2], 0);
    // Cluster total: every replica holds the weights, but each holds only its own users' KV (uneven splits handled);
    // GPUs left over when n is not a multiple of TP × PP are idle and not counted.
    const nonKvGpu = perGpu - kvGpu - fragGpu;
    const repGpuBytes = (u) => {
      const kv = offKvFlag || u <= 0 ? 0 : kvGpuFrom(kvBytesReplica(K, promptTok, outTok, u, batch, u > 1 ? Math.round(promptTok * clamp(inp.prefixShare, 0, 1)) : 0, kvB));
      return shards * (nonKvGpu + kv + R.frag * (kv + actGpu));
    };
    const totalUse = rU * repGpuBytes(qU + 1) + (replicas - rU) * repGpuBytes(qU);
    // Serving engines cap their own pool: vLLM gpu_memory_utilization and SGLang mem_fraction_static default to 0.90 of device
    // memory (raise the flag to use more); TensorRT-LLM gives the KV cache 90% of the memory left after weights/activations.
    let cap = usable.v * GiB, capNote = `${usable.v.toFixed(1)} GiB usable (${usable.note})`;
    if (rtKey === 'vllm' || rtKey === 'sglang') {
      cap = 0.9 * usable.v * GiB;
      capNote = `${rtKey === 'vllm' ? 'vLLM gpu_memory_utilization' : 'SGLang mem_fraction_static'} 0.90 × ${usable.v.toFixed(1)} GiB (${usable.note})`;
    } else if (rtKey === 'trtllm') {
      cap = usable.v * GiB - 0.1 * Math.max(0, usable.v * GiB - (perGpu - kvGpu));
      capNote = `TensorRT-LLM: KV may use 90% of the ${gib(Math.max(0, usable.v * GiB - (perGpu - kvGpu)))} left after weights & buffers`;
    }
    const pct = (perGpu / cap) * 100;
    T.push({ g: 'Memory', k: 'Per-GPU usage', f: `sum of the rows above ÷ ${gib(cap)} budget — ${capNote}`, v: `${gib(perGpu)} (${pct.toFixed(1)}%)` });

    // ---- decode step time ----
    const avgOut = outTok / 2;
    const avgCtx = promptTok + avgOut;
    // Weights streamed per step: output head + attention + the rest; the untied input table is only gathered (one row
    // per token), never read in full. MoE: the shared bucket holds output, input table and attention, so
    // shared bytes = outP·outRate + attnP·attnRate + (shared − outP − inP − attnP)·restRate.
    const inEmb = PR.inP * PR.inRate;
    const sharedRestP = Math.max(0, G.shared - PR.outP - PR.inP - PR.attnP);
    const sharedB = PR.outP * PR.outRate + PR.attnP * PR.attnRate + sharedRestP * PR.restRate;
    const MOE_KERNEL_EFF = 0.75; // fused-MoE GEMVs reach ~75% of dense-GEMV bandwidth
    const expertSlice = G.moe ? (m.moe_ffn_intermediate_size || G.ffn) / Lyt.tp : G.ffn;
    const sliceEffBase = G.moe && Lyt.tp > 1 && expertSlice < 1024 ? clamp(0.32 + 0.68 * (expertSlice / 1024), 0.32, 1) : 1;

    // Pipeline parallelism micro-batching: with sRep > 1, up to min(PP, sRep) micro-batches circulate across PP stages.
    const ppM = Lyt.pp > 1 ? Math.min(Lyt.pp, sRep) : 1;
    const sMicro = sRep / ppM;
    const uMicro = uRep / ppM;
    const ppEff = ppM > 1 ? 0.92 : 1;

    const moeEffFor = (nTok) => MOE_KERNEL_EFF * (sliceEffBase + (1 - sliceEffBase) * Math.min(1, Math.max(0, nTok - 1) / 31));
    const moeEff = moeEffFor(sMicro);
    const touchFrac = G.moe ? 1 - Math.pow(1 - G.f, sMicro) : 1;
    const routedB = G.routed * effBpw / 8 * touchFrac;
    const wRead = G.moe ? sharedB + routedB : Math.max(0, wB - ub - inEmb);
    const wReadEff = G.moe ? sharedB + routedB / moeEff : wRead; // bandwidth-equivalent bytes

    // KV bytes read per micro-batch decode step. Sparse attention (DSA / CSA): the indexer scans every stored key, the main
    // attention gathers only the top-k entries (compressed HCA layers are read densely). Recurrent states are read AND
    // written back every step.
    const kvReadFor = (uCount, nBatch) => {
      const shAvg = uCount > 1 ? Math.min(sharedTok, promptTok) : 0;
      const obj = kvBytesReplica(K, promptTok, avgOut, uCount, nBatch, shAvg, kvB);
      const linPart = obj.linPart * 2;
      if (K.groups) {
        const fullTok = shAvg + uCount * (promptTok - shAvg) + uCount * nBatch * avgOut;
        const stored = fullTok * (K.fullElems / Math.max(1e-9, K.baseElems || K.fullElems));   // stored entries (sharing-aware)
        const main = Math.min(attendedAt(K, avgCtx) * uCount * nBatch, stored);
        const fullPart = K.full * (main * (K.baseElems || K.fullElems) * kvB + fullTok * K.indexer);
        return { total: fullPart + obj.swaPart + linPart, fullPart, swaPart: obj.swaPart, linPart };
      }
      return { total: obj.fullPart + obj.swaPart + linPart, fullPart: obj.fullPart, swaPart: obj.swaPart, linPart };
    };
    const kvReadObj = kvReadFor(uMicro, batch);
    const kvRead = kvReadObj.total;
    const kvReadGpu = offKvFlag ? 0 : (kvReadObj.fullPart / kvShardFull + kvReadObj.swaPart / kvShardSwa + kvReadObj.linPart / Lyt.tp);
    const gpuW = wReadEff * (1 - offFrac);
    // Per-stage time (divided by Lyt.pp; when Lyt.pp = 1 this is the full model step).
    const tMemStage = (gpuW / (Lyt.tp * Lyt.pp) + kvReadGpu / Lyt.pp) / (bw * bwEff);
    // FLOPs per decoded token: matmuls 2 × active params (untied input lookup excluded) + attention over the cache.
    // Decode attention per head per entry: standard 4·d (QKᵀ + PV); MLA runs absorbed in decode: 2·(latent+rope) + 2·latent.
    // Full-attention layers use the global head size when it differs (Gemma 4). The sparse indexer scores every entry
    // (2 · index_heads · index_dim FLOPs, run in FP8 on FP8-capable GPUs).
    const hfFullDec = K.mla ? 2 * K.baseElems + 2 * K.lat : 4 * (m.global_head_dim || G.hd);
    const linFlopsTok = 2 * G.Amm;
    const attnFlopsDec = (T) => G.heads * (K.fullAttn * attendedAt(K, T) * hfFullDec + K.swaAttn * Math.min(T, K.window || T) * 4 * G.hd);
    const idxFlopsDec = (T) => K.fullAttn * indexedAt(K, T) * 2 * K.idxHeads * K.idxDim;
    const flopsTok = linFlopsTok + attnFlopsDec(avgCtx) + idxFlopsDec(avgCtx);
    const lowP = lowPrecisionX(q, g, rtKey);
    const idxX = supportsFp8(g) ? 2 : 1;
    // Low-precision tensor cores (W8A8 FP8 / W4A4 FP4) speed up the weight matmuls only; attention runs in BF16.
    const effFlopsTok = linFlopsTok / lowP.x + attnFlopsDec(avgCtx) + idxFlopsDec(avgCtx) / idxX;
    // TP > 1 incurs small-GEMM slicing / sync efficiency loss (~2% per doubling of TP).
    const tpCompEff = Lyt.tp > 1 ? 1 / (1 + 0.04 * Math.log2(Lyt.tp)) : 1;
    const tCompStage = (effFlopsTok * sMicro / Lyt.pp) / (tfl.v * 1e12 * Lyt.tp * tpCompEff * 0.5);
    // Per-layer launch/dispatch overhead: MoE launches router+top-k+permute+grouped-GEMM; multi-GPU MoE + MLA add dispatch/combine sync.
    // Calibrated at batch 1: MLA-MoE TP8 (DeepSeek-V3/R1 8×H200: vLLM 15.9 ms, SGLang 21.0 ms per token) and GQA-MoE TP8
    // (gpt-oss-120b TP8/EP1 TensorRT-LLM 8×B200: 420 tok/s ⇒ ≈58 µs per MoE layer incl. 2 all-reduces).
    const moeMult = G.moe ? (Lyt.tp > 1 ? (G.mla ? 36 : 10) : 2.5) : 1;
    // llama.cpp / Ollama on datacenter GPUs: ≈34 µs per layer per token (scoreboard #15013, Llama-2-7B Q4_0: H100 268 and
    // A100 191 tok/s both imply +0.78 ms per token over the bandwidth time at 32 layers).
    const ggufDc = (rtKey === 'llamacpp' || rtKey === 'ollama') && g.category === 'datacenter';
    const layerUsEff = (ggufDc ? 34 : R.layerUs) * moeMult;
    const tFixedStage = (R.stepMs / Lyt.pp + (G.L / Lyt.pp) * layerUsEff / 1000) / 1000;
    const tpLink0 = LINKS[inp.cluster && Lyt.tp > Lyt.perNode ? Lyt.inter : Lyt.intra] || LINKS.pcie4;
    // Point-to-point meshes (AMD xGMI): a GPU reaches its TP peers over (TP − 1) separate links, not the whole fabric.
    const tpLink = tpLink0.perPeer ? { ...tpLink0, bw: Math.min(tpLink0.bw, tpLink0.perPeer * Math.max(1, Lyt.tp - 1)) } : tpLink0;
    const ppLink0 = LINKS[inp.cluster && Lyt.tp * Lyt.pp > Lyt.perNode ? Lyt.inter : Lyt.intra] || LINKS.pcie4;
    const ppLink = ppLink0.perPeer ? { ...ppLink0, bw: ppLink0.perPeer } : ppLink0;   // a PP hop is one point-to-point transfer
    // NCCL AllReduce achieves ~70% of unidirectional link peak on multi-GPU collectives.
    const tpCommStageFor = (tokensPerRank, layersInStage) => {
      if (Lyt.tp <= 1) return 0;
      const msg = tokensPerRank * G.h * 2;
      const allReducePerLayer = G.moe ? 2 : 2;
      return allReducePerLayer * layersInStage * (tpLink.lat * 1e-6 + 2 * (Lyt.tp - 1) / Lyt.tp * msg / (tpLink.bw * 0.70 * 1e9));
    };
    const ppHopFor = (tokensPerRank) => Lyt.pp > 1 ? (ppLink.lat * 1e-6 + tokensPerRank * G.h * 2 / (ppLink.bw * 1e9)) : 0;
    const tCommStage = tpCommStageFor(sMicro, G.L / Lyt.pp) + (Lyt.pp > 1 ? ((Lyt.pp - 1) / Lyt.pp) * ppHopFor(sMicro) : 0);
    const offTarget = OFFLOAD[inp.offloadTarget] || OFFLOAD.cpu_ram;
    // llama.cpp/Ollama run offloaded layers on the CPU at system-RAM speed; other runtimes stream them over PCIe.
    const cpuRun = (rtKey === 'llamacpp' || rtKey === 'ollama') && inp.offloadTarget === 'cpu_ram';
    const offBw = inp.offloadTarget === 'nvme' ? offTarget.bw : cpuRun ? offTarget.bw * 0.7 : isApple(g) ? offTarget.bw : PCIE_HOST_BW;
    let tOff = 0;
    if (offFrac > 0) {
      tOff = wRead * offFrac / (offBw * 1e9 * (cpuRun ? 1 : Lyt.tp));
      T.push({ g: 'Speed', k: 'Offloaded layers', f: `${fmtBytes(wRead * offFrac)} read per token from ${offTarget.label} at ${offBw} GB/s${cpuRun ? ' (CPU executes these layers)' : ' (streamed to the GPU)'}`, v: ms(tOff) });
    }
    if (offKvFlag) tOff += kvRead / (PCIE_HOST_BW * 1e9);
    const tStage = Math.max(tMemStage, tCompStage) + tFixedStage + tCommStage + tOff / Lyt.pp;
    // Full per-sequence decode step time: traverses Lyt.pp stages (with ppEff steady-state pipeline efficiency when ppM > 1).
    const tMem = tMemStage * Lyt.pp;
    const tComp = tCompStage * Lyt.pp;
    const tFixed = tFixedStage * Lyt.pp;
    const tComm = tCommStage * Lyt.pp;
    const tStepDecode = (tStage * Lyt.pp) / ppEff;
    T.push({ g: 'Speed', k: 'Bytes read per step', f: `${G.moe ? `weights: shared ${fmtBytes(sharedB)} + routed ${fmtBytes(G.routed * effBpw / 8)} × (1 − (1 − ${G.f.toFixed(3)})^${sMicro.toFixed(sMicro % 1 ? 1 : 0)}) [MoE kernels at ${Math.round(moeEff * 100)}% eff.] = ` : 'weights '}${fmtBytes(wRead)} + KV at avg context ${Math.round(avgCtx).toLocaleString()} × ${sMicro.toFixed(sMicro % 1 ? 1 : 0)} seq ${fmtBytes(kvRead)}`, v: fmtBytes(wRead + kvRead) });
    T.push({ g: 'Speed', k: 'Memory time', f: `(weights ÷ ${Lyt.tp} + KV ÷ ${kvShard}) ÷ (${g.bandwidth_gbs} GB/s × ${bwEff.toFixed(2)} efficiency (${R.label} ${R.bwEff}${dBw.v < 1 ? ` × device ${dBw.v}` : ''}${dcGgufBw < 1 ? ` × DC ${dcGgufBw}` : ''}))`, v: ms(tMem) });
    T.push({ g: 'Speed', k: 'Compute time', f: `(${(linFlopsTok / 1e9).toFixed(1)} GFLOP matmul${lowP.x > 1 ? ` ÷ ${lowP.x} (${lowP.label})` : ''} + ${((attnFlopsDec(avgCtx) + idxFlopsDec(avgCtx)) / 1e9).toFixed(2)} GFLOP attention${K.sparse ? ' + indexer' : ''}) × ${sMicro.toFixed(sMicro % 1 ? 1 : 0)} ÷ (${tfl.v.toFixed(1)} TFLOPS × 0.5)  — ${tfl.note}`, v: ms(tComp) });
    T.push({ g: 'Speed', k: 'Fixed overhead', f: `${R.stepMs} ms/step + ${G.L} layers × ${layerUsEff.toFixed(0)} µs${tComm ? ` + interconnect ${ms(tComm)}` : ''}${ppM > 1 ? ` (${ppM} pipelined stages)` : ''}`, v: ms(tFixed + tComm) });

    // ---- prefill / TTFT (prompt is prefilled once per request even when batch n > 1) ----
    const pfTok = promptTok;
    const fp8x = lowP.x;
    // llama.cpp/Ollama prefill uses FP16-acc/MMQ on consumer GeForce (2× halved FP32-acc rate), NOT on datacenter GPUs.
    const ggufPath = q.gguf && (rtKey === 'llamacpp' || rtKey === 'ollama');
    const consumerNv = g.vendor === 'nvidia' && (isGeForceFp16Acc(g) || /^4090/.test(g.key));
    const int8x = g.int8_tensor === false ? 1 : (consumerNv ? 2 : 1);
    // Matmul speed-up in prefill: INT8 MMQ (llama.cpp on GeForce) or FP8/FP4 tensor cores; attention stays BF16.
    const pfLinX = ggufPath && !isApple(g) ? int8x : fp8x;
    const pfTfl = tfl.v * pfLinX;
    // llama.cpp MMQ prefill on datacenter GPUs is limited by integer dequant on quantized weight matmuls (≈ 3.86 · TFLOPS^0.514;
    // scoreboard pp512: H100 9918 vs A100 4850 tok/s = 2.04× for 3.17× the TFLOPS), while FP16 FlashAttention runs at R.pfEff.
    const pfAttnEff = isApple(g) ? 0.68 : R.pfEff;
    const pfLinEff = isApple(g) ? 0.68 : (ggufPath && g.category === 'datacenter' ? Math.min(R.pfEff, 3.86 * Math.pow(tfl.v * pfLinX, -0.486)) : R.pfEff);
    // Prefill attention runs un-absorbed (MLA: 4 × mean(qk 192, v 128) = 4 × 160 per head per pair).
    const hfFullPf = 4 * (K.mla ? G.hd : (m.global_head_dim || G.hd));
    const swaPairsOf = (t) => K.window && t > K.window ? K.window * (t - K.window / 2) : t * t / 2;
    const idxPairsOf = (t) => (K.sparse && K.groups ? K.fullAttn * (K.groups.reduce((a, x) => a + x.n * (!x.dense && x.r < 64 ? t * t / (2 * x.r) : 0), 0) / K.groups.reduce((a, x) => a + x.n, 0)) / (K.idxFreq || 1) : 0);
    const pfAttnFlops = (t) => G.heads * (K.fullAttn * attendedPairs(K, t) * hfFullPf + K.swaAttn * swaPairsOf(t) * 4 * G.hd);
    const pfIdxFlops = (t) => idxPairsOf(t) * 2 * K.idxHeads * K.idxDim;
    const prefillTimeFor = (toks, pipelined) => {
      if (toks <= 0) return 0;
      const pfCompAll = (2 * G.Amm * toks / (pfLinX * pfLinEff) + (pfAttnFlops(toks) + pfIdxFlops(toks) / idxX) / pfAttnEff) / (tfl.v * 1e12 * Lyt.tp * tpCompEff);
      const cTok = Math.min(R.chunk, toks);
      const nChunks = Math.ceil(toks / cTok);
      const pfW = Math.max(0, wB - ub);
      const pfMemAll = nChunks * (pfW * (1 - offFrac)) / Lyt.tp / (bw * bwEff);
      const pfOffAll = offFrac > 0 ? nChunks * pfW * offFrac / (offBw * 1e9 * (cpuRun ? 1 : Lyt.tp)) : 0;
      const pfCommAll = nChunks * (tpCommStageFor(cTok, G.L) + (Lyt.pp > 1 ? (Lyt.pp - 1) * ppHopFor(cTok) : 0));
      const pfFixedAll = nChunks * (G.L * R.layerUs * 1e-6) + pfCommAll;
      const seqTime = Math.max(pfCompAll, pfMemAll) + pfOffAll + pfFixedAll;
      if (Lyt.pp <= 1) return seqTime;
      // Under PP: a single request can pipeline its chunks across stages ((nChunks + PP - 1) / (nChunks * PP));
      // steady-state multi-user prefill pipelines different prompts across all PP stages (1 / (PP * 0.92)).
      if (pipelined) return seqTime / (Lyt.pp * 0.92);
      return seqTime * (nChunks + Lyt.pp - 1) / (nChunks * Lyt.pp);
    };
    const pfFlops = 2 * G.Amm * pfTok + pfAttnFlops(pfTok) + pfIdxFlops(pfTok);
    const pfChunkTok = Math.min(R.chunk, pfTok);
    const chunks = Math.ceil(pfTok / pfChunkTok);
    const tPrefill = prefillTimeFor(pfTok, false);

    // Continuous batching with multiple users: steady-state decode steps share GPU time with incoming uncached prompt prefills.
    const uncachedPrompt = Math.max(0, promptTok - sharedTok);
    const tPrefillAmort = uRep > 1 && uncachedPrompt > 0
      ? (uRep / outTok) * prefillTimeFor(uncachedPrompt, true)
      : 0;
    const tStep = tStepDecode + tPrefillAmort;
    T.push({ g: 'Speed', k: 'Decode step', f: `max(memory, compute) + fixed + interconnect + offload${tPrefillAmort ? ` + amortized prefill ${ms(tPrefillAmort)}` : ''}`, v: ms(tStep) });

    // speculative decoding
    let perSeqTps = 1 / tStep, specInfo = null;
    if (inp.spec) {
      const a = clamp(inp.specAccept, 0.01, 0.999), k = inp.specTokens;
      const E = (1 - Math.pow(a, k + 1)) / (1 - a);
      const draftKvRead = kvReadGpu * draftKvRatio;
      // Draft step: its weights + KV over the same bandwidth, plus its own per-layer launch cost (≈ L·(P_d/P)^⅓ layers).
      const tDraft = (draftB / (Lyt.tp * Lyt.pp) + draftKvRead / Lyt.pp) / (bw * bwEff) * Lyt.pp / ppEff + R.stepMs / 2000 + draftLayers * R.layerUs * 1e-6;
      // Verifying k+1 tokens activates more MoE experts per step and sends (k+1)× larger TP messages.
      const vTok = sMicro * (k + 1);
      const vTouch = G.moe ? 1 - Math.pow(1 - G.f, vTok) : 1;
      const vRoutedB = G.routed * effBpw / 8 * vTouch;
      const vWRead = G.moe ? sharedB + vRoutedB : wRead;
      const vWReadEff = G.moe ? sharedB + vRoutedB / moeEffFor(vTok) : wRead;
      const vTMemStage = (vWReadEff * (1 - offFrac) / (Lyt.tp * Lyt.pp) + kvReadGpu / Lyt.pp) / (bw * bwEff);
      const vTOff = (offFrac > 0 ? vWRead * offFrac / (offBw * 1e9 * (cpuRun ? 1 : Lyt.tp)) : 0) + (offKvFlag ? kvRead / (PCIE_HOST_BW * 1e9) : 0);
      const vTCommStage = tpCommStageFor(vTok, G.L / Lyt.pp) + (Lyt.pp > 1 ? ((Lyt.pp - 1) / Lyt.pp) * ppHopFor(vTok) : 0);
      const tVerify = (Math.max(vTMemStage, tCompStage * (k + 1)) + tFixedStage + vTCommStage + vTOff / Lyt.pp) * Lyt.pp / ppEff + tPrefillAmort;
      const tCycle = (inp.specType === 'parallel' ? tDraft : k * tDraft) + tVerify;
      perSeqTps = E / tCycle;
      specInfo = { E, tDraft, tVerify, tCycle, speedup: perSeqTps * tStep };
      T.push({ g: 'Speed', k: 'Speculative decoding', f: `E[tokens/cycle] = (1 − α^(k+1)) / (1 − α) = ${E.toFixed(2)}; cycle = ${inp.specType === 'parallel' ? '1' : k} × draft ${ms(tDraft)} + verify ${ms(tVerify)}`, v: `${specInfo.speedup.toFixed(2)}× speed-up` });
    }
    const throughput = perSeqTps * S;

    const ttft = tPrefill + tStepDecode;
    T.push({ g: 'Speed', k: 'Prefill FLOPs', f: `2 × ${fmtP(G.Amm)} ${G.moe ? 'active ' : ''}matmul params (input lookup excluded) × ${pfTok.toLocaleString()} prompt tokens + causal attention${K.sparse ? ' (top-' + K.topk + ' sparse) + indexer' : ''}`, v: `${(pfFlops / 1e12).toFixed(2)} TFLOP` });
    T.push({ g: 'Speed', k: 'Time to first token', f: `max(FLOPs ÷ (${pfTfl.toFixed(1)} TFLOPS${ggufPath && !isApple(g) ? (int8x > 1 ? ' (INT8 MMQ)' : ' (MMQ, no INT8 boost)') : fp8x > 1 ? ` (${lowP.label})` : ''} × ${pfLinEff.toFixed(2)}${pfLinEff !== pfAttnEff ? ` matmul / ${pfAttnEff.toFixed(2)} attn` : ''}${Lyt.tp > 1 ? ` × ${Lyt.tp}` : ''}), ${chunks} chunk weight reads)${offFrac > 0 ? ' + offload streaming' : ''} + first decode step`, v: ms(ttft) });
    const ttftQueued = uRep > 1 ? prefillTimeFor(pfTok, true) * uRep + tStepDecode : ttft;

    // ---- power ----
    const uMem = Math.min(1, (tMem / tStepDecode) * bwEff), uComp = Math.min(1, (tComp / tStepDecode) * 0.5);
    const watts = g.tdp_watts * Math.min(1, 0.2 + 0.5 * uMem + 0.8 * uComp) * Lyt.n;
    T.push({ g: 'Power', k: 'Power draw', f: `${Lyt.n} × TDP ${g.tdp_watts} W × (0.2 idle + 0.5 × memory-busy ${uMem.toFixed(2)} + 0.8 × compute-busy ${uComp.toFixed(2)})`, v: `${Math.round(watts)} W` });

    if (pct > 100 && !inp.offload) W.push('The configuration does not fit. Lower the precision, shorten the context, add GPUs, or enable offloading.');
    if (seq > (m.context_length || Infinity)) W.push(`Sequence length exceeds the model's ${m.context_length.toLocaleString()}-token context window.`);
    if (q.gguf && !['llamacpp', 'ollama'].includes(rtKey)) W.push(`${q.label} is a GGUF format — normally served by llama.cpp or Ollama.`);
    if (rtKey === 'mlx' && !isApple(g)) W.push('MLX only runs on Apple Silicon.');
    if ((KV_SUPPORT[rtKey] || []).length && !KV_SUPPORT[rtKey].includes(inp.kvQuant)) W.push(`${R.label} has no ${KV_LABELS[inp.kvQuant] || inp.kvQuant} KV cache; it supports ${KV_SUPPORT[rtKey].map((k) => KV_LABELS[k].split(' (')[0]).join(', ')}.`);
    if (inp.kvQuant === 'nvfp4' && rtKey === 'trtllm' && !isBlackwell(g)) W.push('TensorRT-LLM NVFP4 KV cache needs a Blackwell GPU.');
    if ((WEIGHT_UNSUPPORTED[rtKey] || []).includes(inp.quant)) W.push(`${R.label} cannot load ${q.label} checkpoints${rtKey === 'llamacpp' || rtKey === 'ollama' ? ' — it loads GGUF files (Q8_0 ≈ INT8, Q4_K_M ≈ INT4)' : rtKey === 'mlx' ? ' — use MLX 4-bit / 8-bit' : ''}.`);
    if (q.fp8 && !supportsFp8(g)) W.push(`${g.label} has no FP8 tensor cores: FP8 weights save memory but compute runs at FP16 speed (weight-only).`);
    if (inp.quant === 'nvfp4' && !isBlackwell(g)) W.push('NVFP4 is accelerated only on Blackwell; elsewhere it is dequantized to FP16 on the fly.');
    if (inp.quant === 'mxfp4' && !supportsMxfp4(g)) W.push('MXFP4 is accelerated only on Blackwell and AMD MI350X/MI355X; elsewhere it is dequantized on the fly.');

    const status = statusFor(pct);
    const fits = pct <= 100;
    const kvOneUser = kvBytesReplica(K, promptTok, outTok, 1, batch, 0, kvB).total;
    const kvTwoUsers = kvBytesReplica(K, promptTok, outTok, 2, batch, sharedTok, kvB).total;
    const perUserKv = Math.max(0, kvTwoUsers - kvOneUser);
    return {
      mode: 'inference', runtime: rtKey, runtimeLabel: R.label, layout: Lyt, status,
      memory: {
        perGpu: perGpu / GiB, total: totalUse / GiB, capPerGpu: cap / GiB, capTotal: cap / GiB * Lyt.n, rawPerGpu: g.memory, pct,
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
      // FP8 training (Transformer Engine / Megatron) keeps BF16 parameters plus an FP8 weight cache (3 B), not FP8-only weights.
      wBpp = { fp32: 4, fp16: 2, fp8: 3 }[inp.ftQuant] ?? 2;
      trainable = P; wLabel = `${fmtP(P)} × ${wBpp} B (${inp.ftQuant === 'fp8' ? 'BF16 params + FP8 weight cache' : inp.ftQuant.toUpperCase()})`;
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
    // QLoRA keeps embeddings & lm_head in BF16 (2 B/param) while quantizing linear layers to NF4.
    const qloraEmb = method === 'qlora' ? Math.min(G.embedParams, P) : 0;
    const weights = (P - qloraEmb) * wBpp + qloraEmb * 2 + G.unread * 2;             // vision tower / lookup tables stay frozen in BF16
    if (qloraEmb) wLabel += ` (embeddings ${fmtP(qloraEmb)} kept in BF16)`;
    if (G.unread) wLabel += ` + ${fmtP(G.unread)} frozen vision / lookup params × 2 B`;
    T.push({ g: 'Memory', k: 'Model weights', f: wLabel, v: gib(weights) });
    if (method !== 'full') T.push({ g: 'Memory', k: 'LoRA parameters', f: `r ${inp.loraRank} × Σ(d_in + d_out) over q,k,v,o,gate,up,down × ${G.L} layers`, v: fmtP(trainable) });

    // ---- gradients, optimizer, master weights ----
    const pureBf16 = !!o.use_bf16_true && (method !== 'full' || inp.ftQuant !== 'fp32');
    const gradB = method === 'full' ? (inp.ftQuant === 'fp32' ? 4 : 2) : 4;         // BF16 grads (FP32 for FP32 training); LoRA adapters train in fp32
    const masterB = method === 'full' ? (inp.ftQuant !== 'fp32' && !pureBf16 ? 4 : 0) : 0; // fp32 master copy in mixed precision
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
    // CPU offloading works on a single GPU (DeepSpeed ZeRO-Offload / PyTorch CPU offload) as well as multi-GPU ZeRO.
    const offOpt = Boolean(o.offload_optimizer && (N === 1 || zero >= 1));
    const offGrad = Boolean(o.offload_gradients && (N === 1 || zero >= 2));
    const offParam = Boolean(o.offload_param && (N === 1 || zero >= 3));
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
    // Fused / chunked cross-entropy (Liger / Unsloth) avoids materializing the full [tokens × vocab] fp32 logits tensor.
    const logitsTok = o.fused_cross_entropy ? Math.min(tokens, 256) : tokens;
    // HF loss path: bf16 logits (2 B) + fp32 upcast (4 B) + fp32 gradient (4 B) = 10 B per element; fused/chunked CE
    // (Liger, Unsloth, cut-cross-entropy) keeps only a small chunk of fp32 logits + gradient alive.
    const logitsBpe = o.fused_cross_entropy ? 8 : 10;
    const logits = logitsTok * (G.V || 32000) * logitsBpe;
    if (o.activation_offloading) act = act * 0.15;
    T.push({ g: 'Memory', k: 'Activations', f: `${tokens.toLocaleString()} tokens × (7h + 2·kv + 4·ffn) × 2 B × ${fullLayers.toFixed(0)} layers${ck ? ` + checkpoints h × 2 B × ${Math.round(G.L * ck)} layers` : ''}${o.activation_offloading ? ' × 0.15 (offloaded)' : ''}`, v: gib(act) });
    T.push({ g: 'Memory', k: 'Logits', f: `${logitsTok.toLocaleString()}${o.fused_cross_entropy ? ' (chunked fused CE)' : ''} × vocab ${(G.V || 32000).toLocaleString()} × ${logitsBpe} B (${o.fused_cross_entropy ? 'fp32 chunk + grad' : 'bf16 logits + fp32 upcast + fp32 grad'})`, v: gib(logits) });

    const ctx = (isApple(g) ? 0.3 : 0.6) * GiB, frag = 0.06 * (act + logits);
    // ZeRO-3 gathers full unsharded layers (~2 layers prefetch buffer), independent of N.
    const zeroBuf = zero >= 3 ? 2 * (P / Math.max(1, G.L)) * wBpp : zero >= 1 ? Math.min(0.5 * GiB, trainable * 2 / N * 0.1) : 0;
    const comm = N > 1 ? 0.25 * GiB + zeroBuf : 0;
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
    const linF = 2 * G.Amm, attnF = 4 * attended * G.heads * G.hd;
    const flopsFwd = linF + attnF;
    const mult = method === 'full' ? 3 : 2;                                          // fwd + bwd (LoRA skips weight grads)
    const recompute = ck * 1;                                                        // one extra forward for checkpointed layers
    // Attention scores have no frozen weights: their backward is always 2× forward, so they cost 3× even under LoRA.
    const flopsTok = linF * (mult + recompute) + attnF * (3 + recompute);
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
    // gradient all-reduce / ZeRO collectives once per optimizer step (overlap 70%)
    let commS = 0;
    if (N > 1) {
      const link = LINKS[inp.cluster ? inp.interNode : inp.interconnect] || LINKS.pcie4;
      const bwBps = link.bw * 1e9;
      const gatherB = method === 'full' ? Math.min(wBpp, 2) : 4;                    // BF16 params (FP8 cache rebuilt locally); fp32 adapters
      const rs = (N - 1) / N;                                                         // ring reduce-scatter / all-gather volume per element
      if (zero === 0) {
        // DDP: gradients all-reduced once per optimizer step (accumulation is local)
        commS = 0.3 * 2 * rs * trainable * gradB / bwBps;
      } else if (zero === 1) {
        // ZeRO-1: one gradient reduce-scatter at the accumulation boundary + one updated-param all-gather
        commS = 0.3 * rs * trainable * (gradB + gatherB) / bwBps;
      } else if (zero === 2) {
        // ZeRO-2: gradients are partitioned during backward, so they are reduce-scattered on every micro-batch
        commS = 0.3 * rs * trainable * (inp.gradSteps * gradB + gatherB) / bwBps;
      } else {
        // ZeRO-3: per micro-batch, gradient reduce-scatter + (2 + ck) param all-gathers (forward, backward, recompute)
        commS = 0.3 * rs * inp.gradSteps * (trainable * gradB + (2 + ck) * P * gatherB) / bwBps;
      }
      tps = tokensPerStep / (tokensPerStep / tps + commS);
    }
    const offloadPenalty = (offOpt ? 0.85 : 1) * (offGrad ? 0.90 : 1) * (offParam ? 0.6 : 1) * (o.activation_offloading ? 0.9 : 1);
    tps *= offloadPenalty;
    T.push({ g: 'Speed', k: 'FLOPs / token', f: mult === 3
      ? `(2 × ${fmtP(G.Amm)} + attention 4 × ${KL.full + KL.swa} layers × avg attended × d) × 3${recompute ? ` + ${ck.toFixed(2)} recompute` : ''}`
      : `2 × ${fmtP(G.Amm)} × 2 (no frozen-weight grads) + attention 4 × ${KL.full + KL.swa} layers × avg attended × d × 3${recompute ? `, each + ${ck.toFixed(2)} recompute` : ''}`, v: `${(flopsTok / 1e9).toFixed(1)} GFLOP` });
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
  // Achieved fraction of dense BF16 TFLOPS for eager diffusers DiTs: ≈0.72–0.75 on Ada (L20 / RTX 4090 FLUX), ≈0.31–0.35 on H100
  // (FLUX 28 steps 6.75 s; Wan2.1 720p×81: 14B 1875 s, 1.3B 410 s). p = 1 keeps achieved TFLOPS rising with peak TFLOPS.
  const DIFF_EFF = { a: 0.98, t0: 510, p: 1 };
  function diffusion(inp, m, g) {
    const T = [], W = [];
    const q = QUANTS[inp.quant] || QUANTS.fp16;
    const P = num(m.num_of_params) * 1e9, A = num(m.num_of_expert_params) * 1e9 || P, aux = num(m.num_of_auxiliary_params) * 1e9;
    const h = m.hidden_dim_size || 3072, L = m.num_of_layers || 28;
    const isUnet = /stable-diffusion-(1|2)|sdxl|sd-2|kandinsky|svd/.test(m.slug);
    const video = m.task_type === 'video_generation';
    const frames = video ? inp.numFrames : 1;
    const latFrames = video ? 1 + Math.floor((frames - 1) / 4) : 1;                // 4× temporal VAE compression
    const spatialDs = m.spatial_downsample || (isUnet ? 32 : 16);                  // VAE downsample × patch size (e.g. 16×2=32 for Wan 2.2 TI2V-5B)
    const tokens = latFrames * Math.ceil(inp.imageWidth / spatialDs) * Math.ceil(inp.imageHeight / spatialDs);
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
    // Classifier-free guidance: the SD3 / SDXL pipelines batch the conditional and unconditional passes (2× activations);
    // the Wan and Qwen-Image pipelines run them one after the other (same FLOPs, 1× activations).
    const cfgSequential = /wan|qwen-image/.test(m.slug);
    const cfgAct = cfgSequential ? 1 : cfg;
    const textTok = m.text_tokens || 0;                                             // MMDiT text tokens join the image tokens in attention
    const Ntok = tokens + textTok;
    const actB = b * cfgAct * Ntok * h * 2 * 12;
    T.push({ g: 'Memory', k: 'Latent activations', f: `batch ${b} × CFG ${cfgAct}${cfgSequential && cfg > 1 ? ' (passes run sequentially)' : ''} × ${Math.round(Ntok).toLocaleString()} tokens (${Math.round(tokens).toLocaleString()} latent at ${spatialDs}× spatial${textTok ? ` + ${textTok} text` : ''}) × h ${h} × 2 B × 12 live tensors`, v: gib(actB) });
    // Causal 3D video VAEs decode 1 temporal chunk (4 frames) at a time.
    const vaeFrames = video ? Math.min(frames, 4) : 1;
    const px = inp.imageWidth * inp.imageHeight * vaeFrames;
    const vaeB = (inp.vaeTiling ? Math.min(px, 512 * 512) : px) * 2560;
    const vaeW = (video ? 0.5 : 0.2) * GiB;                                        // VAE weights (FLUX VAE 84M, Wan VAE 127M params)
    T.push({ g: 'Memory', k: 'VAE decode workspace', f: `${inp.vaeTiling ? 'tile 512×512' : `${inp.imageWidth}×${inp.imageHeight}${video ? ` × ${vaeFrames} frames` : ''}`} px × 2.5 KiB (runs after the denoiser)`, v: gib(vaeB) });
    const ctx = (isApple(g) ? 0.3 : 0.6) * GiB;
    const shard = inp.cluster ? Math.min(Lyt.tp, Lyt.n) : Math.max(1, Lyt.n);
    const actGpu = actB / Math.max(1, Lyt.n);
    // Peak = max over the two phases. With model CPU offload (diffusers enable_model_cpu_offload) only the running model is
    // on the GPU, so the denoiser is gone before the VAE decodes; without it everything stays resident.
    const offloadModels = Boolean(inp.encoderOffload);
    const phaseDenoise = wB / shard + encGpu / shard + actGpu + (offloadModels ? 0 : vaeW);
    const phaseVae = (offloadModels ? 0 : wB / shard + encGpu / shard) + vaeB + vaeW;
    const vaePeak = phaseVae > phaseDenoise;
    const segs = vaePeak
      ? [...(offloadModels ? [] : [['weights', 'Denoiser weights', wB / shard], ...(encGpu ? [['enc', 'Text encoders', encGpu / shard]] : [])]), ['vae', 'Peak workspace (VAE decode + VAE weights)', vaeB + vaeW]]
      : [['weights', 'Denoiser weights', wB / shard], ...(encGpu ? [['enc', 'Text encoders', encGpu / shard]] : []), ['act', `Peak workspace (latent activations${offloadModels ? '' : ' + VAE weights'})`, actGpu + (offloadModels ? 0 : vaeW)]];
    const peakWork = vaePeak ? vaeB : actGpu;
    segs.push(['ctx', 'Runtime context', ctx], ['frag', 'Allocator fragmentation', 0.05 * peakWork]);
    const perGpu = segs.reduce((a, s) => a + s[2], 0);
    const pct = perGpu / (usable.v * GiB) * 100;
    T.push({ g: 'Memory', k: 'Peak phase', f: `${offloadModels ? 'model CPU offload: only the running model is resident — ' : ''}denoise ${gib(phaseDenoise)} vs VAE decode ${gib(phaseVae)}`, v: vaePeak ? 'VAE decode' : 'denoising' });

    // FLOPs per step: only the params that act on every token (adaLN modulation, text-stream weights excluded) × all tokens
    // in the joint attention (image latents + text for MMDiT), plus attention 4·L·N²·h.
    const imgRatio = m.image_param_ratio ?? 1;
    const linearFlops = 2 * A * imgRatio * Ntok * cfg * b;
    const attnFlops = 4 * L * Ntok * Ntok * h * cfg * b;
    const flopsStep = linearFlops + attnFlops;
    // Low-precision speed-up measured end to end (NVIDIA TensorRT FLUX on RTX 5090: FP16 10.93 s → FP8 6.68 s (1.64×) →
    // FP4 3.85 s (2.84×)); spec-sheet tensor ratios (2×, 8×) overstate it because attention, norms and the VAE stay 16-bit.
    const dLow = (q.fp8 && supportsFp8(g)) ? { x: 1.64, label: 'FP8' } : ((inp.quant === 'nvfp4' && isBlackwell(g)) || (inp.quant === 'mxfp4' && supportsMxfp4(g))) ? { x: 2.84, label: 'FP4' } : { x: 1, label: '' };
    // UNet denoisers (SDXL) run many small conv / attention kernels: ≈0.55× the DiT efficiency (SDXL A100 30 steps ≈ 3.3 s).
    const eff = (DIFF_EFF.a / (1 + Math.pow(tfl.v / DIFF_EFF.t0, DIFF_EFF.p))) * (isUnet ? 0.55 : 1) * (/tensorrt/.test(inp.runtime) ? 1.35 : 1);
    const gpuScale = Math.max(1, Lyt.n === 1 ? 1 : Lyt.n * 0.85);
    const tStep = (linearFlops + attnFlops) / dLow.x / (tfl.v * 1e12 * eff * gpuScale);
    // VAE decode: analytic conv FLOPs per output pixel — 2D decoder (FLUX/SD) ≈ 9.3 MFLOP; causal 3D video decoder ≈ 19 MFLOP
    // per pixel-frame (3×3×3 kernels, narrower channels). Calibrated: Wan 720p × 81 frames ≈ 5–7 s.
    const vaeFlopsPx = video ? 1.9e7 : 9.3e6;
    const tVae = (inp.imageWidth * inp.imageHeight * frames) * vaeFlopsPx * b / (tfl.v * 1e12 * 0.25);
    // Text encoding: one forward of the encoder(s) over the prompt, plus moving them to the GPU when model offload is on.
    const encTok = Math.max(77, textTok || 512);
    const tEnc = (offloadModels ? encB / (PCIE_HOST_BW * 1e9) : 0) + 2 * aux * encTok / (tfl.v * 1e12 * 0.3) + 0.02;
    const total = inp.inferenceSteps * tStep + tVae + tEnc;
    T.push({ g: 'Speed', k: 'FLOPs per step', f: `(2 × ${fmtP(A)}${imgRatio < 1 ? ` × ${imgRatio} per-token` : ''} × ${Math.round(Ntok).toLocaleString()} tokens + 4·${L}·tokens²·${h}) × CFG ${cfg} × batch ${b}`, v: `${(flopsStep / 1e12).toFixed(1)} TFLOP` });
    T.push({ g: 'Speed', k: 'Step time', f: `FLOPs ÷ (${tfl.v.toFixed(1)} TFLOPS${dLow.x > 1 ? ` × ${dLow.x} measured ${dLow.label} speed-up` : ''} × ${eff.toFixed(2)} achieved)`, v: ms(tStep) });
    T.push({ g: 'Speed', k: 'Generation time', f: `${inp.inferenceSteps} steps × step + VAE decode ${ms(tVae)} (${(vaeFlopsPx / 1e6).toFixed(1)} MFLOP/px) + text encode ${ms(tEnc)}${offloadModels ? ' incl. moving encoders to the GPU' : ''}`, v: `${total.toFixed(1)} s` });
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
  function sanitize(raw) {
    const numNodes = Math.max(1, Math.round(num(raw.numNodes, 1)));
    const gpusPerNode = Math.max(1, Math.round(num(raw.gpusPerNode, 1)));
    const numGpus = Math.max(1, Math.round(num(raw.numGpus, 1)));
    const tpDegree = Math.max(1, Math.round(num(raw.tpDegree, 1)));
    const ppDegree = Math.max(1, Math.round(num(raw.ppDegree, 1)));
    const seq = Math.max(2, Math.round(num(raw.seq, 4096)));
    const outputTokens = clamp(Math.round(num(raw.outputTokens, 512)), 1, seq - 1);
    const batch = Math.max(1, Math.round(num(raw.batch, 1)));
    const users = Math.max(1, Math.round(num(raw.users, 1)));
    const gradSteps = Math.max(1, Math.round(num(raw.gradSteps, 1)));
    const numSamples = Math.max(1, Math.round(num(raw.numSamples, 10000)));
    const tokensPerSample = Math.max(1, Math.round(num(raw.tokensPerSample, 512)));
    const numEpochs = Math.max(0.01, num(raw.numEpochs, 1));
    const loraRank = Math.max(1, Math.round(num(raw.loraRank, 16)));
    const imageWidth = Math.max(64, Math.round(num(raw.imageWidth, 1024)));
    const imageHeight = Math.max(64, Math.round(num(raw.imageHeight, 1024)));
    const numFrames = Math.max(1, Math.round(num(raw.numFrames, 1)));
    const inferenceSteps = Math.max(1, Math.round(num(raw.inferenceSteps, 28)));
    const guidanceScale = Math.max(0, num(raw.guidanceScale, 3.5));
    const specTokens = Math.max(1, Math.round(num(raw.specTokens, 4)));
    const specDraft = Math.max(0.01, num(raw.specDraft, 1));
    const elec = Math.max(0, num(raw.elec, 0.16));
    const carbon = Math.max(0, num(raw.carbon, 400));
    return {
      ...raw,
      numNodes, gpusPerNode, numGpus, tpDegree, ppDegree, seq, outputTokens, batch, users,
      gradSteps, numSamples, tokensPerSample, numEpochs, loraRank,
      imageWidth, imageHeight, numFrames, inferenceSteps, guidanceScale,
      specTokens, specDraft, elec, carbon,
      opt: raw.opt || {},
    };
  }

  function calculate(rawInp, model, gpu) {
    const inp = sanitize(rawInp || {});
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


  const api = { calculate, geometry, weightBytes, unreadBytes, paramRates, attendedAt, attendedPairs, indexedAt, KV_SUPPORT, kvLayout, kvBytesSeq, kvBytesGroup, kvBytesReplica, effTflops, deviceBw, usableGiB, autoRuntime, supportsFp8, supportsMxfp4, lowPrecisionX, QUANTS, KV_BYTES, KV_LABELS, RUNTIMES, LINKS, OFFLOAD, GiB };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.GpuCalc = api;
})(typeof window !== "undefined" ? window : globalThis);
