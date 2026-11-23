# How the calculator compares with real measurements

Run with `node tests/real-data/compare.js` (add `--rows` for every row, or `--json out.json` for machine output).
Error is ours ÷ measured − 1. "Bias" is the median signed error; "|err|" is the median absolute error.
Sources are described in `SOURCES.md`. Run date: 2026-11-23.

## Scorecard (before → after the calibration fixes in this commit)

| # | Comparison | n | Bias before → after | \|err\| before → after |
|---|---|---:|---:|---:|
| D2 | KV bytes per token per GPU vs engine logs (standard attention) | 94 | 0.0% → 0.0% | 0.0% → 0.0% (94/94 within ±10%) |
| D1 | Weight memory per GPU vs engine logs | 201 | −3.0% → −3.0% | 3.6% → 3.6% |
| F2 | Diffusion denoiser peak memory (diffusers, L40S) | 3 | −0.7% → −0.7% | 0.7% → 0.7% |
| B3 | Fits vs out-of-memory verdicts (XiongjieDai, 1–8 GPUs) | 132 | 132/132 agree | 132/132 agree |
| A3 | Apple decode, llama.cpp F16/Q8_0/Q4_0 | 37 | −28.0% → **−1.3%** | 38.9% → **4.3%** |
| A4 | Apple prefill, llama.cpp (incl. M5) | 37 | −4.8% → −3.4% | 6.4% → **5.0%** |
| A1 | llama.cpp tg128, CUDA + ROCm scoreboards | 45 | −7.3% → −7.2% | 9.4% → **7.9%** |
| A2 | llama.cpp pp512, scoreboards, flash attention on (default) | 45 | −12.3% → **−2.6%** | 34.8% → **12.4%** |
| A2b | same, flash attention off | 44 | −0.4% → +8.4% | 24.6% → 14.1% |
| B1 | llama.cpp tg1024 Llama 3 8B/70B, 1–8 GPUs | 96 | −7.0% → **+2.0%** | 15.7% → **7.8%** |
| B2 | llama.cpp pp1024 (May 2024, before FA default) | 91 | +9.3% → +22.5% | 26.6% → 38.1% (expected: old no-FA builds) |
| D3 | KV pool size at the engine's memory fraction | 78 | +9.7% → **+5.6%** | 9.7% → **5.6%** |
| G1 | Mistral Large 4 KV pool, 8× GB300 (vLLM blog) | 2 | +14.7% → +12.3% | — |
| D4 | Serving TPOT, InferenceX | 2649 | +0.8% → −6.2% | 26.9% → **20.0%** |
| D6 | Serving output tok/s per GPU, InferenceX | 2649 | +4.5% → +12.0% | 27.1% → **20.5%** |
| E1/E2 | torchtitan Llama 3.1 8B FSDP memory / tok/s | 2 / 2 | +8.6% / +10.5% (unchanged) | — |
| F1 | Diffusion seconds per step (L40S) | 3 | −28.1% (unchanged) | 39.1% |
| D5 | Serving TTFT, InferenceX | 2649 | −64.6% → −65.8% | open (see 12) |
| C1 | Argonne vLLM batch latency (2024 software) | 1213 | −57.8% (unchanged) | vintage gap (see 13) |

Per model, InferenceX TPOT bias before → after:
- DeepSeek-R1: +71% → +10%.
- Kimi-K2.5: +52% → −6%.
- GLM-5: +32% → −13%.
- DeepSeek-V4-Pro on NVIDIA: +5..+59% → −5% (median error 10%).
- Unchanged: Llama-70B −23%, gpt-oss −1%, MiniMax +1..+3%, Qwen3.5 +8%.

## Fixes applied (each has a hand-calculated test in `tests/hand.js` section I)

1. **Apple llama.cpp decode** (finding 1):
   - 82% of bandwidth, or 70% on the two-die Ultra.
   - Plus a per-layer cost by weight format: F16 20 µs, Q8_0 45 µs, 4–6-bit 80 µs.
2. **llama.cpp F16 on NVIDIA non-HBM cards:** bandwidth ×1.13 (finding 1).
3. **The datacenter llama.cpp penalty (0.68× bandwidth, 34 µs/layer, dequant-limited prefill) now applies only to HBM parts.** L40/L40S/L20/L4/A40/A10G/A16/A2 use the workstation path, and L40 gets the INT8 MMQ doubling (finding 2).
4. **AMD Instinct llama.cpp:** 86 µs per layer (finding 3).
5. **GeForce llama.cpp prefill:** ×1.35 for flash attention, the default since `-fa auto` (finding 4).
6. **M5 GPU neural accelerators now also serve llama.cpp and Ollama Metal**, not only MLX (finding 5).
7. **Usable memory:** B300 267.7 GiB (was 288), B200 178.4 GiB (was 180), from vLLM's "Free memory on device" (finding 7).
8. **Mistral Large 4 has 44 indexer layers** (`indexer_layer_count`), so KV is 78,384 B/token (finding 8).
9. **MoE per-layer multiplier under TP:**
   - MLA: 36 → 12.
   - Compressed-KV MLA (DeepSeek-V4): 20.
   - GQA MoE: unchanged at 10 (InferenceX confirms 432 tok/s for gpt-oss TRT TP8) (finding 10).

Not changed, because the mechanism isn't settled:
- **Dense TP overhead (finding 9).** The extra cost per layer fits all-reduce latency for Llama-70B, but that would break gpt-oss (50 µs per layer total at TP8 B200 vLLM). That suggests a bandwidth-efficiency loss for small sharded GEMMs instead.
- **FP4 decode efficiency (finding 9).**
- **The vLLM activation reserve (finding 7).**
- **TTFT overhead (finding 12).** It varies by framework: vLLM +44 ms, SGLang +83 ms, TensorRT-LLM +168 ms at concurrency 1, and it grows with prompt length.
- **L40S diffusion clocks (finding 11).** Only one GPU in the data.
- **AMD MI300X/MI325X serving.** Ours is 20–50% too fast; it looks like ROCm software maturity.
- **The `fp8_ds_mla` 656 B/token layout.** SGLang GLM-5 measures 576 + 132 B, and vLLM's ML4 layout isn't logged.

## Findings, each checked by hand (as measured before the fixes)

**1. Apple decode is 20–35% too slow, and the gap depends on the weight format.**
- Hand calculation, M4 Max (546 GB/s), Llama-2-7B, bytes read per token × measured tok/s:
  - F16: 13.22 GB × 31.64 = 418 GB/s, which is 77% of peak.
  - Q8_0: 7.02 GB × 54.05 = 379 GB/s, which is 69%.
  - Q4_0: 3.76 GB × 83.06 = 312 GB/s, which is 57%.
- Ours applies a flat 0.74 × 0.62 = 46% to every format, plus 0.67 ms per step.
- Result: F16 comes out −26..−39% on every chip and Q8_0 −20..−37%.
- Q4_0 matches only on M2/M3 Ultra and M4 Max (−5..+1%). It is −11..−27% on other Max chips and −20..−32% on base and Pro chips.
- CUDA shows the same pattern with a smaller gap:
  - 4090 F16 8B: 16.06 GB × 54.34 = 873 GB/s (87%). Ours is −12%.
  - 4090 Q4_K_M 8B: within +5%.

**2. The "datacenter llama.cpp" penalty is applied to GDDR cards. It should apply only to HBM cards.**
- Our datacenter path applies 0.68× bandwidth and 34 µs per layer.
- L40 (864 GB/s): 3.79 GB × 153.8 tok/s = 583 GB/s, which is 67%. That is the consumer factor (0.74), not 0.50.
  - The consumer path predicts 151.5 tok/s (−1.5%). Ours gives 98.5 (−36%).
- A40: 478 GB/s (69%). The consumer path gives −1%; ours gives −35%.
- XiongjieDai: L40S 8B F16 is −36% and L40S/A40 70B F16 ×4 is −37..−38%.
- HBM cards fit the penalty: H100 −3%, A100 −3%. A30 (HBM2) sits in between at 54% of bandwidth.

**3. llama.cpp on AMD Instinct has about 2.5× the per-layer overhead of CUDA.**
- MI300X tg128: 218.5 tok/s, i.e. 4.58 ms per token.
  - Memory time at 0.50 efficiency: 3.79 GB / (5300 × 0.503) = 1.42 ms.
  - Overhead: (4.58 − 1.42 − 0.35) / 32 layers = 88 µs per layer.
- MI210 gives the same: 7.68 ms per token, 4.60 ms of memory time, 85 µs per layer.
- Ours uses the CUDA value of 34 µs, so MI300X comes out +50% and MI210 +15%.
- RDNA consumer cards are not affected (7900 XTX −18%, 7800 XT −2%).

**4. GeForce prefill with flash attention on (now the llama.cpp default) is 25–35% too slow.**
- 4090 pp512: 14,771 tok/s × 13.36 GFLOP per token = 197 TFLOPS achieved.
- Ours: 330 INT8 TFLOPS × 0.42 = 139 TFLOPS.
- With flash attention off, the same scoreboard gives bias −0.4% (A2b), so the current calibration matches the old no-flash-attention numbers.
- Workstation cards are close: RTX 6000 Ada +8%, A6000 −14%.
- L40 (−59%) also gets the datacenter integer-dequant formula (3.86·T^−0.486). Measured 10,098 tok/s is 135 TFLOPS, near the GeForce Ada path.
- RDNA4 (9070, 9060 XT, R9700) and older datacenter cards (T4, A30, MI210) are 25–120% too fast. This looks like immature ROCm/MMQ kernels, and the community runs use mixed commits.

**5. M5 prefill is about 3.8× too slow: the data needs the M5 GPU neural accelerators.**
- M5 Max pp512 is 3,220 tok/s against 886 on the M4 Max (3.6×). Our `fp16_tflops` makes the two chips equal (33.2 vs 34.1).
- 3,220 × 13.36 GFLOP = 43 TFLOPS achieved, already above the listed 33.2 peak.

**6. Weight-memory outliers (D1).**
- MiniMax-M3 FP8 on MI300X/MI325X: measured 2× ours (101.3 vs 50.1 GiB at TP8). The same model on NVIDIA and MI355X matches within 3%. vLLM on CDNA3 loads this checkpoint in BF16, so this is not our error.
- Kimi-K2.5 INT4 is −13..−17%. Measured 140.56 GiB/GPU at TP4 is 604 GB in total, the size of the native checkpoint, which uses INT4 groups of 32 (4.5 bits per weight). Our `int4` is AWQ g128 at 4.156 bits per weight.
- gpt-oss-120b in vLLM is −6..−33%. TP1 H200 loads 66.14 GiB against a 60.8 GiB checkpoint (ours 60.68 matches the checkpoint). vLLM pads the MXFP4 expert layouts.

**7. The KV pool is 7–15% too large. Three causes, all measured in the logs.**
- **B300 usable memory.** vLLM reports 267.69 GiB total (InferenceX `gpu_specs` also says 268); ours assumes 288.
  - B300 rows are +14.9% against +6.6..+8% elsewhere.
  - Kimi-K2.5 INT4 TP4 at 0.95: 0.95 × 20.3 GiB = 19.3 GiB of KV that doesn't exist, about 295k tokens.
  - Smaller gaps elsewhere:
    - H200: 140.4 GiB real vs 141.
    - B200: 178.35 vs 180.
    - H100: 79.6 vs 80.
    - RTX PRO 6000 Server: 95.0 vs 96.
- **Activation reserve.** MiniMax-M2.5 H200 TP4 at 0.95:
  - Budget 0.95 × 140.4 = 133.4 GiB, minus weights 53.75, CUDA graphs 0.60 and KV 73.13, leaves 5.9 GiB of profiled activations. vLLM profiles a `max_num_batched_tokens` = 32,768 dummy batch.
  - Ours reserves 1.75 GiB in total.
  - gpt-oss TP1: real 1.93 GiB vs ours 2.02.
- **Weight-format differences** (finding 6).

**8. Mistral Large 4 on 8× GB300 (blog: 3.60M tokens at 0.85, FP8 KV).**
- Real layout from the blog:
  - 63 layers store a latent in vLLM's `fp8_ds_mla` format, 656 B each (512 FP8 + 16 B scales + 128 B BF16 RoPE).
  - Only 44 layers have an indexer, at 132 B each.
  - That's 63 × 656 + 44 × 132 = 47,136 B per token. Ours is 63 × (576 + 132) = 44,604 B (−5.4%).
- Budget: 0.85 × 278 GiB = 236.3 GiB, minus 66.8 GiB of weights and about 4 GiB of other memory, leaves about 165.5 GiB. 165.5 GiB / 47,136 B = 3.77M tokens, close to the reported 3.60M.
- Ours: 4.23M with the 288 GB entry, 4.03M at 278 GB. Correcting the bytes per token as well gives 3.81M (+6%).

**9. Dense models get too fast as tensor parallelism grows.**
- Bias by TP size: TP1 −13%, TP2 −11%, TP4 −25%, TP8 −39%.
- Hand fit, Llama-3.3-70B FP8 on H200 vLLM at concurrency 4. The bandwidth part is 71.6 GB / (4.8 TB/s × 0.72) / TP; the rest is overhead per layer:
  - TP1: 21.47 ms measured, 20.7 ms bandwidth, 0.77 ms overhead, about 10 µs per layer.
  - TP2: 13.91 ms measured, 10.36 ms bandwidth, 3.55 ms overhead, 44 µs per layer.
  - TP4: 9.85 ms measured, 5.18 ms bandwidth, 4.67 ms overhead, 58 µs per layer.
  - TP8: 7.93 ms measured, 2.59 ms bandwidth, 5.34 ms overhead, 67 µs per layer.
  - Ours: 2 all-reduces × (10 µs + tiny transfer) + 6 µs, about 27 µs per layer.
  - Real small-message all-reduces cost about 20–30 µs each.
- B200 achieves only about 54–56% of 8 TB/s in FP8 decode (TP1: 72.7 GB / 16.34 ms = 4.45 TB/s). H200 reaches 69–72%.
- NVFP4 decode is only 1.25× faster than FP8, though it reads 1.7× fewer bytes: (68.5B × 0.5625 + 4.2 GB of BF16 embeddings) = 42.7 GB / 13.07 ms = 3.27 TB/s, which is 41%. Ours treats FP4 as purely byte-proportional, so FP4 rows are −30..−60%.

**10. MLA MoE models (DeepSeek-R1, Kimi-K2.5, GLM-5) are 45–77% too slow.**
- Our per-layer multiplier for MLA MoE under TP is 36 × 6 µs = 216 µs per layer. It was calibrated on early-2025 vLLM/SGLang (15.9 / 21.0 ms per token on 8× H200).
- Current measurements:
  - H200 SGLang FP8 TP8 at concurrency 4: 9.70 ms. Bytes: (17B shared + 656B × 12% of routed experts touched) / 8 GPUs = 11.9 GB per GPU, which takes 3.1 ms. Overhead: 6.6 ms / 61 layers = 108 µs per layer.
  - B200 SGLang FP4 TP4 at concurrency 1: 5.64 ms, about 79 µs per layer.
- Other MoE models (gpt-oss, MiniMax, Qwen3.5) are close: TP2–8 between −4% and +13%. AMD MI300X/MI325X runs −21..−31%.

**11. Diffusion step time is 21–36% too fast on L40S.**
- Flux at 1024² with 512 text tokens: our FLOPs are 7.43e13, matching diffusers' own count (74,395 GFLOP).
- 7.43e13 / 0.498 s = 149 TFLOPS, which is 41% of the L40S's 362. Our curve gives 57%.
- Likely cause: the 350 W L40S can't sustain its peak clock. This is the only diffusion GPU in the data, so the fix should come from more GPUs.

**12. TTFT is missing a fixed serving overhead and queueing.**
- At concurrency 1 with 1k prompts, the median measured TTFT is 2.4× ours, a constant +47 ms (HTTP, tokenizing, scheduling, first detokenize).
- At higher concurrency, requests wait behind other prompts. For example, gpt-oss H200 TP1 with 8k prompts at concurrency 4: 349 ms measured, about 2 prefills of 170 ms each; ours gives 178 ms for one.
- The prefill compute itself matches. gpt-oss 8k: 9.4e13 FLOP / (989 TF × 0.55) = 0.17 s, and ours is 178 ms.

**13. The Argonne vLLM data reflects 2024 software.**
- H100 Llama-3-8B FP16, batch 1: 126 tok/s × 16.06 GB = 2.02 TB/s, which is 60% of 3.35 TB/s. Ours uses 80%, so ours is −21..−22% at every length.
- The shape across batch size and length is right (−21..−36% everywhere on H100).
- MI300X 2024 ROCm vLLM is about 4× slower than ours (−75%).
- Useful as a scaling check, not as a level check.

## Problems in the source data (not counted)

- **XiongjieDai prompt processing:** five F16 cells are below 60% of the same card's Q4_K_M:
  - A100 SXM: 681 tok/s, vs 7,504 on the PCIe card.
  - L40S: 2,492.
  - RTX 5000 Ada ×4: 2,878.
  - L40S ×4 and A100 SXM ×4.
- **Argonne TensorRT-LLM:** `run.py` stops at end-of-sequence (`end_id`, no `ignore_eos`). For example, Qwen2-7B with "2,048 output tokens" takes 0.08 s. Not used.
- **InferenceX:**
  - A gpt-oss H200 TP4 concurrency-16 run has a 4.7 s mean TTFT where its neighbours are about 50 ms (a glitch). Median TTFT is used.
  - Early DeepSeek-V4-Pro runs on MI355X take 1.46 s per token.
  - For hybrid, sparse or compressed attention (gpt-oss sliding window, Qwen3.5 linear attention, MiniMax-M3 sparse, DeepSeek-V4, GLM-5.2/5.3), the engines' "GPU KV cache size" counts tokens differently. For example, vLLM counts every gpt-oss layer as full attention: 57.25 GiB / 825,734 = 74,448 B per token, the cost of all 36 layers. These rows are reported separately (D2h/D3h).
