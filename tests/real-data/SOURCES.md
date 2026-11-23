# Real-world measurements to compare against

Measured data from reputable GitHub repos (plus their Hugging Face datasets) that can be checked against this calculator.
Collected 2026-11-22. Files below were downloaded as-is; nothing has been compared yet.

Trust levels:
- **A**: vendor or maintainer-run, reproducible, versioned (MLPerf, TensorRT-LLM, torchtitan, DeepSeek, vLLM/SGLang team posts).
- **B**: curated community scoreboards with the exact command, commit and hardware recorded (llama.cpp threads).
- **C**: single-author benchmark repos; useful but older or less controlled.

## 1. Decode and prefill speed, single GPU (llama.cpp)

| Source | Trust | What's measured | Coverage | Saved as |
|---|---|---|---|---|
| [llama.cpp #15013: CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013) | B | `llama-bench` pp512 and tg128 tok/s, Llama-2-7B Q4_0, with and without flash attention | 56 NVIDIA GPUs (RTX 20–50, PRO 6000, A100, H100, …) | `llamacpp/cuda_scoreboard_15013.csv` |
| [llama.cpp #15021: ROCm scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15021) | B | same test | ~25 AMD GPUs (MI300X, MI210, 7900 XTX, 9070, …) | `llamacpp/rocm_scoreboard_15021.csv` |
| [llama.cpp #10879: Vulkan scoreboard](https://github.com/ggml-org/llama.cpp/discussions/10879) | B | same test | ~130 GPUs across NVIDIA, AMD, Intel, iGPUs | `llamacpp/vulkan_scoreboard_10879.csv` |
| [llama.cpp #4167: Apple Silicon](https://github.com/ggml-org/llama.cpp/discussions/4167) | B | PP and TG at F16, Q8_0 and Q4_0, LLaMA-7B, with bandwidth and GPU cores | M1–M4 (base, Pro, Max, Ultra), plus M2 Ultra history 2023–2025 | `llamacpp/apple_scoreboard_4167.csv` |
| [llama.cpp #16578: DGX Spark](https://github.com/ggml-org/llama.cpp/discussions/16578) | B | pp and tg at several context depths and batch sizes, gpt-oss and other models | GB10 | not saved (results are in code blocks; scrape later) |
| [XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) | C | llama.cpp decode and prompt tok/s, Llama 3 8B and 70B at Q4_K_M and F16, multi-GPU | 31 GPU configs (3070 to H100 PCIe ×4) plus 4 Macs; OOM entries usefully test "does it fit" | `llamacpp/xiongjiedai_gpu_benchmarks_README.md` |
| [mlc-ai/llm-perf-bench](https://github.com/mlc-ai/llm-perf-bench) | C | decode tok/s: MLC vs ExLlamaV2 vs llama.cpp vs vLLM, Llama-2 7B/13B/70B, CodeLlama-34B, TP 1–8 | 3090 Ti, 4090, A100 ×2/4/8, A10G, 7900 XTX (2023) | `llamacpp/mlc_llm_perf_bench_README.md` |
| [turboderp/exllamav2](https://github.com/turboderp/exllamav2) README | C | decode tok/s for GPTQ and EXL2 models | 3090 Ti, 4090 | not saved |

## 2. Serving throughput and latency (vLLM, SGLang, TensorRT-LLM)

| Source | Trust | What's measured | Coverage | Saved as |
|---|---|---|---|---|
| [argonne-lcf/LLM-Inference-Bench](https://github.com/argonne-lcf/LLM-Inference-Bench) ([paper](https://arxiv.org/abs/2411.00136)) | A | latency and throughput, every combination of input/output length (128–2048) and batch size | **4,771 rows**: A100, H100, GH200, MI250, MI300X, Gaudi2, Intel PVC, SN40L × vLLM, llama.cpp, TRT-LLM, DeepSpeed-MII × Llama-2/3 7–70B, Mistral-7B, Mixtral-8x7B, Qwen2 7B/72B, 1–8 GPUs | `serving/argonne_llm_inference_bench_all_results.csv` |
| [TensorRT-LLM perf-overview.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/perf-overview.md) | A | max-load output tok/s per GPU by input/output length | DeepSeek-R1, gpt-oss 20B/120B, Llama 3.3 70B, Llama 4 Maverick, Qwen3 235B and 30B on H100, H200, B200, GB200, RTX PRO 6000 Server | `serving/tensorrt_llm_perf_overview.md` |
| [HF optimum llm-perf leaderboard](https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard) ([code](https://github.com/huggingface/optimum-benchmark)) | A | **measured memory** (max allocated, reserved, process VRAM) plus prefill and decode latency and throughput, HF Transformers | 844 configs on 1×A100; also A10, T4 and CPU files, with AWQ, GPTQ, bnb and torchao variants | `serving/hf_optimum_llm_perf_pytorch_unquantized_1xA100.csv` (other GPUs not downloaded yet) |
| [vLLM blog: Mistral Large 4](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-10-29-mistral-large-4.md) | A | real architecture (DSA: 44 indexer layers, 28 reuse top-k; 9 layers share layer 35's KV; layer 0 dense); **KV capacity 3.60M tokens on 8×GB300 TP8 at 0.85**; 1.97M on TP4; TTFT and TPOT, speculative decoding, P/D | GB300, plus AMD and TPU sections | `serving/vllm_blog_mistral_large_4.md` |
| [vLLM blog (176 posts)](https://github.com/vllm-project/vllm-project.github.io/tree/main/_posts) | A | posts with measured tables, e.g. MiniMax-H3 production serving, Kimi K3/K3.1, Qwen3.8 P/D, DeepSeek-V4.1 wide-EP, FP8 KV cache, KV offloading, Intel Arc Pro B | many | only the Mistral Large 4 post saved |
| [LMSYS / SGLang blog](https://github.com/lm-sys/lm-sys.github.io/tree/main/blog) | A | e.g. DeepSeek on 96 H100 (52.3k input / 22.3k output tok/s per node), GB200 NVL72 parts I and II, Kimi K2 on 128 H200, DGX Spark, gpt-oss | H100, H200, GB200, DGX Spark | not saved |
| [deepseek-ai/open-infra-index, day 6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | A | production: ~73.7k input tok/s prefill or ~14.8k output tok/s decode **per H800 node**, 20–22 tok/s per user, mean KV length 4,989, 56.3% cache hits | H800 production fleet | `serving/deepseek_v3r1_inference_system_day6.md` |
| [ROCm/rocm-blogs](https://github.com/ROCm/rocm-blogs) | A | AMD-measured vLLM and SGLang numbers, e.g. GLM-5.3 on MI300X, GLM-5.2 MXFP4 on MI355X, speculative decoding, multi-node inference, Flux on MI300 | MI300X, MI325X, MI355X | not saved |
| [SemiAnalysis InferenceX](https://github.com/SemiAnalysisAI/InferenceX) ([dashboard](https://inferencex.semianalysis.com), [app/API](https://github.com/SemiAnalysisAI/InferenceX-app)) | A | nightly throughput-vs-interactivity sweeps over TP and concurrency, plus power | **8,444 rows** (latest snapshot, pulled 2026-11-22 from the public API): 17 models (Llama-3.3-70B FP8, gpt-oss-120b, DeepSeek-R1/V4/V4.1, Qwen-3.5-397B, Kimi-K2.5/K3/K3.1, GLM-5/5.2/5.3, MiniMax-M2.5/M3/M3.1, …) × H100, H200, B200, B300, GB200, GB300, MI300X, MI325X, MI355X, RTX PRO 6000, VR200, TPU v7 × vLLM, SGLang, TRT-LLM, Dynamo, ATOM; ISL/OSL 1k/1k, 1k/8k, 8k/1k and agentic. Each row has TP/EP/DP, concurrency, TTFT, TPOT, ITL, throughput per GPU; **2,606 rows report the real `kv_cache_pool_tokens`**, about 4,000 report measured power | `inferencex/inferencex_benchmarks_all.csv` (156 metric columns), plus `availability.json`, `gpu_specs.csv` (their usable memory, bandwidth and TFLOPS per chip), `model_architectures.json` and `openapi.json`. `server_log_memory.json` holds weights / KV memory / KV tokens / memory fraction parsed from 281 server logs (`fetch_logs.sh` + `parse_logs.js`). Re-pull rows with `GET /api/v1/benchmarks?model=<name>`; `/api/v1/server-log-search` can grep each run's server log (for example for vLLM's "GPU KV cache size" line). The in-repo `experimental/` JSONs are flagged WIP by the authors and were not used. |
| [vLLM performance dashboard](https://docs.vllm.ai/en/stable/benchmarking/dashboard/) | A | nightly latency, throughput and serving tests on H100 and others | — | not downloaded (Buildkite artifacts and PyTorch HUD) |
| [kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | C | CPU+GPU hybrid DeepSeek-R1/V3, e.g. 227.85 prefill / 87.58 decode tok/s on 8×L20 + Xeon | offloading cases | not saved |
| [LLMServingSim references](https://github.com/casys-kaist/LLMServingSim) | A | real vLLM per-request timings | RTX PRO 6000 Server, RTX 4090 | already in `tests/llmservingsim/references/` |

## 3. Datacenter benchmarks (MLPerf Inference)

| Source | Trust | What's measured | Coverage | Saved as |
|---|---|---|---|---|
| [mlcommons/inference_results_v5.1](https://github.com/mlcommons/inference_results_v5.1) | A (audited) | Offline, Server and Interactive tok/s per system; GPU count, software stack, precision | 1,448 results (416 LLM): Llama-2-70B, Llama-3.1-8B and 405B, Mixtral-8x7B, DeepSeek-R1, SDXL on B200, GB200, GB300, H200, H100, MI300X, MI325X, MI355X, RTX PRO 6000 | `mlperf/inference_v5.1_summary_results.json` |
| [mlcommons/inference_results_v6.0](https://github.com/mlcommons/inference_results_v6.0) | A (audited) | same, adding gpt-oss-120b, Qwen3-VL-235B, Wan-2.2 T2V | 520 results, 28 accelerators | `mlperf/inference_v6.0_summary_results.json` |
| inference_results_v4.1, v5.0, v6.1 | A | older/newer rounds (v6.1 has no `summary_results.json` at the root) | — | not saved |

Note: MLPerf numbers are tuned peak throughput under latency limits. Compare them with the calculator's multi-user throughput, not single-stream speed.

## 4. Training and fine-tuning (throughput and memory)

| Source | Trust | What's measured | Saved as |
|---|---|---|---|
| [torchtitan Llama 3.1 on H100](https://github.com/pytorch/torchtitan/blob/main/docs/benchmarks/llama3_h100_202412_torchtitan.md) | A | **TPS/GPU and peak memory (GiB)**, 8B/70B/405B, FSDP / compile / Float8, context parallel at 32k–262k sequence length, 8–512 H100 | `training/llama3_h100_202412_torchtitan.md` |
| [torchtitan Llama 3 8B on H200](https://github.com/pytorch/torchtitan/blob/main/docs/benchmarks/llama3-8b_h200_202506_trainy-whitefiber.md) | A | TPS/GPU and memory, 4–8 nodes | `training/llama3-8b_h200_202506_trainy-whitefiber.md` |
| [torchtitan async TP](https://github.com/pytorch/torchtitan/blob/main/docs/benchmarks/asyncTP_llama3_h100_2025-06_torchtitan.md) | A | tok/s, BF16 vs FP8 | `training/asyncTP_llama3_h100_2025-06_torchtitan.md` |
| [Megatron-Bridge performance summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) ([recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/scripts/performance)) | A | training throughput and MFU by model, GPU and parallelism | not saved |
| [MLPerf Training v5.0 / v5.1 / v6.0](https://github.com/mlcommons/training_results_v6.0) | A (audited) | time-to-train per benchmark and system (contents not yet inspected) | not saved |
| [LlamaFactory paper](https://arxiv.org/abs/2403.13372) | B | measured peak memory and throughput for full, freeze, LoRA and QLoRA tuning (Gemma-2B, Llama-2 7B/13B) | not saved |

Not usable as ground truth: the LLaMA-Factory README "Hardware Requirement" table is a rule of thumb (for example "18x GB"), not a measurement. Unsloth's README gives only relative speedups.

## 5. Kernel and communication profiles

| Source | Trust | What's measured | Saved as |
|---|---|---|---|
| [microsoft/vidur profiling](https://github.com/microsoft/vidur/tree/main/data/profiling) | A | per-operator CUDA times (QKV, attention, MLP, norms) vs token count, and all-reduce / send-recv times vs message size | `vidur/profiling/` (mlp.csv for Llama-2-7B/70B and Llama-3-8B/70B on A100 and H100, plus network CSVs; the 10–20 MB attention.csv files were left out) |

## 6. Diffusion

| Source | Trust | What's measured | Saved as |
|---|---|---|---|
| [HF diffusers benchmarks](https://huggingface.co/datasets/diffusers/benchmarks) (workflow in [huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/benchmarks)) | A | seconds per step and **peak memory (GB)**: Flux.1-dev BF16 / NF4 / offload, LTX-Video, Wan, SDXL, plain vs compiled | `diffusion/hf_diffusers_benchmarks_collated.csv` |
| MLPerf SDXL and Wan-2.2 rows | A | samples/s | inside the MLPerf summaries above |

## Excluded

- **ray-project/llmperf-leaderboard:** benchmarks hosted API providers, so the hardware is unknown.
- **Blog and SEO aggregators** (markaicode, insiderllm, etc.): unverifiable or estimated numbers.
- **Search-result snippets:** none of the numbers in this file come from them; every figure above was read from the downloaded file.
