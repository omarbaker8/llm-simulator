# LLMServingSim comparison

```bash
node tests/llmservingsim/compare.js          # table report
node tests/llmservingsim/compare.js --json   # machine-readable
```

These are the test scenarios from [LLMServingSim](https://github.com/casys-kaist/LLMServingSim) (KAIST, MIT licence — see
`LICENSE-LLMServingSim`), replayed through this calculator. Token ids were stripped from the workloads; nothing else was changed.

| Path | What it is |
|---|---|
| `references/<hw>/<model>/` | Real vLLM runs: 300 ShareGPT requests at about 10 req/s. Includes `vllm_requests.jsonl` (per-request timestamps), `meta.json` (engine settings and recorded KV blocks), `cluster.json`, and their simulator's error table in `llmservingsim_summary.txt`. |
| `scenarios.txt`, `baselines.txt` | The 70 scenarios from their `serving/validate.sh`, plus the total clock their simulator records for each. These are simulator-vs-simulator numbers, not measurements. |
| `clusters/`, `workloads/`, `spec_decode.json` | Their cluster configs, request traces and published speculative-decoding acceptance rates. |

## How the replay works

This calculator works at steady state, so `compare.js` adds a small vLLM-V1-style batching loop:
- a token budget and `max_num_seqs`;
- chunked prefill, with the long-prefill threshold;
- admission by KV block, reserving the full input length, with recompute preemption;
- LOAD / RR / RAND routing;
- DP lockstep;
- prefill/decode handoff over the link;
- agentic tool-call chains.

The calculator times every step:
- **Decode step:** `perf.decodeStepMs` for B sequences at their mean context.
- **Prefill chunk:** the marginal prefill time `P(c+n) − P(c)`.
- **KV capacity:** taken from the calculator's memory model at the scenario's `mem_util`.

The calculator has no model for these, so they are approximated:
- **Expert parallelism:** a DP×EP group of TP1 ranks runs as one TP=EP instance; EP inside TP runs as TP.
- **RTX PRO 6000:** their card is the Server Edition, which uses this calculator's `rtx_6000_blackwell_server_96` entry (1,597 GB/s).

These features aren't modeled, so those scenarios run the base configuration and are flagged in the report:
- CXL and PIM memory tiers;
- CPU and CXL prefix pools;
- prefix-cache hits;
- local offloading;
- expert-routing policies.
