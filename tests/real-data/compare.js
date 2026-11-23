// Compare this calculator against the measured data in tests/real-data/ (see SOURCES.md).
// Usage: node tests/real-data/compare.js [--json out.json] [--rows]
// Every "measured" number is read from the files in this folder; every "ours" number comes from calc.js.
const fs = require('fs'), vm = require('vm'), path = require('path');
const ROOT = path.join(__dirname, '..', '..'), HERE = __dirname;
const ctx = {}; ctx.window = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(ROOT, 'data.js'), 'utf8'), ctx);
const D = ctx.VRAM_DATA, C = require(path.join(ROOT, 'calc.js'));
const M = (s) => D.models.find((m) => m.slug === s), G = (k) => D.gpus.find((g) => g.key === k);
const GiB = 2 ** 30;
const args = process.argv.slice(2), showRows = args.includes('--rows');

/* ---------- helpers ---------- */
const INF = { mode: 'inference', quant: 'fp16', kvQuant: 'fp16', runtime: 'vllm', numGpus: 1, cluster: false, parallelism: 'tensor', interconnect: 'nvlink_gen4',
  batch: 1, users: 1, seq: 2048, outputTokens: 512, prefixShare: 0, offload: false, offloadTarget: 'cpu_ram', offloadFrac: 0, offloadKv: false,
  spec: false, specDraft: 1, specAccept: 0.7, specTokens: 4, specType: 'sequential', elec: 0.12, carbon: 400, opt: {} };
const calc = (m, g, o) => C.calculate({ ...INF, ...o, opt: { ...INF.opt, ...(o.opt || {}) } }, typeof m === 'string' ? M(m) : m, typeof g === 'string' ? G(g) : g);
const seg = (r, k) => (r.memory.segments.find((s) => s.key === k) || { gib: 0 }).gib;
const num = (s) => { if (s == null || s === '') return NaN; const m = String(s).replace(/,/g, '').match(/-?[\d.]+(e[-+]?\d+)?/i); return m ? Number(m[0]) : NaN; };
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : NaN; };
const parseCsvLine = (l) => { const o = []; let c = '', q = false; for (let i = 0; i < l.length; i++) { const ch = l[i]; if (ch === '"') { if (q && l[i + 1] === '"') { c += '"'; i++; } else q = !q; } else if (ch === ',' && !q) { o.push(c); c = ''; } else c += ch; } o.push(c); return o; };
const readCsv = (f) => { const L = fs.readFileSync(f, 'utf8').trim().split('\n'); const H = parseCsvLine(L[0]); return L.slice(1).map((l) => { const c = parseCsvLine(l); return Object.fromEntries(H.map((h, i) => [h, c[i]])); }); };
const lnErr = (ours, meas) => Math.log(ours / meas);           // symmetric; reported as % = e^x − 1
const pct = (x) => (x >= 0 ? '+' : '') + (x * 100).toFixed(1) + '%';

const sections = [];
function section(name, metric, rows) {
  const ok = rows.filter((r) => Number.isFinite(r.ours) && Number.isFinite(r.meas) && r.meas > 0 && r.ours > 0);
  ok.forEach((r) => { r.err = r.ours / r.meas - 1; r.lerr = lnErr(r.ours, r.meas); });
  const abs = ok.map((r) => Math.abs(r.lerr));
  const s = { name, metric, n: ok.length, skipped: rows.length - ok.length,
    bias: ok.length ? Math.exp(median(ok.map((r) => r.lerr))) - 1 : NaN,
    medAbs: ok.length ? Math.exp(median(abs)) - 1 : NaN,
    within10: ok.filter((r) => Math.abs(r.err) <= 0.10).length, within25: ok.filter((r) => Math.abs(r.err) <= 0.25).length, rows: ok };
  sections.push(s);
  return s;
}

/* ---------- custom model configs (published Hugging Face config.json values) ---------- */
const llama2_7b = { slug: 'llama-2-7b', name: 'Llama-2-7B', architecture: 'dense', num_of_params: '6.74', hidden_dim_size: 4096, ffn_intermediate_size: 11008, vocab_size: 32000,
  num_of_layers: 32, num_attention_heads: 32, num_key_value_heads: 32, attention_head_dim: 128, attention_structure: 'mha', context_length: 4096, tie_word_embeddings: false, base_precision: 'fp16', task_type: 'text_generation' };
const llama2_70b = { ...llama2_7b, slug: 'llama-2-70b', name: 'Llama-2-70B', num_of_params: '68.98', hidden_dim_size: 8192, ffn_intermediate_size: 28672, num_of_layers: 80,
  num_attention_heads: 64, num_key_value_heads: 8, attention_structure: 'gqa' };

/* ===================================================================== *
 * A. llama.cpp scoreboards: Llama-2-7B, pp512 and tg128 (llama-bench)
 * ===================================================================== */
const CUDA_MAP = { 'RTX 5090': '5090_32', 'RTX PRO 6000 Blackwell': 'rtx_6000_blackwell_96', 'H100 80 GB': 'h100_80', 'A100 80 GB': 'a100_80', 'RTX 4090 D': '4090d_24', 'RTX 4090': '4090_24',
  'RTX 5080': '5080_16', 'RTX 5070 Ti': '5070ti_16', 'RTX 6000 Ada': 'rtx_6000_ada_48', 'RTX 3090 Ti': '3090ti_24', 'RTX PRO 4500 Blackwell': 'rtx_4500_blackwell_32', 'RTX 3090': '3090_24',
  L40: 'l40_48', 'RTX 4080 SUPER': '4080super_16', 'RTX 4080': '4080_16', 'RTX A6000': 'rtx_a6000_48', 'RTX 3080': '3080_10', 'RTX PRO 4000 Blackwell': 'rtx_4000_blackwell_24',
  'RTX A5000': 'rtx_a5000_24', 'RTX 4070 Ti SUPER': '4070tisuper_16', 'RTX 5070': '5070_12', A40: 'a40_48', A30: 'a30_24', 'RTX 5060 Ti': '5060ti_16', 'RTX A4000': 'rtx_a4000_16',
  'RTX 3060': '3060_12', 'RTX 4060 Ti': '4060ti_16', 'RTX A4000 Ada': 'rtx_4000_ada_20', 'RTX 2000 Ada': 'rtx_2000_ada_16', 'DGX Spark': 'gb10_128', 'Tesla T4': 't4_16' };
const ROCM_MAP = { 'Instinct MI300X': 'mi300x_192', 'RX 7900 XTX': 'rx_7900xtx_24', 'Instinct MI210': 'mi210_64', 'Pro W7900': 'radeon_pro_w7900_48', 'RX 9070': 'rx_9070_16',
  'RX 7900 XT': 'rx_7900xt_20', 'AI PRO R9700': 'radeon_ai_pro_r9700_32', 'RX 7900 GRE': 'rx_7900gre_16', 'RX 9070 XT': 'rx_9070xt_16', 'RX 7800 XT': 'rx_7800xt_16',
  'RX 6900 XT': 'rx_6900xt_16', 'RX 9060 XT': 'rx_9060xt_16', 'AI Max+ 395': 'ryzen_ai_max_395_128', 'RX 7600 XT': 'rx_7600xt_16' };
// Apple: (chip, bandwidth GB/s) → our entry; the scoreboard lists core-count variants, we use the full-core one per bandwidth.
const APPLE_MAP = { 'M2 Pro|200': 'm2_pro_16', 'M2 Max|400': 'm2_max_64', 'M2 Ultra|800': 'm2_ultra_192', 'M3 Pro|150': 'm3_pro_36', 'M3 Max|300': 'm3_max_36', 'M3 Max|400': 'm3_max_64',
  'M3 Ultra|800': 'm3_ultra_512', 'M4|120': 'm4_32', 'M4 Pro|273': 'm4_pro_48', 'M4 Max|546': 'm4_max_128', 'M5|154': 'm5_32', 'M5 Pro|307': 'm5_pro_48', 'M5 Max|614': 'm5_max_128' };

function llamaBench(m, gKey, quant, extra = {}) {
  const tg = calc(m, gKey, { runtime: 'llamacpp', quant, seq: 129, outputTokens: 128, ...extra }).perf;   // tg128 from an empty context
  const pp = calc(m, gKey, { runtime: 'llamacpp', quant, seq: 513, outputTokens: 1, ...extra }).perf;     // pp512
  return { tg: tg ? tg.tps : NaN, pp: pp ? pp.prefillTps : NaN };
}
{
  const ppRows = [], tgRows = [];
  const ppNo = [], tgNo = [];
  const board = (file, map, label) => {
    for (const r of readCsv(path.join(HERE, 'llamacpp', file))) {
      if (/no_fa$/.test(r.table)) {
        const k0 = map[r.Chip]; if (!k0 || !G(k0)) continue;
        const o0 = llamaBench(llama2_7b, k0, 'q4_0');
        ppNo.push({ src: label, gpu: k0, what: 'pp512 Q4_0 no-FA', meas: num(r['pp512 t/s']), ours: o0.pp });
        tgNo.push({ src: label, gpu: k0, what: 'tg128 Q4_0 no-FA', meas: num(r['tg128 t/s']), ours: o0.tg });
        continue;
      }
      if (!/_fa$/.test(r.table)) continue;
      const k = map[r.Chip]; if (!k || !G(k)) continue;
      const o = llamaBench(llama2_7b, k, 'q4_0');
      ppRows.push({ src: label, gpu: k, what: 'pp512 Q4_0', meas: num(r['pp512 t/s']), ours: o.pp });
      tgRows.push({ src: label, gpu: k, what: 'tg128 Q4_0', meas: num(r['tg128 t/s']), ours: o.tg });
    }
  };
  board('cuda_scoreboard_15013.csv', CUDA_MAP, 'CUDA #15013');
  board('rocm_scoreboard_15021.csv', ROCM_MAP, 'ROCm #15021');
  section('A1 llama.cpp scoreboards (CUDA + ROCm), Llama-2-7B Q4_0, FA on', 'tg128 tok/s', tgRows);
  section('A2 llama.cpp scoreboards (CUDA + ROCm), Llama-2-7B Q4_0, FA on', 'pp512 tok/s', ppRows);
  section('A1b same scoreboards, flash attention off', 'tg128 tok/s', tgNo);
  section('A2b same scoreboards, flash attention off', 'pp512 tok/s', ppNo);
  const aTg = [], aPp = [];
  const best = {};
  for (const r of readCsv(path.join(HERE, 'llamacpp', 'apple_scoreboard_4167.csv'))) {
    if (r.table !== 'apple_summary') continue;
    const chip = r.Chip.replace(/^[^\w]+\s*/, '').replace(/\s+\d$/, '').trim();
    const key = APPLE_MAP[`${chip}|${num(r['BW [GB/s]'])}`]; if (!key) continue;
    if (!best[key] || num(r['GPU Cores']) > best[key].cores) best[key] = { cores: num(r['GPU Cores']), r };
  }
  for (const [key, { r }] of Object.entries(best)) {
    for (const [q, col] of [['fp16', 'F16'], ['q8_0', 'Q8_0'], ['q4_0', 'Q4_0']]) {
      const o = llamaBench(llama2_7b, key, q);
      aTg.push({ src: 'Apple #4167', gpu: key, what: `tg128 ${col}`, meas: num(r[`${col} TG [t/s]`]), ours: o.tg });
      aPp.push({ src: 'Apple #4167', gpu: key, what: `pp512 ${col}`, meas: num(r[`${col} PP [t/s]`]), ours: o.pp });
    }
  }
  section('A3 llama.cpp Apple Silicon, Llama-2-7B F16/Q8_0/Q4_0', 'tg128 tok/s', aTg);
  section('A4 llama.cpp Apple Silicon, Llama-2-7B F16/Q8_0/Q4_0', 'pp512 tok/s', aPp);
}

/* ===================================================================== *
 * B. XiongjieDai: Llama 3 8B/70B, Q4_K_M and F16, tg1024 / pp1024, 1–8 GPUs
 * ===================================================================== */
{
  const XMAP = { '3070 8GB': '3070_8', '3080 10GB': '3080_10', '3080 Ti 12GB': '3080ti_12', '4070 Ti 12GB': '4070ti_12', '4080 16GB': '4080_16', 'RTX 4000 Ada 20GB': 'rtx_4000_ada_20',
    '3090 24GB': '3090_24', '4090 24GB': '4090_24', 'RTX 5000 Ada 32GB': 'rtx_5000_ada_32', 'RTX A6000 48GB': 'rtx_a6000_48', 'RTX 6000 Ada 48GB': 'rtx_6000_ada_48', 'A40 48GB': 'a40_48',
    'L40S 48GB': 'l40s_48', 'A100 PCIe 80GB': 'a100_80', 'A100 SXM 80GB': 'a100_80', 'H100 PCIe 80GB': 'h100pcie_80', 'M2 Ultra 76-Core GPU 192GB': 'm2_ultra_192', 'M3 Max 40‑Core GPU 64GB': 'm3_max_64' };
  const md = fs.readFileSync(path.join(HERE, 'llamacpp', 'xiongjiedai_gpu_benchmarks_README.md'), 'utf8').split('\n');
  const tables = []; let cur = null;
  for (const l of md) { if (/^\|/.test(l)) { (cur = cur || []).push(l.split('|').slice(1, -1).map((c) => c.trim())); } else if (cur) { tables.push(cur); cur = null; } }
  const tg = [], pp = [], fit = [], anomalies = [];
  const cols = [['8B Q4_K_M', 'llama-3-1-8b', 'q4_k_m'], ['8B F16', 'llama-3-1-8b', 'fp16'], ['70B Q4_K_M', 'llama-3-1-70b', 'q4_k_m'], ['70B F16', 'llama-3-1-70b', 'fp16']];
  // tables[0] = TG 1024 overview, tables[1] = PP 1024 overview (README "Overview" section)
  [[tables[0], tg, 'tg1024'], [tables[1], pp, 'pp1024']].forEach(([tab, out, kind]) => {
    const H = tab[0];
    for (const row of tab.slice(2)) {
      const m = row[0].match(/^(.*?)(?:\s*\*\s*(\d+))?$/); const name = m[1].trim(), n = Number(m[2] || 1);
      const key = XMAP[name]; if (!key) continue;
      for (const [col, slug, q] of cols) {
        const ci = H.indexOf(col); if (ci < 0) continue;
        const cell = row[ci];
        const o = { runtime: 'llamacpp', quant: q, numGpus: n, parallelism: n > 1 ? 'pipeline' : 'tensor', interconnect: 'pcie4' };
        const r = kind === 'tg1024' ? calc(slug, key, { ...o, seq: 1025, outputTokens: 1024 }) : calc(slug, key, { ...o, seq: 1025, outputTokens: 1 });
        const oursFits = r.memory.pct <= 100;
        if (kind === 'tg1024') fit.push({ src: 'XiongjieDai', gpu: `${key}×${n}`, what: col, measOOM: /OOM/.test(cell), oursOOM: !oursFits, pct: r.memory.pct });
        if (/OOM/.test(cell) || !oursFits) continue;
        if (kind === 'pp1024' && /F16/.test(col)) { const qi = H.indexOf(col.replace('F16', 'Q4_K_M')); if (qi >= 0 && num(cell) < 0.6 * num(row[qi])) { anomalies.push(`${name}${n > 1 ? ' ×' + n : ''} ${col} pp ${cell} vs Q4_K_M ${row[qi]}`); continue; } }
        out.push({ src: 'XiongjieDai', gpu: `${key}×${n}`, what: `${kind} ${col}`, meas: num(cell), ours: kind === 'tg1024' ? r.perf.tps : r.perf.prefillTps });
      }
    }
  });
  section('B1 llama.cpp Llama 3 8B/70B (XiongjieDai), 1–8 GPUs, layer split', 'tg1024 tok/s', tg);
  section('B2 llama.cpp Llama 3 8B/70B (XiongjieDai), 1–8 GPUs, layer split', 'pp1024 tok/s', pp).note = 'excluded as source errors (F16 slower than Q4_K_M by >40%): ' + anomalies.join('; ');
  const agree = fit.filter((f) => f.measOOM === f.oursOOM).length;
  sections.push({ name: 'B3 fits-or-OOM agreement (XiongjieDai)', metric: 'OOM verdict', n: fit.length, agree, mismatches: fit.filter((f) => f.measOOM !== f.oursOOM), custom: true });
}

/* ===================================================================== *
 * C. Argonne LLM-Inference-Bench: vLLM and TensorRT-LLM batch latency (FP16)
 *    latency = time to prefill bs prompts of L tokens and decode L tokens each (input = output = L)
 * ===================================================================== */
{
  const HW = { 'Nvidia A100 GPU': ['a100_40', 'nvlink_gen3'], 'Nvidia H100 GPU': ['h100_80', 'nvlink_gen4'], 'AMD MI300X GPU': ['mi300x_192', 'amd_xgmi'] };
  const MOD = { 'meta-llama/Llama-2-7b-hf': llama2_7b, 'meta-llama/Meta-Llama-3-8B': 'llama-3-1-8b', 'meta-llama/Meta-Llama-3-70B': 'llama-3-1-70b', 'meta-llama/Llama-2-70b-hf': llama2_70b,
    'mistralai/Mistral-7B-v0.1': 'mistral-7b-instruct-v0-2', 'mistralai/Mixtral-8x7B-v0.1': 'mixtral-8x7b-v0-1', 'Qwen/Qwen2-7B': 'qwen2-5-7b', 'Qwen/Qwen2-72B': 'qwen2-5-72b' };
  const RT = { vLLM: 'vllm' };
  const rows = [];
  for (const r of readCsv(path.join(HERE, 'serving', 'argonne_llm_inference_bench_all_results.csv'))) {
    const hw = HW[r.Hardware], m = MOD[r.Model], rt = RT[r.Framework]; if (!hw || !m || !rt) continue;
    const n = num(r['Num of Hardware']), L = num(r['Input Output Length']), bs = num(r['Batch Size']);
    const res = calc(m, hw[0], { runtime: rt, quant: 'fp16', numGpus: n, parallelism: 'tensor', interconnect: hw[1], users: bs, seq: 2 * L, outputTokens: L });
    if (!res.perf || res.memory.pct > 100) continue;
    const ours = res.perf.ttftQueued / 1000 + (L - 1) * res.perf.decodeStepMs / 1000;
    rows.push({ src: `Argonne ${r.Framework}`, gpu: `${hw[0]}×${n}`, what: `${typeof m === 'string' ? m : m.slug} L=${L} bs=${bs}`, meas: num(r.Latency), ours, rt, bs, L });
  }
  section('C1 Argonne vLLM, batch latency (in = out = L, FP16)', 'seconds', rows.filter((r) => r.rt === 'vllm'));
  // TensorRT-LLM rows are not used: run.py stops at the tokenizer's end_id (no ignore_eos), so output lengths are unknown.
}

/* ===================================================================== *
 * D. InferenceX: memory from server logs, and TPOT / TTFT / throughput per GPU
 * ===================================================================== */
const IX_MODEL = { 'Llama-3.3-70B-Instruct-FP8': 'llama-3-3-70b', 'gpt-oss-120b': 'gpt-oss-120b', 'DeepSeek-R1-0528': 'deepseek-r1-671b', 'DeepSeek-V4-Pro': 'deepseek-v4-pro',
  'DeepSeek-V4.1-Flash': 'deepseek-v4-1-flash', 'GLM-5': 'glm-5', 'GLM-5.2': 'glm-52', 'GLM-5.3-Flash': 'glm-53-flash', 'Kimi-K2.5': 'kimi-k25', 'Kimi-K3': 'kimi-k3',
  'Kimi-K3.1-Pro': 'kimi-k3-1-pro', 'MiniMax-M2.5': 'minimax-m2', 'MiniMax-M3': 'minimax-m3', 'MiniMax-M3.1-Flash': 'minimax-m3-1-flash', 'Qwen-3.5-397B-A17B': 'qwen35-397b-a17b',
  'Qwen3.8-Flash-Next': 'qwen-38-flash' };
const IX_HW = { h100: ['h100_80', 'nvlink_gen4'], h200: ['h200_141', 'nvlink_gen4'], b200: ['b200_192', 'nvlink_gen5'], b300: ['b300_288', 'nvlink_gen5'],
  mi300x: ['mi300x_192', 'amd_xgmi'], mi325x: ['mi325x_256', 'amd_xgmi'], mi355x: ['mi355x_288', 'amd_xgmi'], rtx6000pro: ['rtx_6000_blackwell_server_96', 'pcie5'] };
const IX_RT = { vllm: 'vllm', sglang: 'sglang', trt: 'trtllm' };
const ixQuant = (prec, hw, model) => prec === 'fp8' ? 'fp8' : prec === 'int4' ? 'int4' : prec === 'bf16' ? 'fp16' : prec === 'fp4' ? ((/^mi/.test(hw) || model === 'gpt-oss-120b') ? 'mxfp4' : 'nvfp4') : null;
const ixKv = (logKv) => logKv && /fp8/i.test(logKv) ? 'fp8' : 'fp16';
{
  const rows = readCsv(path.join(HERE, 'inferencex', 'inferencex_benchmarks_all.csv'));
  const logs = JSON.parse(fs.readFileSync(path.join(HERE, 'inferencex', 'server_log_memory.json'), 'utf8'));
  const clean = rows.filter((r) => r.disagg === 'false' && r.is_multinode === 'false' && IX_RT[r.framework] && IX_HW[r.hardware] && r.spec_method === 'none'
    && r.offload_mode !== 'on' && IX_MODEL[r.display_model] && M(IX_MODEL[r.display_model]) && r.benchmark_type === 'single_turn' && num(r.isl) > 0);
  // config key → log facts (one log per config)
  const cfgKey = (r) => [r.display_model, r.hardware, r.framework, r.precision, r.decode_tp, r.decode_ep, r.decode_dp_attention].join('|');
  const logByCfg = {};
  for (const r of rows) if (logs[r.id]) logByCfg[cfgKey(r)] = { ...logs[r.id], row: r };
  const memW = [], memKvTok = [], memCap = [], memKvTokH = [], memCapH = [];
  // Hybrid / sparse / compressed attention: the engines' "KV cache size" counts tokens differently (see README notes).
  const HYBRID = /gpt-oss|Qwen-3.5|Qwen3.8|MiniMax-M3|DeepSeek-V4|GLM-5\.[23]/;
  for (const [k, L] of Object.entries(logByCfg)) {
    const r = L.row; if (!clean.includes(r) && !(r.disagg === 'false' && IX_MODEL[r.display_model] && IX_HW[r.hardware] && IX_RT[r.framework])) continue;
    const slug = IX_MODEL[r.display_model]; if (!M(slug)) continue;
    const [gk, link] = IX_HW[r.hardware], quant = ixQuant(r.precision, r.hardware, r.display_model); if (!quant) continue;
    const tp = num(r.decode_tp) || 1, rt = IX_RT[r.framework], dpa = r.decode_dp_attention === 'true';
    const kvQuant = ixKv(L.kv_cache_dtype);
    const base = { runtime: rt, quant, kvQuant, numGpus: tp, parallelism: 'tensor', interconnect: link, users: 1 };
    const r1 = calc(slug, gk, { ...base, seq: 16384, outputTokens: 1 }), r2 = calc(slug, gk, { ...base, seq: 32768, outputTokens: 1 });
    const weightsOurs = seg(r1, 'weights');
    const kvPerTokGpu = (seg(r2, 'kv') - seg(r1, 'kv')) * GiB / 16384;             // bytes per token per GPU (ours)
    const tag = `${r.display_model} ${r.hardware} ${r.framework} ${r.precision} TP${tp}${num(r.decode_ep) > 1 ? ' EP' + r.decode_ep : ''}${dpa ? ' DPattn' : ''}`;
    const hybrid = HYBRID.test(r.display_model);
    const wMeas = L.vllm_weights_gib ?? L.sgl_weights_gib ?? L.trt_weights_gib;
    if (Number.isFinite(wMeas) && !dpa) memW.push({ src: `IX ${r.framework}`, gpu: gk, what: tag, meas: wMeas, ours: weightsOurs, id: r.id });
    const kvG = L.vllm_kv_gib ?? L.sgl_kv_gib ?? L.trt_kv_gib, kvT = L.vllm_kv_tokens ?? L.sgl_kv_tokens ?? L.trt_kv_tokens;
    if (Number.isFinite(kvG) && Number.isFinite(kvT) && kvT > 0 && !dpa) (hybrid ? memKvTokH : memKvTok).push({ src: `IX ${r.framework}`, gpu: gk, what: tag, meas: kvG * GiB / kvT, ours: kvPerTokGpu, id: r.id, kvQuant });
    // capacity in tokens at the engine's own memory fraction (vLLM: gpu_memory_utilization; SGLang: mem_fraction_static)
    const util = L.gpu_mem_util ?? L.sgl_mem_fraction;
    if (Number.isFinite(kvT) && Number.isFinite(util) && !dpa) {
      const nonKv = r1.memory.perGpu - seg(r1, 'kv');
      const usable = r1.memory.capPerGpu / (rt === 'vllm' || rt === 'sglang' || rt === 'trtllm' ? 0.9 : 1);
      const capOurs = (util * usable - nonKv) * GiB / kvPerTokGpu;
      (hybrid ? memCapH : memCap).push({ src: `IX ${r.framework}`, gpu: gk, what: tag + ` util ${util}`, meas: kvT, ours: capOurs, id: r.id });
    }
  }
  section('D1 InferenceX server logs: weight memory per GPU', 'GiB', memW);
  section('D2 InferenceX server logs: KV bytes per token per GPU (standard attention)', 'bytes', memKvTok);
  section('D3 InferenceX server logs: KV pool size at the engine memory fraction', 'tokens', memCap);
  section('D2h same, hybrid/sparse/compressed-attention models', 'bytes', memKvTokH);
  section('D3h same, hybrid/sparse/compressed-attention models', 'tokens', memCapH);
  // speed
  const tpot = [], ttft = [], thr = [];
  for (const r of clean) {
    if (r.decode_dp_attention === 'true') continue;
    const slug = IX_MODEL[r.display_model], [gk, link] = IX_HW[r.hardware], quant = ixQuant(r.precision, r.hardware, r.display_model); if (!quant) continue;
    const tp = num(r.decode_tp) || 1, conc = num(r.conc), isl = num(r.isl), osl = num(r.osl), rt = IX_RT[r.framework];
    const L = logByCfg[cfgKey(r)] || {};
    const res = calc(slug, gk, { runtime: rt, quant, kvQuant: ixKv(L.kv_cache_dtype), numGpus: tp, parallelism: 'tensor', interconnect: link, users: conc, seq: isl + osl, outputTokens: osl });
    if (!res.perf) continue;
    const tag = `${r.display_model} ${r.hardware} ${r.framework} ${r.precision} TP${tp}${num(r.decode_ep) > 1 ? ' EP' + r.decode_ep : ''} c${conc} ${isl}/${osl}`;
    const common = { src: `IX ${r.framework}`, gpu: gk, what: tag, id: r.id, model: r.display_model, hw: r.hardware, fw: r.framework, conc, tp, fits: res.memory.pct <= 100 };
    tpot.push({ ...common, meas: num(r.mean_tpot) * 1000, ours: res.perf.msPerToken });
    ttft.push({ ...common, meas: num(r.median_ttft) * 1000, ours: res.perf.ttft });
    thr.push({ ...common, meas: num(r.output_tput_per_gpu), ours: res.perf.throughput / tp });
  }
  const d4 = section('D4 InferenceX: mean TPOT (single node, no DP-attention, no spec decode)', 'ms', tpot);
  const cls = (m) => /Llama/.test(m) ? 'dense' : /DeepSeek-R1|Kimi|GLM-5$/.test(m) ? 'MLA MoE' : 'other MoE';
  d4.groups = {};
  for (const r of d4.rows) for (const k of [`${cls(r.model)} TP${r.tp}`, `${cls(r.model)} ${r.hw}`]) (d4.groups[k] = d4.groups[k] || []).push(r.lerr);
  section('D5 InferenceX: median TTFT', 'ms', ttft);
  section('D6 InferenceX: output throughput per GPU', 'tok/s/GPU', thr);
}

/* ===================================================================== *
 * E. torchtitan: Llama 3.1 8B FSDP on H100 96GB HBM2e (2.4 TB/s, 500 W), seq 8192, local batch 2
 * ===================================================================== */
{
  const h100_96 = { ...G('h100_80'), key: 'h100_96_hbm2e', memory: 96, bandwidth_gbs: 2400, label: 'H100 96GB HBM2e (Grand Teton)' };
  const FT = { mode: 'finetuning', quant: 'fp16', ftMethod: 'full', ftQuant: 'fp16', loraRank: 16, gradSteps: 1, numSamples: 1000, tokensPerSample: 8192, numEpochs: 1, seq: 8192, batch: 2,
    parallelism: 'replicated', interconnect: 'nvlink_gen4', opt: { optimizer_type: 'adamw', zero_stage: 3, flash_attention: true, gradient_checkpointing: true, gradient_checkpointing_ratio: 0.5, use_8bit_optimizer: false, sequence_packing: true } };
  const rowsMem = [], rowsTps = [];
  for (const [tbl, n, measTps, measMem] of [['Table 1 (8 GPUs)', 8, 5762, 82.4], ['Table 3 (128 GPUs)', 128, 5605, 67.0]]) {
    const r = calc('llama-3-1-8b', h100_96, { ...FT, numGpus: n });
    rowsMem.push({ src: 'torchtitan', gpu: `H100-96×${n}`, what: `${tbl} FSDP, selective AC (ours: ratio 0.5)`, meas: measMem, ours: r.memory.perGpu });
    const tps = r.perf ? r.perf.trainTps / n : NaN;
    rowsTps.push({ src: 'torchtitan', gpu: `H100-96×${n}`, what: tbl, meas: measTps, ours: tps, perf: r.perf });
  }
  section('E1 torchtitan Llama 3.1 8B FSDP: peak memory per GPU', 'GiB', rowsMem);
  section('E2 torchtitan Llama 3.1 8B FSDP: tokens/s per GPU', 'tok/s/GPU', rowsTps);
}

/* ===================================================================== *
 * F. diffusers benchmarks (L40S 48GB, batch 1, one denoiser forward, BF16)
 * ===================================================================== */
{
  const DIF = { mode: 'inference', quant: 'fp16', numGpus: 1, cluster: false, parallelism: 'tensor', batch: 1, encoderOffload: true, vaeTiling: false, diffRuntime: 'diffusers', inferenceSteps: 1, guidanceScale: 1, opt: {} };
  const cases = [['flux-1-dev', 1024, 1024, 1, 0.498, 22.64], ['sdxl-1-0-base', 1024, 1024, 1, 0.072, 5.05], ['wan-2-1-t2v-14b', 832, 480, 81, 10.198, 30.24]];
  const t = [], mem = [];
  for (const [slug, w, h, f, sec, gb] of cases) {
    if (!M(slug)) continue;
    const r = C.calculate({ ...INF, ...DIF, imageWidth: w, imageHeight: h, numFrames: f }, M(slug), G('l40s_48'));
    t.push({ src: 'diffusers', gpu: 'l40s_48', what: `${slug} ${w}×${h}${f > 1 ? '×' + f : ''}`, meas: sec, ours: r.perf ? r.perf.stepMs / 1000 : NaN });
    mem.push({ src: 'diffusers', gpu: 'l40s_48', what: `${slug} weights + activations`, meas: gb, ours: seg(r, 'weights') + seg(r, 'act') });
  }
  section('F1 diffusers: seconds per denoiser step', 's', t);
  section('F2 diffusers: denoiser peak allocated memory', 'GiB', mem);
}

/* ===================================================================== *
 * G. vLLM blog: Mistral Large 4 NVFP4 KV capacity on 8× GB300, TP8, FP8 KV, util 0.85 → 3.60M tokens
 * ===================================================================== */
{
  const gb300 = { ...G('b300_288'), key: 'gb300_278', memory: 278, label: 'GB300 (278 GB usable)' };
  const rows = [];
  for (const [g, lbl] of [[G('b300_288'), 'b300_288 entry'], [gb300, 'GB300 278 GB usable']]) {
    const base = { runtime: 'vllm', quant: 'nvfp4', kvQuant: 'fp8', numGpus: 8, parallelism: 'tensor', interconnect: 'nvlink_gen5', users: 1, outputTokens: 1 };
    const r1 = calc('mistral-large-4-1t-a52b', g, { ...base, seq: 131072 }), r2 = calc('mistral-large-4-1t-a52b', g, { ...base, seq: 262144 });
    const slope = (seg(r2, 'kv') - seg(r1, 'kv')) / 131072, nonKv = r1.memory.perGpu - seg(r1, 'kv');
    const cap = (0.85 * r1.memory.capPerGpu / 0.9 - nonKv) / slope;
    rows.push({ src: 'vLLM blog', gpu: lbl, what: 'ML4 NVFP4 TP8 KV pool', meas: 3.60e6, ours: cap });
  }
  section('G1 vLLM blog: Mistral Large 4 KV pool, 8× GB300 TP8', 'tokens', rows);
}

/* ---------- report ---------- */
if (args.includes('--json')) { const f = args[args.indexOf('--json') + 1]; fs.writeFileSync(f, JSON.stringify(sections, null, 1)); }
console.log('Real-world comparison (ours ÷ measured − 1; bias = median signed error, |err| = median absolute)\n');
for (const s of sections) {
  if (s.custom) { console.log(`${s.name}: ${s.agree}/${s.n} agree`); s.mismatches.forEach((m) => console.log(`    mismatch ${m.gpu} ${m.what}: measured ${m.measOOM ? 'OOM' : 'runs'}, ours ${m.oursOOM ? 'OOM' : 'fits'} (${m.pct.toFixed(0)}%)`)); continue; }
  console.log(`${s.name} [${s.metric}]  n=${s.n}${s.skipped ? ` (+${s.skipped} skipped)` : ''}  bias ${pct(s.bias)}  |err| ${pct(s.medAbs).replace('+', '')}  ±10%: ${s.within10}/${s.n}  ±25%: ${s.within25}/${s.n}`);
  if (s.note) console.log('    note: ' + s.note);
  if (s.groups) for (const [k, v] of Object.entries(s.groups).sort()) console.log(`    group ${k.padEnd(22)} n=${String(v.length).padStart(4)}  bias ${pct(Math.exp(median(v)) - 1).padStart(7)}  |err| ${pct(Math.exp(median(v.map(Math.abs))) - 1).replace('+', '')}`);
  if (showRows || s.n <= 12) for (const r of [...s.rows].sort((a, b) => Math.abs(b.lerr) - Math.abs(a.lerr))) console.log(`    ${pct(r.err).padStart(8)}  ${r.src} ${r.gpu} ${r.what}: measured ${+r.meas.toPrecision(5)}, ours ${+r.ours.toPrecision(5)}`);
  else for (const r of [...s.rows].sort((a, b) => Math.abs(b.lerr) - Math.abs(a.lerr)).slice(0, 5)) console.log(`    worst ${pct(r.err).padStart(8)}  ${r.src} ${r.gpu} ${r.what}: measured ${+r.meas.toPrecision(5)}, ours ${+r.ours.toPrecision(5)}`);
}
