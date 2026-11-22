// Replays LLMServingSim's test scenarios through this calculator and compares the results.
//
//   node tests/llmservingsim/compare.js            all scenarios + the real-vLLM references
//   node tests/llmservingsim/compare.js --json     machine-readable output
//
// Data in this folder is copied from github.com/casys-kaist/LLMServingSim (MIT, see LICENSE-LLMServingSim), with token
// ids stripped from the workloads:
//   scenarios.txt  + baselines.txt  their serving/validate.sh scenario list and the total clock (ns) THEIR simulator records
//   references/                     real vLLM runs (300 ShareGPT requests at ~10 req/s) and their simulator's error table
//
// This calculator is a steady-state model, not a discrete-event simulator, so the harness below adds a thin vLLM-V1-like
// continuous-batching loop (token budget, max_num_seqs, chunked prefill, KV-block admission and recompute preemption,
// LOAD / RR / RAND routing, DP lockstep, P/D handoff, agentic chains). Every step is timed with the calculator itself:
//   decode step   = perf.decodeStepMs for B running sequences at their mean context
//   prefill chunk = marginal prefill time P(c + n) − P(c) of an n-token chunk after c cached tokens (one weight pass is
//                   already paid by the decode part of the step; a prefill-only step pays it once)
//   KV capacity   = (mem_util × usable memory − weights, runtime and activation memory) ÷ KV bytes per token, all from
//                   the calculator's memory model
// Features this calculator does not model (CXL / PIM tiers, CPU prefix pools, local offloading, expert-routing policies,
// prefix-cache hits) run as the base configuration and are flagged.
const fs = require('fs'), vm = require('vm'), path = require('path');
const HERE = __dirname, ROOT = path.join(HERE, '..', '..');
const ctx = {}; ctx.window = ctx; vm.createContext(ctx);
vm.runInContext(fs.readFileSync(ROOT + '/data.js', 'utf8'), ctx);
const D = ctx.VRAM_DATA, C = require(ROOT + '/calc.js');
const JSON_OUT = process.argv.includes('--json');

/* ---------------- model / GPU mapping ---------------- */
const M = (s) => { const m = D.models.find((x) => x.slug === s); if (!m) throw new Error('model ' + s); return m; };
// DeepSeek-V3.2-Exp-16L64E: the reduced checkpoint LLMServingSim profiles (16 layers, 3 dense, 64 routed + 1 shared
// expert). Counted from its config.json: attention (MLA + DSA indexer) 201M/layer, dense FFN 396M, expert 44.0M,
// embeddings 2 × 129280 × 7168 → 43.48B total, 11.41B active.
const DS16 = { ...M('deepseek-v32'), slug: 'deepseek-v32-16l64e', name: 'DeepSeek-V3.2-Exp-16L64E', num_of_layers: 16, num_of_experts: 64, num_of_params: '43.48', num_of_expert_params: '11.41' };
const MODELS = {
  'meta-llama/Llama-3.1-8B': [M('llama-3-1-8b'), 'fp16', 16], 'meta-llama/Llama-3.1-70B': [M('llama-3-1-70b'), 'fp16', 16],
  'Qwen/Qwen3-30B-A3B-Instruct-2507': [M('qwen3-30b-a3b'), 'fp16', 16], 'Qwen/Qwen3-32B': [M('qwen3-32b'), 'fp16', 16],
  'Qwen/Qwen3.8-27B': [M('qwen38-27b'), 'fp16', 784], 'deepseek-ai/DeepSeek-V3.2-Exp': [M('deepseek-v32'), 'fp8', 64],
  'deepseek-ai/DeepSeek-V3.2-Exp-16L64E': [DS16, 'fp8', 64], 'zai-org/GLM-5': [M('glm-5'), 'fp16', 64], 'MiniMaxAI/MiniMax-M3': [M('minimax-m3'), 'fp16', 128],
};
// LLMServingSim's RTXPRO6000 is the Server Edition (hardware.yaml: "RTX PRO 6000 Blackwell Server Edition", 1597 GB/s).
const GPUS = { RTXPRO6000: 'rtx_6000_blackwell_server_96', RTX4090: '4090_24', H100: 'h100_80' };
const GPU = (hw) => { const g = D.gpus.find((x) => x.key === GPUS[hw]); if (!g) throw new Error('gpu ' + hw); return g; };
const SPEC = JSON.parse(fs.readFileSync(HERE + '/spec_decode.json', 'utf8')).models;

const INF = { mode: 'inference', quant: 'fp16', kvQuant: 'fp16', runtime: 'vllm', numGpus: 1, cluster: true, numNodes: 1, gpusPerNode: 1, tpDegree: 1, ppDegree: 1,
  parallelism: 'tensor', interconnect: 'pcie5', interNode: 'infiniband_hdr', batch: 1, users: 1, seq: 4096, outputTokens: 512, prefixShare: 0, offload: false,
  offloadTarget: 'cpu_ram', offloadFrac: 0, offloadKv: false, spec: false, specDraft: 1, specAccept: 0.7, specTokens: 4, specType: 'sequential', elec: 0.12, carbon: 400, opt: {} };

/* ---------------- per-instance cost model (all numbers from calc.js) ---------------- */
function costModel(modelName, hw, tp, pp, memUtil, specK) {
  const [m, quant] = MODELS[modelName] || (() => { throw new Error('unmapped model ' + modelName); })();
  const g = GPU(hw), gBig = { ...g, memory: 1e6 };   // speed only: same bandwidth / FLOPS, never "does not fit"
  const n = tp * pp;
  const base = { ...INF, quant, gpusPerNode: n, tpDegree: tp, ppDegree: pp };
  const run = (o, gg = gBig) => C.calculate({ ...base, ...o }, m, gg);
  const memo = new Map();
  const cached = (k, f) => { if (!memo.has(k)) memo.set(k, f()); return memo.get(k); };
  const bucket = (c) => Math.max(1, Math.round(c / 32) * 32);
  const decode = (B, c) => cached(`d${B}|${bucket(c)}`, () => run({ users: B, seq: bucket(c) + 2, outputTokens: 2 }).perf.decodeStepMs / 1000);
  const prefill = (N) => N <= 0 ? 0 : cached(`p${Math.round(N / 16)}`, () => { const N16 = Math.max(1, Math.round(N / 16) * 16); const p = run({ users: 1, seq: N16 + 1, outputTokens: 1 }).perf; return (p.ttft - p.decodeStepMs) / 1000; });
  // One full weight pass: the decode part of a mixed step already streams the weights, so a chunk's marginal cost excludes
  // it. For MoE a full pass needs enough tokens to touch ~all routed experts (99%: n = ln 0.01 / ln(1 − k/E)); dense: 1.
  const fRoute = m.num_of_experts > 1 ? (m.num_of_active_experts || 2) / m.num_of_experts : 0;
  const nFull = fRoute ? Math.ceil(Math.log(0.01) / Math.log(1 - fRoute)) : 1;
  const P1 = cached('p1', () => { const p = run({ users: 1, seq: nFull + 1, outputTokens: 1 }).perf; return (p.ttft - p.decodeStepMs) / 1000; });
  const chunk = (nTok, c) => Math.max(0, prefill(c + nTok) - prefill(c) - (c === 0 ? P1 : 0));
  // Speculative decoding: LLMServingSim's acceptance_rate is the marginal accepted/drafted, mean accepted length
  // 1 + rate·k; solve this calculator's per-position α so (1 − α^(k+1)) / (1 − α) gives that same length.
  let spec = null;
  if (specK) {
    const sp = SPEC[modelName];
    if (!sp) throw new Error('no spec entry for ' + modelName);
    const target = 1 + sp.acceptance_rate * specK;
    let lo = 0.001, hi = 0.999;
    for (let i = 0; i < 60; i++) { const a = (lo + hi) / 2; ((1 - a ** (specK + 1)) / (1 - a) < target ? (lo = a) : (hi = a)); }
    const G = C.geometry ? C.geometry(m) : null;
    const draftB = (G ? G.P : Number(m.num_of_params) * 1e9) / m.num_of_layers / 1e9;   // MTP / EAGLE head ≈ one decoder layer
    spec = { k: specK, alpha: (lo + hi) / 2, draftB, method: sp.method, target };
  }
  const specStep = (B, c) => cached(`s${B}|${bucket(c)}`, () => {
    const p = run({ users: B, seq: bucket(c) + 2, outputTokens: 2, spec: true, specTokens: spec.k, specAccept: spec.alpha, specDraft: spec.draftB, specType: 'sequential' }).perf;
    return { dt: p.spec.tCycle - p.prefillAmortMs / 1000, E: p.spec.E };
  });
  // KV capacity per replica (tokens): memory grows linearly with tokens at fixed activations; fit the slope from two lengths.
  const mem = (s) => run({ users: 1, seq: s, outputTokens: 1 }, g).memory;
  const m1 = mem(16384), m2 = mem(32768);
  const slope = (m2.perGpu - m1.perGpu) / 16384, fixed = m1.perGpu - slope * 16384;   // GiB per GPU
  const usable = m1.capPerGpu / 0.9;   // calculator caps vLLM at 0.90 × usable; undo to apply the scenario's mem_util
  const capTokens = Math.max(0, Math.floor((memUtil * usable - fixed) / slope));
  return { decode, chunk, P1, spec, specStep, capTokens, weightsFit: fixed < memUtil * usable, label: `${m.name} ${quant} on ${n}× ${g.label} (TP${tp} PP${pp})` };
}

/* ---------------- vLLM-V1-like scheduler ---------------- */
class Engine {
  constructor(cm, o) {
    Object.assign(this, { cm, maxSeqs: o.maxSeqs, budget: o.budget, chunked: o.chunked, thresh: o.thresh, bs: o.bs, reserve: o.reserve, role: o.role || null });
    this.totalBlocks = Math.floor(cm.capTokens / o.bs);
    this.freeBlocks = this.totalBlocks;
    this.waiting = []; this.running = []; this.busyUntil = null; this.preemptions = 0; this.steps = 0;
  }
  blocksFor(t) { return Math.ceil(t / this.bs); }
  load() { return (this.waiting.length * 4 + this.running.length) / this.maxSeqs; }
  hasWork() { return this.waiting.length + this.running.length > 0; }
  grow(r, toks) {   // allocate blocks so r holds `toks` tokens
    const need = this.blocksFor(toks) - r.blocks;
    if (need <= 0) return true;
    if (need > this.freeBlocks) return false;
    this.freeBlocks -= need; r.blocks += need; return true;
  }
  ctxOf(r) { return r.done + r.gen - (r.genBase || 0); }   // tokens held in KV
  release(r) { this.freeBlocks += r.blocks; r.blocks = 0; }
  preemptLast(exceptR) {
    for (let i = this.running.length - 1; i >= 0; i--) {
      const v = this.running[i];
      if (v === exceptR) continue;
      this.running.splice(i, 1); this.release(v);
      this.recompute = (this.recompute || 0) + v.done; v.prompt = v.input + v.gen; v.done = 0; v.genBase = v.gen; v.preempted = (v.preempted || 0) + 1; this.preemptions++;
      this.waiting.unshift(v); return true;
    }
    return false;
  }
  // Build one step; returns { dt, apply } or null when idle.
  plan() {
    let budget = this.budget;
    const dec = [], pf = [];
    const specK = this.cm.spec ? this.cm.spec.k : 0;
    for (let i = 0; i < this.running.length && budget > 0; i++) {
      const r = this.running[i];
      if (r.done < r.prompt) {
        let n = Math.min(r.prompt - r.done, budget);
        if (this.thresh) n = Math.min(n, this.thresh);
        if (!this.chunked && n < r.prompt - r.done) continue;
        while (!this.grow(r, r.done + n)) { if (!this.preemptLast(r)) break; }
        if (r.blocks < this.blocksFor(r.done + n)) continue;
        pf.push([r, n]); budget -= n;
      } else {
        const need = 1 + specK;
        while (!this.grow(r, this.ctxOf(r) + need)) { if (!this.preemptLast(r)) break; }
        if (!this.running.includes(r)) continue;
        dec.push(r); budget -= need;
      }
    }
    while (this.waiting.length && this.running.length < this.maxSeqs && budget > 0) {
      const r = this.waiting[0];
      if (r.done >= r.prompt) {   // P/D handoff: KV arrives complete, decode only
        if (!this.grow(r, this.ctxOf(r) + 1)) break;
        this.waiting.shift(); this.running.push(r); continue;
      }
      let n = Math.min(r.prompt - r.done, budget);
      if (this.thresh) n = Math.min(n, this.thresh);
      if (!this.chunked && n < r.prompt - r.done) break;
      if (this.reserve && this.blocksFor(r.prompt + 1) - r.blocks > this.freeBlocks) break;
      if (!this.grow(r, r.done + n)) break;
      this.waiting.shift(); this.running.push(r); pf.push([r, n]); budget -= n;
    }
    if (!dec.length && !pf.length) return null;
    const ctxMean = dec.length ? dec.reduce((a, r) => a + this.ctxOf(r), 0) / dec.length : 0;
    let dt, E = 1;
    if (dec.length) {
      if (this.cm.spec) { const s = this.cm.specStep(dec.length, ctxMean); dt = s.dt; E = s.E; } else dt = this.cm.decode(dec.length, ctxMean);
    } else dt = this.cm.P1;
    for (const [r, n] of pf) dt += this.cm.chunk(n, r.done);
    return { dt, dec, pf, E };
  }
  apply(st, tEnd, out) {
    this.steps++;
    for (const [r, n] of st.pf) {
      r.done += n;
      if (r.done >= r.prompt) { if (r.first == null) r.first = tEnd; r.gen++; r.accum = 0; }
    }
    for (const r of st.dec) { r.accum = (r.accum || 0) + st.E; const k = Math.floor(r.accum); r.accum -= k; r.gen += k; }
    for (const r of [...st.pf.map((x) => x[0]), ...st.dec]) {
      if (!this.running.includes(r)) continue;
      if (this.role === 'prefill' && r.first != null) { this.running.splice(this.running.indexOf(r), 1); this.release(r); out.handoff.push(r); continue; }
      if (r.gen >= r.output) { r.gen = r.output; r.end = tEnd; this.running.splice(this.running.indexOf(r), 1); this.release(r); out.finished.push(r); }
    }
  }
}

/* ---------------- cluster simulation ---------------- */
function simulate(sc) {
  const engines = sc.instances.map((ins) => new Engine(ins.cm, ins.o));
  const groups = {};   // DP groups step in lockstep
  sc.instances.forEach((ins, i) => { if (ins.dp != null) (groups[ins.dp] = groups[ins.dp] || []).push(i); });
  const groupOf = engines.map((_, i) => Object.values(groups).find((g) => g.includes(i)) || [i]);
  const units = [...new Set(groupOf.map((g) => g.join(',')))].map((k) => k.split(',').map(Number));
  const arrivals = sc.requests.map((r, id) => ({ ...r, id })).sort((a, b) => a.arrival - b.arrival);
  const pending = [...arrivals];   // sorted by arrival; agentic follow-ups are inserted when their parent finishes
  const finished = [];
  let t = 0, rr = 0, rnd = 12345, ctr = { prefill: 0, decode: 0, any: 0 };
  const rand = () => (rnd = (rnd * 1103515245 + 12345) % 2147483648) / 2147483648;
  const route = (role) => {
    const idx = engines.map((e, i) => i).filter((i) => (role ? engines[i].role === role : engines[i].role !== 'decode'));
    if (sc.policy === 'RR') return idx[rr++ % idx.length];
    if (sc.policy === 'RAND') return idx[Math.floor(rand() * idx.length)];
    const key = role || 'any', start = ctr[key]++ % idx.length;
    let best = idx[start], bs = Infinity;
    for (let o = 0; o < idx.length; o++) { const i = idx[(start + o) % idx.length]; const s = engines[i].load(); if (s < bs) { bs = s; best = i; } }
    return best;
  };
  const transfers = [];   // P/D KV handoffs in flight: { at, r }
  const insertPending = (r) => { let i = pending.findIndex((x) => x.arrival > r.arrival); if (i < 0) i = pending.length; pending.splice(i, 0, r); };
  const unitBusy = new Map(); // unit key -> { until, plans }
  let guard = 0;
  for (;;) {
    if (++guard > 5e6) throw new Error('simulation did not terminate: ' + sc.name);
    // start steps on idle units with work
    for (const u of units) {
      const key = u.join(',');
      if (unitBusy.has(key)) continue;
      if (!u.some((i) => engines[i].hasWork())) continue;
      const plans = u.map((i) => engines[i].plan());
      if (plans.every((p) => !p)) continue;   // blocked (waiting for memory) — cannot progress until something frees
      // DP lockstep: an idle member runs a dummy step; the wave ends when the slowest member finishes.
      const dts = plans.map((p, j) => (p ? p.dt : engines[u[j]].cm.decode(1, 1)));
      unitBusy.set(key, { until: t + Math.max(...dts), plans, u });
    }
    const nextArr = pending.length ? pending[0].arrival : Infinity;
    const nextStep = Math.min(...[...unitBusy.values()].map((b) => b.until), Infinity);
    const nextXfer = transfers.length ? Math.min(...transfers.map((x) => x.at)) : Infinity;
    const tn = Math.min(nextArr, nextStep, nextXfer);
    if (tn === Infinity) break;
    t = tn;
    for (const [key, b] of [...unitBusy]) {
      if (b.until > t) continue;
      unitBusy.delete(key);
      const out = { finished: [], handoff: [] };
      b.plans.forEach((p, j) => p && engines[b.u[j]].apply(p, t, out));
      for (const r of out.finished) {
        finished.push(r);
        if (r.next && r.next.length) { const [nx, ...rest] = r.next; insertPending({ ...nx, arrival: t + nx.tool, next: rest, session: r.session, id: r.id + '.' }); }
      }
      for (const r of out.handoff) transfers.push({ at: t + sc.linkLat + r.input * sc.kvBytesPerTok / sc.linkBw, r });
    }
    for (let i = transfers.length - 1; i >= 0; i--) {
      if (transfers[i].at > t) continue;
      const { r } = transfers.splice(i, 1)[0];
      const e = engines[route('decode')];
      if (r.gen >= r.output) { r.end = t; finished.push(r); continue; }
      r.blocks = 0; r.done = r.prompt; e.waiting.push(r);
    }
    while (pending.length && pending[0].arrival <= t) {
      const a = pending.shift();
      const r = { id: a.id, arrival: a.arrival, input: a.input, output: a.output, prompt: a.input, done: 0, gen: 0, blocks: 0, first: null, end: null, next: a.next, session: a.session };
      engines[route(sc.pd ? 'prefill' : null)].waiting.push(r);
    }
  }
  const pre = engines.reduce((a, e) => a + e.preemptions, 0);
  const unfinished = arrivals.length - finished.filter((r) => !String(r.id).includes('.')).length;
  return { finished, makespan: Math.max(0, ...finished.map((r) => r.end)), preemptions: pre, unfinished, engines };
}

/* ---------------- scenario construction ---------------- */
function parseArgs(s) {
  const a = s.trim().split(/\s+/), o = { flags: [] };
  for (let i = 0; i < a.length; i++) {
    const k = a[i];
    const v = a[i + 1];
    if (k === '--cluster-config') o.cluster = v, i++;
    else if (k === '--dataset') o.dataset = v, i++;
    else if (k === '--num-reqs') o.numReqs = +v, i++;
    else if (k === '--npu-memory-utilization') o.memUtil = +v, i++;
    else if (k === '--max-num-batched-tokens') o.budget = +v, i++;
    else if (k === '--max-num-seqs') o.maxSeqs = +v, i++;
    else if (k === '--long-prefill-token-threshold') o.thresh = +v, i++;
    else if (k === '--block-size') o.bs = +v, i++;
    else if (k === '--num-speculative-tokens') o.specK = +v, i++;
    else if (k === '--request-routing-policy') o.policy = v, i++;
    else if (k === '--no-enable-chunked-prefill') o.chunked = false;
    else if (k === '--no-reserve-full-isl') o.reserve = false;
    else if (k === '--log-interval') i++;
    else if (k === '--prefix-storage' || k === '--expert-routing-policy') { o.flags.push(k + ' ' + v); i++; }
    else o.flags.push(k);
  }
  return o;
}
const loadWorkload = (f, n) => {
  const rows = fs.readFileSync(`${HERE}/workloads/${f}`, 'utf8').trim().split('\n').map((l) => JSON.parse(l)).slice(0, n || Infinity);
  return rows.map((r, i) => r.sub_requests
    ? { arrival: r.arrival_time_ns / 1e9, input: r.sub_requests[0].input_toks, output: r.sub_requests[0].output_toks, session: r.session_id,
        next: r.sub_requests.slice(1).map((s) => ({ input: s.input_toks, output: s.output_toks, tool: (s.tool_duration_ns || 0) / 1e9 })) }
    : { arrival: r.arrival_time_ns / 1e9, input: r.input_toks, output: r.output_toks });
};
const NOT_MODELED = { '--enable-prefix-sharing': 'CPU/CXL prefix pool', '--prefix-storage CPU': 'CPU prefix pool', '--prefix-storage CXL': 'CXL prefix pool',
  '--enable-local-offloading': 'local KV offloading', '--enable-attn-offloading': 'PIM attention offload', '--enable-sub-batch-interleaving': 'PIM sub-batch interleaving',
  '--expert-routing-policy RR': 'expert-routing policy', '--expert-routing-policy RAND': 'expert-routing policy', '--no-enable-block-copy': 'block-copy toggle',
  '--skip-prefill': 'TTFT accounting only', '--no-enable-prefix-caching': 'prefix-cache hits (not modeled either way)', '--enable-prefix-caching': 'prefix-cache hits' };

function buildScenario(name, args) {
  const a = parseArgs(args);
  const cfg = JSON.parse(fs.readFileSync(`${HERE}/clusters/${a.cluster}`, 'utf8'));
  return buildFromCluster(name, cfg, a, a.cluster);
}
function buildFromCluster(name, cfg, a, cfgName = '') {
  const notes = [...new Set(a.flags.map((f) => NOT_MODELED[f] || null).filter(Boolean))];
  const instances = [];
  cfg.nodes.forEach((node) => node.instances.forEach((ins) => {
    const tp = ins.tp_size || 1, pp = ins.pp_size || 1;
    const npus = ins.num_npus || tp * pp;
    const memUtil = a.memUtil ?? ins.npu_mem?.mem_util ?? 0.9;
    const bsDefault = (MODELS[ins.model_name] || [])[2] || 16;
    const cm = costModel(ins.model_name, ins.hardware, tp, pp, memUtil, a.specK);
    if (ins.ep_size && ins.ep_size > 1) notes.push(`EP${ins.ep_size} → modeled as ${tp > 1 ? 'TP' + tp : 'full experts per GPU'} (no EP in this calculator)`);
    if (ins.placement && JSON.stringify(ins.placement).match(/cxl|cpu"/) && /cxl/.test(JSON.stringify(ins.placement))) notes.push('weights on CXL memory');
    if (ins.placement && !/cxl/.test(JSON.stringify(ins.placement))) notes.push('per-block placement (NPU) — same as base');
    if (cfg.nodes.some((n) => n.power) || ins.power) notes.push('power model (clock unaffected)');
    if (/pim/.test(cfgName)) notes.push('PIM device');
    if (a.memUtil == null && ins.npu_mem?.mem_util != null && ins.npu_mem.mem_util !== 0.9) notes.push(`config mem_util ${+ins.npu_mem.mem_util.toFixed(4)} (tuned for LLMServingSim's zero-overhead memory model)`);
    instances.push({ cm, dp: ins.dp_group ?? null, o: { maxSeqs: a.maxSeqs ?? ins.max_num_seqs ?? 128, budget: a.budget ?? (ins.max_num_batched_tokens || (ins.max_num_batched_tokens === 0 ? Infinity : 2048)),
      chunked: a.chunked ?? ins.enable_chunked_prefill ?? true, thresh: a.thresh || 0, bs: a.bs || bsDefault, reserve: a.reserve ?? true, role: ins.pd_type || null }, ins });
  }));
  // DP + EP (each member TP1/PP1, experts sharded over the whole group): attention is replicated but every GPU holds 1/EP of
  // the experts. This calculator has no EP, so the group is modeled as ONE TP=EP instance serving all of the group's
  // sequences (same per-GPU expert bytes and pooled KV memory; TP all-reduces stand in for EP all-to-alls).
  const byGroup = {};
  instances.forEach((x, i) => { if (x.dp != null) (byGroup[x.dp] = byGroup[x.dp] || []).push(i); });
  for (const [gname, idx] of Object.entries(byGroup)) {
    const xs = idx.map((i) => instances[i]);
    const ep = xs[0].ins.ep_size || 1;
    if (ep > 1 && ep === xs.length && xs.every((x) => (x.ins.tp_size || 1) === 1 && (x.ins.pp_size || 1) === 1)) {
      const x0 = xs[0];
      const cm = costModel(x0.ins.model_name, x0.ins.hardware, ep, 1, a.memUtil ?? x0.ins.npu_mem?.mem_util ?? 0.9, a.specK);
      const merged = { ...x0, cm, dp: null, o: { ...x0.o, maxSeqs: x0.o.maxSeqs * xs.length } };
      for (const i of idx.sort((p, q) => q - p)) instances.splice(i, 1);
      instances.push(merged);
      for (let k = notes.length - 1; k >= 0; k--) if (/^EP/.test(notes[k])) notes.splice(k, 1);
      notes.push(`DP${xs.length}×EP${ep} → one TP${ep} instance (no EP in this calculator)`);
    }
  }
  const reqs = loadWorkload(a.dataset, a.numReqs);
  const first = instances[0];
  const [m0] = MODELS[first.ins.model_name];
  const kvTok = (() => { const r = C.calculate({ ...INF, quant: MODELS[first.ins.model_name][1], users: 1, seq: 2049, outputTokens: 1 }, m0, GPU(first.ins.hardware)); const s = r.memory.segments.find((x) => x.key === 'kv'); return s.gib * 2 ** 30 / 2049; })();
  const linkBw = (Array.isArray(cfg.link_bw) ? cfg.link_bw[0] : cfg.link_bw || 16) * 1e9, linkLat = (cfg.link_latency ?? 20000) * 1e-9;
  return { name, instances, requests: reqs, policy: a.policy || 'LOAD', pd: instances.some((i) => i.o.role), linkBw, linkLat, kvBytesPerTok: kvTok, notes: [...new Set(notes)] };
}

/* ---------------- statistics ---------------- */
const pct = (arr, p) => { const s = [...arr].sort((x, y) => x - y); if (!s.length) return NaN; const k = (s.length - 1) * p, f = Math.floor(k), c = Math.ceil(k); return s[f] + (s[c] - s[f]) * (k - f); };
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const stats = (a) => ({ mean: mean(a), median: pct(a, 0.5), p90: pct(a, 0.9), p95: pct(a, 0.95), p99: pct(a, 0.99) });
const metricsOf = (rows) => ({
  ttft: stats(rows.map((r) => r.ttft)), tpot: stats(rows.filter((r) => r.out > 1).map((r) => r.tpot)), latency: stats(rows.map((r) => r.lat)),
});
const dev = (a, b) => (a - b) / b * 100;
const f1 = (x) => (x >= 0 ? '+' : '') + x.toFixed(1) + '%';

/* ---------------- 1. real-vLLM references ---------------- */
const refs = [];
for (const hw of fs.readdirSync(HERE + '/references')) for (const mdl of fs.readdirSync(`${HERE}/references/${hw}`)) {
  const dir = `${HERE}/references/${hw}/${mdl}`;
  const meta = JSON.parse(fs.readFileSync(dir + '/meta.json', 'utf8'));
  const cfg = JSON.parse(fs.readFileSync(dir + '/cluster.json', 'utf8'));
  const truthRows = fs.readFileSync(dir + '/vllm_requests.jsonl', 'utf8').trim().split('\n').map((l) => JSON.parse(l));
  const off = Math.min(...truthRows.map((r) => r.queued_ts - r.arrival_time));
  const truth = truthRows.map((r) => { const arr = r.arrival_time + off; return { ttft: (r.first_token_ts - arr) * 1e3, tpot: r.output_toks > 1 ? (r.last_token_ts - r.first_token_ts) / (r.output_toks - 1) * 1e3 : NaN, lat: (r.last_token_ts - arr) * 1e3, out: r.output_toks }; });
  const ek = meta.engine_kwargs;
  // Predict the REAL engine: use the gpu_memory_utilization vLLM ran with, not the value LLMServingSim tuned to match its block count.
  const a = { flags: [], dataset: path.basename(meta.dataset_path), numReqs: truthRows.length, maxSeqs: ek.max_num_seqs, budget: ek.max_num_batched_tokens, bs: meta.kv_cache.block_size, memUtil: meta.kv_cache.gpu_memory_utilization };
  // KV capacity: the calculator's prediction vs the engine's recorded block count
  const sc = buildFromCluster(`${hw}/${mdl}`, cfg, a);
  const res = simulate(sc);
  const ours = res.finished.map((r) => ({ ttft: (r.first - r.arrival) * 1e3, tpot: r.output > 1 ? (r.end - r.first) / (r.output - 1) * 1e3 : NaN, lat: (r.end - r.arrival) * 1e3, out: r.output }));
  const theirs = {};
  for (const line of fs.readFileSync(dir + '/llmservingsim_summary.txt', 'utf8').split('\n')) {
    const mm = line.match(/^(TTFT|TPOT|Latency) (Mean|Median|P90|P95|P99)\s+([\d.]+)\s+([\d.]+)/);
    if (mm) theirs[`${mm[1].toLowerCase()}.${mm[2].toLowerCase()}`] = { vllm: +mm[3], sim: +mm[4] };
  }
  const perEngine = meta.kv_cache.per_engine;
  const merged = sc.notes.some((x) => /^DP\d+×EP/.test(x));   // DP group pooled into one instance: compare against the sum
  const measuredTok = (perEngine ? (merged ? perEngine.reduce((t, e) => t + e.num_gpu_blocks, 0) : perEngine[0].num_gpu_blocks) : meta.kv_cache.num_gpu_blocks / Math.max(1, ek.data_parallel_size)) * meta.kv_cache.block_size;
  refs.push({ name: `${hw}/${mdl}`, config: sc.instances.map((i) => i.cm.label).join(' + '), notes: sc.notes, truth: metricsOf(truth), ours: metricsOf(ours), theirs,
    kv: { measuredTokens: measuredTok, oursTokens: sc.instances[0].cm.capTokens, util: meta.kv_cache.gpu_memory_utilization, simUtil: cfg.nodes[0].instances[0].npu_mem?.mem_util },
    makespan: { truth: Math.max(...truthRows.map((r) => r.last_token_ts)) - Math.min(...truthRows.map((r) => r.arrival_time + off)), ours: res.makespan - Math.min(...sc.requests.map((r) => r.arrival)) },
    preemptions: res.preemptions, recompute: res.engines.reduce((t, e) => t + (e.recompute || 0), 0), unfinished: res.unfinished });
}

/* ---------------- 2. LLMServingSim regression scenarios ---------------- */
const baselines = Object.fromEntries(fs.readFileSync(HERE + '/baselines.txt', 'utf8').split('\n').filter((l) => l && !l.startsWith('#') && !l.includes(':')).map((l) => l.split(/\s+/)).map(([k, v]) => [k, +v]));
const scen = [];
for (const line of fs.readFileSync(HERE + '/scenarios.txt', 'utf8').trim().split('\n')) {
  const [name, args] = line.split('|');
  try {
    const sc = buildScenario(name, args);
    const fitsAll = sc.instances.every((i) => i.cm.weightsFit && i.cm.capTokens > 0);
    let res = fitsAll ? simulate(sc) : null;
    const maxNeed = Math.max(...sc.requests.map((r) => r.input + 1));
    const minCap = Math.min(...sc.instances.map((i) => i.cm.capTokens));
    if (res && res.unfinished === sc.requests.length) { sc.notes.push(`does not run: KV pool ${minCap.toLocaleString()} tokens < first prompt (vLLM refuses to start)`); res = null; }
    scen.push({ name, config: [...new Set(sc.instances.map((i) => i.cm.label))].join(' + ') + (sc.instances.length > 1 ? ` ×${sc.instances.length}` : ''), requests: sc.requests.length,
      theirsNs: baselines[name], oursNs: res ? Math.round(res.makespan * 1e9) : null, dev: res ? dev(res.makespan * 1e9, baselines[name]) : null,
      preemptions: res ? res.preemptions : null, unfinished: res ? res.unfinished : null, notes: sc.notes, fits: fitsAll, kvTokens: sc.instances[0].cm.capTokens });
  } catch (e) { scen.push({ name, error: e.message }); }
}

/* ---------------- report ---------------- */
if (JSON_OUT) { console.log(JSON.stringify({ refs, scen }, null, 1)); process.exit(0); }
console.log('\n=== A. Against REAL vLLM (300 ShareGPT requests, ~10 req/s) ===');
for (const r of refs) {
  console.log(`\n${r.name} — ${r.config}${r.notes.length ? '  [' + r.notes.join('; ') + ']' : ''}`);
  console.log(`  KV capacity per engine: measured ${r.kv.measuredTokens.toLocaleString()} tokens (util ${r.kv.util}) | ours ${r.kv.oursTokens.toLocaleString()} at util ${r.kv.util} (${f1(dev(r.kv.oursTokens, r.kv.measuredTokens))})`);
  console.log(`  ${'metric'.padEnd(16)}${'vLLM'.padStart(10)}${'ours'.padStart(10)}${'Δ ours'.padStart(9)}${'LLMServingSim Δ'.padStart(17)}`);
  for (const k of ['ttft', 'tpot', 'latency']) for (const s of ['mean', 'median', 'p90', 'p99']) {
    const tv = r.truth[k][s], ov = r.ours[k][s], th = r.theirs[`${k}.${s}`];
    console.log(`  ${(k.toUpperCase() + ' ' + s).padEnd(16)}${tv.toFixed(1).padStart(10)}${ov.toFixed(1).padStart(10)}${f1(dev(ov, tv)).padStart(9)}${(th ? f1(dev(th.sim, th.vllm)) : '—').padStart(17)}`);
  }
  console.log(`  makespan: vLLM ${r.makespan.truth.toFixed(1)} s | ours ${r.makespan.ours.toFixed(1)} s (${f1(dev(r.makespan.ours, r.makespan.truth))}); preemptions ${r.preemptions} (${r.recompute.toLocaleString()} tokens recomputed); unfinished ${r.unfinished}`);
}
console.log('\n=== B. LLMServingSim regression scenarios: total clock (their simulator) vs this calculator ===');
console.log(`${'scenario'.padEnd(26)}${'reqs'.padStart(5)}${'theirs s'.padStart(11)}${'ours s'.padStart(10)}${'Δ'.padStart(9)}  config / notes`);
for (const s of scen) {
  if (s.error) { console.log(`${s.name.padEnd(26)}  ERROR ${s.error}`); continue; }
  const th = s.theirsNs / 1e9, ou = s.oursNs != null ? s.oursNs / 1e9 : null;
  console.log(`${s.name.padEnd(26)}${String(s.requests).padStart(5)}${th.toFixed(2).padStart(11)}${(ou != null ? ou.toFixed(2) : 'n/fit').padStart(10)}${(s.dev != null ? f1(s.dev) : '—').padStart(9)}  ${s.config}${s.preemptions ? ` · ${s.preemptions} preemptions` : ''}${s.unfinished ? ` · ${s.unfinished} unfinished` : ''}${s.notes.length ? '  [' + s.notes.join('; ') + ']' : ''}`);
}
const ok = scen.filter((s) => s.dev != null);
const abs = ok.map((s) => Math.abs(s.dev)).sort((a, b) => a - b);
console.log(`\n${ok.length}/${scen.length} scenarios simulated; |Δ| median ${pct(abs, 0.5).toFixed(1)}%, within ±10%: ${abs.filter((x) => x <= 10).length}, within ±25%: ${abs.filter((x) => x <= 25).length}`);
