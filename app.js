/* GPU Simulator — UI. All math lives in calc.js; this file only collects inputs and renders results. */
(function () {
  'use strict';
  const D = window.VRAM_DATA;
  const C = window.GpuCalc;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const pf = (x) => Number.parseFloat(x) || 0;

  /* ---------------- data ---------------- */
  const MODELS = D.models;
  const GPUS = D.gpus.filter((g) => !g.key.startsWith('custom') && g.released !== false); // shipping hardware only
  const modelBySlug = Object.fromEntries(MODELS.map((m) => [m.slug, m]));
  const gpuByKey = Object.fromEntries(GPUS.map((g) => [g.key, g]));
  const isDiff = (m) => m.task_type === 'image_generation' || m.task_type === 'video_generation';
  const isApple = (g) => g.type === 'apple_silicon';
  const totalParams = (m) => Number(m.num_of_params || 0) + Number(m.num_of_unread_params || 0); // incl. vision tower / lookup tables
  const nativeBits = (m) => { const p = (m.base_precision || 'fp16').toLowerCase(); return p.includes('fp32') ? 32 : p.includes('fp8') ? 8 : p.includes('fp4') ? 4.25 : 16; };
  const nativeQuant = (m) => { const p = (m.base_precision || '').toLowerCase(); return p.includes('fp8') ? 'fp8' : p.includes('mxfp4') ? 'mxfp4' : p.includes('fp32') ? 'fp32' : 'fp16'; };

  // Colors live in index.html as --seg-* variables so they adapt to light and dark themes.
  const SEG_COLORS = Object.fromEntries(["weights", "kv", "act", "draft", "ctx", "graph", "comm", "frag", "lora", "grads", "optim", "logits", "qbuf", "enc", "vae"].map((k) => [k, `var(--seg-${k})`]));
  const ELEC = [{ k: 'US avg', v: 0.16 }, { k: 'EU avg', v: 0.28 }, { k: 'China', v: 0.08 }, { k: 'Cloud DC', v: 0.08 }];
  const CARBON = [[400, 'Global average'], [370, 'United States'], [240, 'European Union'], [560, 'China'], [60, 'France'], [690, 'India'], [450, 'Japan'], [100, 'Brazil'], [12, 'Nuclear'], [11, 'Wind'], [45, 'Solar'], [24, 'Hydro'], [490, 'Natural gas'], [950, 'Coal']];
  const STATUS_LABEL = { good: 'Comfortable fit', okay: 'Fits', moderate: 'Tight fit', high: 'Very tight', very_high: 'At the limit', insufficient: 'Does not fit' };

  /* ---------------- state ---------------- */
  const OPT_DEFAULTS = { optimizer_type: 'adamw', zero_stage: 3, flash_attention: true, gradient_checkpointing: true, gradient_checkpointing_ratio: 1, use_8bit_optimizer: false, use_bf16_true: false, use_fused_kernels: true, sequence_packing: true, dynamic_padding: false, activation_offloading: false, offload_optimizer: false, offload_gradients: false, offload_param: false };
  const state = {
    mode: 'inference', model: 'llama-3-1-8b', quant: 'fp16', kvQuant: 'fp16', runtime: 'auto', diffRuntime: 'diffusers',
    gpu: '4090_24', customApple: false, custom: { memory: 24, bandwidth_gbs: 1000, fp16_tflops: 165, tdp_watts: 350 },
    cluster: false, numGpus: 1, numNodes: 2, gpusPerNode: 8, tpDegree: 8, ppDegree: 2, interconnect: 'pcie4', interNode: 'infiniband_ndr', parallelism: 'tensor',
    seq: 8192, outputTokens: 512, batch: 1, users: 1, prefixShare: 0,
    offload: false, offloadTarget: 'cpu_ram', offloadPct: 25, offloadKv: false,
    spec: false, specDraft: 1, specAccept: 0.7, specTokens: 4, specType: 'sequential',
    ftMethod: 'lora', ftQuant: 'fp16', loraRank: 16, gradSteps: 4, numSamples: 10000, tokensPerSample: 512, numEpochs: 3, ftSeq: 2048, ftBatch: 2,
    opt: { ...OPT_DEFAULTS },
    imageWidth: 1024, imageHeight: 1024, numFrames: 49, inferenceSteps: 28, guidanceScale: 3.5, encoderOffload: true, vaeTiling: false, diffBatch: 1,
    elec: 0.16, carbon: 400,
    tab: 'breakdown', view: 'per', open: { adv: false, opt: false },
  };
  let result = null;

  const model = () => modelBySlug[state.model] || MODELS[0];
  const gpu = () => {
    if (state.gpu !== 'custom') return gpuByKey[state.gpu] || GPUS[0];
    const c = state.custom;
    return { key: 'custom', label: state.customApple ? 'Custom Apple Silicon' : 'Custom GPU', type: state.customApple ? 'apple_silicon' : 'discrete', vendor: state.customApple ? 'apple' : 'nvidia', category: 'consumer', memory: c.memory, bandwidth_gbs: c.bandwidth_gbs, fp16_tflops: c.fp16_tflops, tdp_watts: c.tdp_watts, hourly_price_usd: null };
  };
  const totalGpus = () => (state.cluster ? state.numNodes * state.gpusPerNode : state.numGpus);
  const maxCtx = () => { const c = model().context_length || 131072; return state.mode === 'finetuning' ? Math.min(c, 131072) : c; };

  function quantOptions(m) {
    const nb = nativeBits(m);
    const diff = isDiff(m);
    const groups = {};
    for (const [k, q] of Object.entries(C.QUANTS)) {
      if (q.bpw > nb + 0.01 && k !== nativeQuant(m)) continue;
      if (diff && (q.gguf || k === 'mxfp4' || k === 'int4')) continue;
      (groups[q.group] ||= []).push([k, `${q.label}${k === nativeQuant(m) ? '  · native' : ''}`]);
    }
    return groups;
  }
  function runtimeOptions(g) {
    const all = Object.entries(C.RUNTIMES);
    if (isApple(g)) return all.filter(([k]) => ['mlx', 'llamacpp', 'ollama', 'hf'].includes(k));
    if (g.type === 'apu') return all.filter(([k]) => ['llamacpp', 'ollama', 'vllm', 'hf'].includes(k));
    return all.filter(([k]) => k !== 'mlx');
  }
  function linkOptions(g, interOnly) {
    const L = C.LINKS;
    const k = g.key;
    const v = isApple(g) ? 'apple' : g.type === 'apu' ? 'apu' : g.vendor;
    const dc = g.category === 'datacenter';
    const nv = { nvlink_gen6: ['r100'], nvlink_gen5: ['b100', 'b200', 'b300'], nvlink_gen4: ['h100', 'h200', 'h800', 'h20'], nvlink_gen3: ['a100', 'a800'], nvlink_gen2: ['rtx_a5000', 'rtx_a6000', 'a30', 'a40'], nvlink_bridge: ['3090'] };
    return Object.keys(L).filter((id) => {
      if (interOnly) return id.startsWith('infiniband') || id.startsWith('ethernet');
      if (v === 'apple') return id.startsWith('thunderbolt') || id.startsWith('ethernet');
      if (v === 'apu') return id.startsWith('ethernet');
      if (id.startsWith('nvlink')) return v === 'nvidia' && (k === 'custom' || (nv[id] || []).some((p) => k.startsWith(p)));
      if (id === 'amd_xgmi') return v === 'amd' && dc;
      if (id.startsWith('infiniband')) return dc || k === 'custom';
      if (id.startsWith('thunderbolt')) return false;
      if (id === 'pcie5') return /^(50|b|h1|h2|rtx_\d+_blackwell|mi3|rx_9|custom)/.test(k);
      return true;
    }).map((id) => [id, L[id].label]);
  }
  function defaultLink(g) {
    const o = linkOptions(g).map((x) => x[0]);
    return ['nvlink_gen6', 'nvlink_gen5', 'nvlink_gen4', 'nvlink_gen3', 'amd_xgmi', 'nvlink_gen2', 'thunderbolt5', 'pcie5', 'pcie4', 'ethernet100g'].find((x) => o.includes(x)) || o[0];
  }

  /* ---------------- engine input ---------------- */
  function engineInput() {
    const inf = state.mode === 'inference';
    const m = model();
    const diff = isDiff(m);
    return {
      mode: diff ? 'inference' : state.mode, quant: state.quant, kvQuant: state.kvQuant, runtime: diff ? state.diffRuntime : state.runtime,
      cluster: state.cluster, numGpus: state.numGpus, numNodes: state.numNodes, gpusPerNode: state.gpusPerNode, tpDegree: state.tpDegree, ppDegree: state.ppDegree,
      interconnect: state.interconnect, interNode: state.interNode, parallelism: state.parallelism,
      seq: inf ? state.seq : state.ftSeq, outputTokens: state.outputTokens, batch: diff ? state.diffBatch : inf ? state.batch : state.ftBatch,
      users: inf ? state.users : 1, prefixShare: state.prefixShare / 100,
      offload: inf && state.offload, offloadTarget: state.offloadTarget, offloadFrac: state.offloadPct / 100, offloadKv: inf && state.offload && state.offloadKv,
      spec: inf && state.spec, specDraft: state.specDraft, specAccept: state.specAccept, specTokens: state.specTokens, specType: state.specType,
      ftMethod: state.ftMethod, ftQuant: state.ftQuant, loraRank: state.loraRank, gradSteps: state.gradSteps, numSamples: state.numSamples, tokensPerSample: state.tokensPerSample, numEpochs: state.numEpochs,
      opt: { ...state.opt, zero_stage: totalGpus() > 1 ? state.opt.zero_stage : 0 },
      imageWidth: state.imageWidth, imageHeight: state.imageHeight, numFrames: state.numFrames, inferenceSteps: state.inferenceSteps, guidanceScale: state.guidanceScale, encoderOffload: state.encoderOffload, vaeTiling: state.vaeTiling,
      elec: state.elec, carbon: state.carbon,
    };
  }

  /* ---------------- formatting ---------------- */
  const f1 = (x) => Number(x).toLocaleString('en-US', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
  const gb = (x) => `${x >= 100 ? Math.round(x).toLocaleString('en-US') : x.toFixed(x >= 10 ? 1 : 2)} GB`;
  const fmtRate = (x) => (x >= 100 ? Math.round(x).toLocaleString('en-US') : x >= 10 ? x.toFixed(1) : x.toFixed(2));
  const fmtMs = (ms) => (ms >= 60000 ? `${(ms / 60000).toFixed(1)} min` : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
  const fmtDur = (h) => (h < 1 / 60 ? `${Math.max(1, Math.round(h * 3600))} s` : h < 1 ? `${Math.round(h * 60)} min` : h < 48 ? `${h.toFixed(1)} h` : `${(h / 24).toFixed(1)} days`);
  const fmtTok = (n) => (n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(Math.round(n)));
  const fmtCtx = (n) => (n >= 1048576 ? `${+(n / 1048576).toFixed(1)}M` : n >= 1024 ? `${+(n / 1024).toFixed(1)}K` : String(n));
  const usd = (x, d = 2) => `$${x.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })}`;

  /* ---------------- input widgets ---------------- */
  const opt = (list, cur) => list.map(([v, l]) => `<option value="${esc(v)}"${String(v) === String(cur) ? ' selected' : ''}>${esc(l)}</option>`).join('');
  const grouped = (groups, cur) => Object.entries(groups).map(([g, items]) => `<optgroup label="${esc(g)}">${opt(items, cur)}</optgroup>`).join('');
  function field(label, inner, help, val) {
    return `<div class="field"><div class="lbl">${label}${val != null ? `<span class="val">${val}</span>` : ''}</div>${inner}${help ? `<div class="help">${help}</div>` : ''}</div>`;
  }
  const select = (k, list, cur, extra = '') => `<select class="input" data-k="${k}"${extra}>${list}</select>`;
  const number = (k, v, min, max, step = 1) => `<input class="input num" type="number" data-k="${k}" value="${v}" min="${min}" max="${max}" step="${step}">`;
  function range(k, v, min, max, step, ticks, valueText) {
    const p = max > min ? ((v - min) / (max - min)) * 100 : 0;
    return `<input type="range" data-k="${k}" min="${min}" max="${max}" step="${step}" value="${v}" style="--p:${p}%" aria-valuetext="${esc(valueText ?? v)}">${ticks ? `<div class="ticks">${ticks.map((t) => `<span>${t}</span>`).join('')}</div>` : ''}`;
  }
  function slider(label, k, v, min, max, step, ticks, help, display) {
    return field(label, range(k, v, min, max, step, ticks, display), help, `<span id="v-${k.replace('.', '-')}">${display ?? v}</span>`);
  }
  const toggle = (k, label, on, dis) => `<label class="toggle${dis ? ' dis' : ''}"><input type="checkbox" data-k="${k}"${on ? ' checked' : ''}${dis ? ' disabled' : ''}><span class="sw"></span><span>${label}</span></label>`;

  // context-length slider works on a list of "nice" values
  function ctxSteps(max) {
    const base = [512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144, 393216, 524288, 786432, 1048576, 1572864, 2097152, 3145728, 4194304, 6291456, 8388608, 10485760];
    const out = base.filter((v) => v < max);
    out.push(max);
    return out;
  }

  /* ---------------- searchable picker ---------------- */
  let pop = null;
  function modelGroups() {
    const order = { text_generation: 'Text generation', image_generation: 'Image generation', video_generation: 'Video generation', embedding: 'Embedding', decision: 'Classification' };
    const g = {};
    for (const m of MODELS) (g[order[m.task_type] || 'Text generation'] ||= []).push({ v: m.slug, l: m.name, s: `${m.provider || ''} · ${pf(totalParams(m))}B${m.architecture === 'moe' ? ' MoE' : ''}` });
    return Object.values(order).filter((k) => g[k]).map((k) => ({ group: k, items: g[k] }));
  }
  function gpuGroups() {
    const order = []; const g = {};
    const name = (x) => x.type === 'apple_silicon' ? 'Apple Silicon' : x.type === 'apu' ? 'Unified-memory APUs' : `${{ nvidia: 'NVIDIA', amd: 'AMD', intel: 'Intel' }[x.vendor] || x.vendor} ${x.category === 'datacenter' ? 'datacenter' : 'desktop & workstation'}`;
    for (const x of GPUS) { const n = name(x); if (!g[n]) { g[n] = []; order.push(n); } g[n].push({ v: x.key, l: x.label, s: `${x.bandwidth_gbs.toLocaleString()} GB/s${x.hourly_price_usd ? ` · $${x.hourly_price_usd}/h` : ''}` }); }
    const out = order.map((n) => ({ group: n, items: g[n] }));
    out.push({ group: 'Custom', items: [{ v: 'custom', l: 'Custom GPU…', s: 'enter your own specs' }, { v: 'custom-apple', l: 'Custom Apple Silicon…', s: 'unified memory' }] });
    return out;
  }
  function openPicker(btn) {
    closePicker();
    const id = btn.dataset.pick;
    const groups = id === 'model' ? modelGroups() : gpuGroups();
    const cur = id === 'model' ? state.model : state.gpu === 'custom' ? (state.customApple ? 'custom-apple' : 'custom') : state.gpu;
    pop = document.createElement('div');
    pop.className = 'pop';
    pop.innerHTML = `<input class="input" placeholder="${id === 'model' ? `Search ${MODELS.length} models…` : 'Search GPUs, e.g. 4090, H100, M4…'}" aria-label="Search"><div class="pop-list" role="listbox"></div>`;
    document.body.appendChild(pop);
    const r = btn.getBoundingClientRect();
    pop.style.left = `${Math.max(8, r.left + scrollX)}px`;
    pop.style.top = `${r.bottom + scrollY + 6}px`;
    pop.style.width = `${Math.max(r.width, 340)}px`;
    const list = $('.pop-list', pop), inp = $('input', pop);
    let kb = -1;
    const draw = () => {
      const toks = inp.value.toLowerCase().split(/\s+/).filter(Boolean);
      list.innerHTML = groups.map((g) => {
        const items = g.items.filter((i) => toks.every((t) => `${i.l} ${i.s} ${i.v}`.toLowerCase().includes(t)));
        return items.length ? `<div class="pop-group">${esc(g.group)}</div>${items.map((i) => `<div class="pop-item${i.v === cur ? ' sel' : ''}" role="option" data-v="${esc(i.v)}"><span>${esc(i.l)}</span><small>${esc(i.s)}</small></div>`).join('')}` : '';
      }).join('') || '<div class="pop-empty">No matches</div>';
      kb = -1;
    };
    draw();
    inp.addEventListener('input', draw);
    inp.addEventListener('keydown', (e) => {
      const items = $$('.pop-item', list);
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        kb = clamp(kb + (e.key === 'ArrowDown' ? 1 : -1), 0, items.length - 1);
        items.forEach((x, i) => x.classList.toggle('kb', i === kb));
        items[kb]?.scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter') { const it = items[kb >= 0 ? kb : 0]; if (it) choose(it.dataset.v); }
    });
    list.addEventListener('click', (e) => { const it = e.target.closest('.pop-item'); if (it) choose(it.dataset.v); });
    const choose = (v) => { closePicker(); if (id === 'model') setModel(v); else setGpu(v); btn.focus(); };
    setTimeout(() => inp.focus(), 0);
    $('.pop-item.sel', list)?.scrollIntoView({ block: 'center' });
  }
  function closePicker() { if (pop) { pop.remove(); pop = null; } }
  document.addEventListener('mousedown', (e) => { if (pop && !pop.contains(e.target) && !e.target.closest('[data-pick]')) closePicker(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePicker(); });
  window.addEventListener('resize', closePicker);

  function setModel(slug) {
    state.model = slug;
    const m = model();
    const allowed = Object.values(quantOptions(m)).flat().map((x) => x[0]);
    if (!allowed.includes(state.quant)) state.quant = nativeQuant(m);
    if (isDiff(m) && state.mode === 'finetuning') state.mode = 'inference';
    state.seq = Math.min(state.seq, maxCtx());
    state.ftSeq = Math.min(state.ftSeq, maxCtx());
    renderAll();
  }
  function setGpu(v) {
    if (v === 'custom' || v === 'custom-apple') { state.gpu = 'custom'; state.customApple = v === 'custom-apple'; }
    else state.gpu = v;
    const g = gpu();
    if (isApple(g)) { state.cluster = false; state.offloadKv = false; if (state.offloadTarget === 'cpu_ram') state.offloadTarget = 'nvme'; }
    if (!runtimeOptions(g).some(([k]) => k === state.runtime)) state.runtime = 'auto';
    if (!linkOptions(g).some(([k]) => k === state.interconnect)) state.interconnect = defaultLink(g);
    if (!linkOptions(g, true).some(([k]) => k === state.interNode)) state.interNode = linkOptions(g, true)[0]?.[0];
    renderAll();
  }

  /* ---------------- left column ---------------- */
  function renderInputs() {
    const m = model(), g = gpu(), diff = isDiff(m), inf = state.mode === 'inference' || diff, n = totalGpus();
    $$('#mode-seg button').forEach((b) => { b.classList.toggle('on', b.dataset.mode === (diff ? 'inference' : state.mode)); b.disabled = diff && b.dataset.mode === 'finetuning'; b.title = b.disabled ? 'Fine-tuning estimates cover text models' : ''; });
    $('#eyebrow').textContent = diff ? (m.task_type === 'video_generation' ? 'Video diffusion inference' : 'Image diffusion inference') : inf ? 'LLM inference' : `${{ full: 'Full', lora: 'LoRA', qlora: 'QLoRA' }[state.ftMethod]} fine-tuning`;
    const G = C.geometry(m);
    const inferred = Object.entries(G.inferred).filter(([k, v]) => v && ['hd', 'kvh', 'ffn', 'V'].includes(k)).map(([k]) => ({ hd: 'head dim', kvh: 'KV heads', ffn: 'FFN size', V: 'vocab' }[k]));
    const resolvedRt = result?.runtimeLabel;

    let h = '';
    // ---------- 1. Model ----------
    h += `<div class="card"><div class="card-h"><h2><span class="step">1</span>Model</h2><span class="hint">${MODELS.length} models</span></div><div class="card-b">`;
    if (!inf) h += field('Method', `<div class="seg block">${[['full', 'Full'], ['lora', 'LoRA'], ['qlora', 'QLoRA']].map(([v, l]) => `<button data-set="ftMethod" data-v="${v}" class="${state.ftMethod === v ? 'on' : ''}">${l}</button>`).join('')}</div>`, { full: 'Updates every weight. Needs weights + gradients + optimizer states (≈16 bytes/param with AdamW).', lora: 'Freezes the base model in BF16 and trains small low-rank adapters.', qlora: 'Like LoRA, but the frozen base model is stored in 4-bit NF4.' }[state.ftMethod]);
    h += `<div class="field"><div class="lbl">Model</div><button type="button" class="combo" data-pick="model"><span class="c-main"><div class="c-title">${esc(m.name)}</div><div class="c-sub">${esc(m.provider || '')} · ${pf(totalParams(m))}B params${G.moe ? ` · ${(G.A / 1e9).toFixed(1)}B active` : ''}${m.context_length ? ` · ${fmtCtx(m.context_length)} context` : ''}</div></span><span class="chev">⌄</span></button>
      <div class="chips">${diff ? `<span class="chip">${m.task_type === 'video_generation' ? 'Video' : 'Image'} diffusion</span>` : `<span class="chip">${G.moe ? `MoE ${m.num_of_active_experts || '?'}/${m.num_of_experts || '?'} experts` : 'Dense'}</span><span class="chip">${esc((m.attention_structure || 'MHA').toUpperCase())}${!G.mla ? ` · ${G.kvh} KV heads` : ''}</span>${m.uses_sliding_window && m.sliding_window_size && !G.mla ? `<span class="chip">Sliding window ${m.sliding_window_size.toLocaleString()}</span>` : ''}${m.uses_linear_attention ? `<span class="chip">Linear attn ${Math.round((m.linear_attention_ratio ?? 0.75) * 100)}%</span>` : ''}`}<span class="chip">Native ${esc((m.base_precision || 'fp16').toUpperCase())}</span>${m.num_of_unread_params ? `<span class="chip" title="Loaded into memory but not read for every generated token">+${pf(m.num_of_unread_params)}B vision / lookup</span>` : ''}${inferred.length && !diff ? `<span class="chip warn" title="Missing from the dataset; estimated from hidden size and parameter count">${esc(inferred.join(', '))} estimated</span>` : ''}</div></div>`;
    if (inf) {
      h += `<div class="cols2">${field('Weight precision', select('quant', grouped(quantOptions(m), state.quant), state.quant), `${C.QUANTS[state.quant]?.bpw} bits per weight including scales`)}`;
      h += diff ? field('Pipeline', select('diffRuntime', opt([['diffusers', 'Diffusers (PyTorch)'], ['tensorrt', 'TensorRT']], state.diffRuntime), state.diffRuntime))
        : field('KV-cache precision', select('kvQuant', opt(Object.entries(C.KV_LABELS), state.kvQuant), state.kvQuant), 'Lower precision fits longer contexts');
      h += '</div>';
      if (!diff) {
        const rts = runtimeOptions(g);
        h += field('Inference engine', select('runtime', opt([['auto', `Auto${resolvedRt && state.runtime === 'auto' ? ` — ${resolvedRt}` : ''}`], ...rts.map(([k, r]) => [k, r.label])], state.runtime), state.runtime), esc(C.RUNTIMES[result?.runtime || 'vllm']?.desc || ''));
      }
    } else if (state.ftMethod === 'full') {
      h += field('Training precision', select('ftQuant', opt([['fp16', 'BF16 / FP16 mixed precision'], ['fp32', 'FP32'], ['fp8', 'FP8 (Transformer Engine)']], state.ftQuant), state.ftQuant));
    } else {
      h += `<div class="cols2">${field('Base weights', `<div class="input static">${state.ftMethod === 'qlora' ? 'NF4 4-bit (frozen)' : 'BF16 (frozen)'}</div>`)}${field('LoRA rank', number('loraRank', state.loraRank, 1, 512), 'Adapters on q, k, v, o, gate, up, down')}</div>`;
    }
    h += '</div></div>';

    // ---------- 2. Hardware ----------
    h += `<div class="card"><div class="card-h"><h2><span class="step">2</span>Hardware</h2><div class="seg sm"><button data-set="cluster" data-v="0" class="${!state.cluster ? 'on' : ''}">Single machine</button><button data-set="cluster" data-v="1" class="${state.cluster ? 'on' : ''}"${isApple(g) ? ' disabled title="Cluster mode is for datacenter GPUs"' : ''}>Cluster</button></div></div><div class="card-b">`;
    const tfl = C.effTflops(g), dbw = C.deviceBw(g), us = C.usableGiB(g);
    h += `<div class="field"><div class="lbl">GPU</div><button type="button" class="combo" data-pick="gpu"><span class="c-main"><div class="c-title">${esc(g.label)}</div><div class="c-sub">${g.memory} GB · ${g.bandwidth_gbs.toLocaleString()} GB/s · ${g.fp16_tflops} TFLOPS FP16${g.hourly_price_usd ? ` · $${g.hourly_price_usd}/h` : ''}</div></span><span class="chev">⌄</span></button>
      <div class="chips">${us.v < g.memory ? `<span class="chip" title="${esc(us.note)}">${f1(us.v)} GB GPU-usable</span>` : ''}${tfl.v !== g.fp16_tflops ? `<span class="chip" title="${esc(tfl.note)}">${f1(tfl.v)} TFLOPS effective</span>` : ''}${dbw.v < 1 ? `<span class="chip" title="${esc(dbw.note)}">~${Math.round(dbw.v * 100)}% bandwidth reachable</span>` : ''}</div></div>`;
    if (state.gpu === 'custom') {
      h += `<div class="cols4">${field('Memory (GB)', number('custom.memory', state.custom.memory, 1, 4096))}${field('Bandwidth (GB/s)', number('custom.bandwidth_gbs', state.custom.bandwidth_gbs, 1, 100000))}${field('BF16 TFLOPS', number('custom.fp16_tflops', state.custom.fp16_tflops, 0.1, 100000, 0.1))}${field('Power (W)', number('custom.tdp_watts', state.custom.tdp_watts, 1, 5000))}</div><div class="help" style="margin:-8px 0 14px">Use dense tensor TFLOPS with FP32 accumulation (what PyTorch actually gets), not the sparsity figure.</div>`;
    }
    if (!state.cluster) {
      h += `<div class="cols2">${field('Number of GPUs', number('numGpus', state.numGpus, 1, isApple(g) ? 16 : 64))}`;
      if (n > 1) h += inf && !diff ? field('Parallelism', select('parallelism', opt([['tensor', 'Tensor parallel (split each layer)'], ['pipeline', 'Pipeline parallel (split layers)'], ['replicated', 'Data parallel (copy per GPU)']], state.parallelism), state.parallelism))
        : field('Interconnect', select('interconnect', opt(linkOptions(g), state.interconnect), state.interconnect));
      h += '</div>';
      if (n > 1 && inf && !diff) h += field('Interconnect', select('interconnect', opt(linkOptions(g), state.interconnect), state.interconnect), { tensor: 'Tensor parallel all-reduces twice per layer — fast links matter most here.', pipeline: 'Pipeline parallel adds capacity, not single-stream speed: each token still passes every layer in sequence.', replicated: 'Each GPU holds a full copy and serves its share of the users.' }[state.parallelism]);
    } else {
      h += `<div class="cols4">${field('Nodes', number('numNodes', state.numNodes, 1, 64))}${field('GPUs / node', number('gpusPerNode', state.gpusPerNode, 1, 8))}${inf ? field('TP degree', number('tpDegree', state.tpDegree, 1, 8)) + field('PP degree', number('ppDegree', state.ppDegree, 1, 64)) : ''}</div>`;
      if (inf && state.tpDegree * state.ppDegree > n) h += `<div class="note warn"><span class="body">TP × PP (${state.tpDegree * state.ppDegree}) is larger than the ${n} GPUs available.</span></div>`;
      h += `<div class="cols2">${field('Within a node', select('interconnect', opt(linkOptions(g), state.interconnect), state.interconnect))}${field('Between nodes', select('interNode', opt(linkOptions(g, true), state.interNode), state.interNode))}</div>`;
      if (!inf) h += '<div class="help" style="margin-top:-6px">Training in cluster mode is modelled as data parallel with ZeRO across all GPUs.</div>';
    }
    h += '</div></div>';

    // ---------- 3. Workload ----------
    h += `<div class="card"><div class="card-h"><h2><span class="step">3</span>Workload</h2></div><div class="card-b">`;
    if (diff) {
      h += `<div class="cols4">${field('Width', number('imageWidth', state.imageWidth, 64, 8192, 64))}${field('Height', number('imageHeight', state.imageHeight, 64, 8192, 64))}${field('Steps', number('inferenceSteps', state.inferenceSteps, 1, 200))}${field('Guidance', number('guidanceScale', state.guidanceScale, 0, 30, 0.5))}</div>`;
      h += `<div class="cols2">${m.task_type === 'video_generation' ? field('Frames', number('numFrames', state.numFrames, 1, 1024), 'Latent frames = 1 + ⌊(frames − 1) / 4⌋') : ''}${field('Batch size', number('diffBatch', state.diffBatch, 1, 64))}</div>`;
      h += `<div class="toggles">${toggle('encoderOffload', 'Offload text encoders to CPU', state.encoderOffload)}${toggle('vaeTiling', 'VAE tiling', state.vaeTiling)}</div>`;
    } else if (inf) {
      const steps = ctxSteps(maxCtx());
      const idx = steps.findIndex((v) => v >= state.seq);
      const i = idx < 0 ? steps.length - 1 : idx;
      h += slider('Context length', 'seqIdx', i, 0, steps.length - 1, 1, [fmtCtx(steps[0]), fmtCtx(steps[Math.floor(steps.length / 2)]), fmtCtx(steps.at(-1))], 'Prompt + generated tokens held in the KV cache per sequence.', `${fmtCtx(steps[i])} tokens`);
      h += `<div class="cols2">${field('Output tokens', number('outputTokens', state.outputTokens, 1, Math.max(1, state.seq - 1)), `Prompt = ${(state.seq - Math.min(state.outputTokens, state.seq - 1)).toLocaleString()} tokens`)}${field('Batch per request', number('batch', state.batch, 1, 256), 'Parallel samples per request (n)')}</div>`;
      h += slider('Concurrent users', 'users', state.users, 1, 256, 1, ['1', '64', '128', '192', '256'], 'Requests decoded together with continuous batching.', state.users);
      if (state.users > 1) h += slider('Shared prompt prefix', 'prefixShare', state.prefixShare, 0, 100, 5, ['0%', '50%', '100%'], 'System prompt shared across users is cached once (prefix caching).', `${state.prefixShare}%`);
      // advanced
      const advCount = (state.spec ? 1 : 0) + (state.offload ? 1 : 0);
      h += `<details class="adv" data-open="adv"${state.open.adv ? ' open' : ''}><summary>Speculative decoding &amp; offloading<span class="muted">${advCount ? `${advCount} on` : 'off'}</span></summary><div class="adv-b">`;
      h += `<div class="subcard">${toggle('spec', 'Speculative decoding', state.spec)}`;
      if (state.spec) {
        h += `<div class="cols3">${field('Draft size (B)', number('specDraft', state.specDraft, 0.05, 100, 0.05))}${field('Draft tokens k', number('specTokens', state.specTokens, 1, 16))}${field('Drafting', select('specType', opt([['sequential', 'Sequential draft model'], ['parallel', 'Parallel heads (EAGLE / Medusa)']], state.specType), state.specType))}</div>`;
        h += slider('Acceptance rate α', 'specAccept', state.specAccept, 0.1, 0.95, 0.05, ['0.1', '0.5', '0.95'], 'Probability each drafted token is accepted. 0.6–0.8 is typical for chat.', state.specAccept.toFixed(2));
      }
      h += `</div><div class="subcard">${toggle('offload', 'Offload part of the model', state.offload)}`;
      if (state.offload) {
        const L = C.geometry(m).L;
        h += `<div class="cols2">${field('Offload to', select('offloadTarget', opt(isApple(g) ? [['nvme', 'NVMe SSD (memory-mapped)']] : [['cpu_ram', 'System RAM'], ['nvme', 'NVMe SSD']], state.offloadTarget), state.offloadTarget))}${isApple(g) ? '' : `<div style="padding-top:26px">${toggle('offloadKv', 'Keep KV cache in system RAM', state.offloadKv)}</div>`}</div>`;
        h += slider('Share of layers offloaded', 'offloadPct', state.offloadPct, 0, 100, 1, ['0%', '50%', '100%'], `≈ ${Math.round(L * state.offloadPct / 100)} of ${L} layers`, `${state.offloadPct}%`);
      }
      h += '</div></div></details>';
    } else {
      const total = state.numSamples * state.tokensPerSample * state.numEpochs;
      h += `<div class="cols3">${field('Samples', number('numSamples', state.numSamples, 1, 1e9))}${field('Tokens / sample', number('tokensPerSample', state.tokensPerSample, 1, 1e6))}${field('Epochs', number('numEpochs', state.numEpochs, 1, 1000))}</div>`;
      h += `<div class="note info"><span class="body">Training on <b>${fmtTok(total)} tokens</b>${result?.perf?.hours != null ? ` · about <b>${fmtDur(result.perf.hours)}</b>` : ''}</span></div>`;
      const steps = ctxSteps(Math.min(maxCtx(), 131072));
      const idx = steps.findIndex((v) => v >= state.ftSeq); const i = idx < 0 ? steps.length - 1 : idx;
      h += slider('Sequence length', 'ftSeqIdx', i, 0, steps.length - 1, 1, [fmtCtx(steps[0]), fmtCtx(steps.at(-1))], 'Max tokens per training sequence.', `${fmtCtx(steps[i])} tokens`);
      h += `<div class="cols2">${field('Micro-batch per GPU', number('ftBatch', state.ftBatch, 1, 256))}${field('Gradient accumulation', number('gradSteps', state.gradSteps, 1, 256), `Effective batch = ${state.ftBatch * state.gradSteps * totalGpus()} sequences`)}</div>`;
      const o = state.opt;
      const z = totalGpus() > 1 ? o.zero_stage : 0;
      h += `<details class="adv" data-open="opt"${state.open.opt ? ' open' : ''}><summary>Memory &amp; speed optimizations<span class="muted">${o.optimizer_type.toUpperCase()}${o.gradient_checkpointing ? ' · checkpointing' : ''}${totalGpus() > 1 ? ` · ZeRO-${z}` : ''}</span></summary><div class="adv-b">`;
      h += `<div class="cols2">${field('Optimizer', select('opt.optimizer_type', opt([['adamw', 'AdamW'], ['adam', 'Adam'], ['sgd', 'SGD with momentum'], ['adafactor', 'Adafactor']], o.optimizer_type), o.optimizer_type))}${totalGpus() > 1 ? field('ZeRO stage', select('opt.zero_stage', opt([[0, 'ZeRO-0 (plain DDP)'], [1, 'ZeRO-1 · optimizer'], [2, 'ZeRO-2 · + gradients'], [3, 'ZeRO-3 · + weights']], o.zero_stage), o.zero_stage)) : ''}</div>`;
      h += `<div class="toggles">${toggle('opt.flash_attention', 'Flash attention', o.flash_attention)}${toggle('opt.gradient_checkpointing', 'Gradient checkpointing', o.gradient_checkpointing)}${toggle('opt.use_8bit_optimizer', '8-bit optimizer states', o.use_8bit_optimizer)}${toggle('opt.use_bf16_true', 'Pure BF16 (no FP32 master)', o.use_bf16_true)}${toggle('opt.use_fused_kernels', 'Fused kernels', o.use_fused_kernels)}${toggle('opt.sequence_packing', 'Sequence packing', o.sequence_packing)}${toggle('opt.dynamic_padding', 'Dynamic padding', o.dynamic_padding, o.sequence_packing)}${toggle('opt.activation_offloading', 'Activation offloading', o.activation_offloading)}</div>`;
      if (o.gradient_checkpointing) h += `<div style="margin-top:10px">${slider('Layers checkpointed', 'opt.gradient_checkpointing_ratio', Math.round(o.gradient_checkpointing_ratio * 100), 0, 100, 5, ['0%', '50%', '100%'], null, `${Math.round(o.gradient_checkpointing_ratio * 100)}%`)}</div>`;
      if (totalGpus() > 1) h += `<div class="toggles" style="margin-top:6px">${toggle('opt.offload_optimizer', 'Offload optimizer (ZeRO ≥ 1)', o.offload_optimizer, z < 1)}${toggle('opt.offload_gradients', 'Offload gradients (ZeRO ≥ 2)', o.offload_gradients, z < 2)}${toggle('opt.offload_param', 'Offload weights (ZeRO-3)', o.offload_param, z < 3)}</div>`;
      h += '</div></details>';
    }
    h += '</div></div>';
    $('#inputs').innerHTML = h;
  }

  /* ---------------- results ---------------- */
  function renderResults() {
    const el = $('#results');
    const r = result;
    if (!r) { el.innerHTML = ''; return; }
    const m = model(), g = gpu(), n = r.layout.n, diff = !!r.diffusion;
    const per = state.view === 'per' || n === 1;
    const mem = r.memory;
    const used = per ? mem.perGpu : mem.total, cap = per ? mem.capPerGpu : mem.capTotal;
    const raw = per ? mem.rawPerGpu : mem.rawPerGpu * n;
    const segs = mem.segments.filter((s) => s.gib > 0.0005);
    const scaleTo = Math.max(used, cap);
    let h = `<div class="res-top"><div class="res-row"><span class="res-label">${per ? (n > 1 ? 'Memory per GPU' : 'Memory needed') : `Memory across ${n} GPUs`}</span><span class="status ${r.status}">${STATUS_LABEL[r.status]}</span></div>
      <div class="bignum">${used >= 100 ? Math.round(used).toLocaleString() : used.toFixed(1)}<small>GB</small></div>
      <div class="of">of ${f1(cap)} GB ${cap < raw - 0.05 ? `usable (${raw.toLocaleString()} GB ${isApple(g) ? 'unified' : 'total'})` : 'available'} · <b class="num">${mem.pct.toFixed(1)}%</b></div>
      <div class="membar" role="img" aria-label="Memory usage ${mem.pct.toFixed(0)} percent">${segs.map((s) => `<i style="width:${(s.gib * (per ? 1 : n)) / scaleTo * 100}%;background:${SEG_COLORS[s.key] || '#999'}" title="${esc(s.label)}: ${gb(s.gib * (per ? 1 : n))}"></i>`).join('')}${used > cap ? `<span class="cap" style="left:${cap / scaleTo * 100}%" title="Capacity"></span>` : ''}</div>
      <div class="membar-scale"><span>0</span><span>${used > cap ? `capacity ${f1(cap)} GB ▲` : `${f1(cap)} GB`}</span></div>
      ${n > 1 ? `<div class="viewtoggle seg sm"><button data-view="per" class="${per ? 'on' : ''}">Per GPU</button><button data-view="total" class="${!per ? 'on' : ''}">All ${n} GPUs</button></div>` : ''}</div>`;
    if (r.warnings.length) h += `<div class="res-sec" style="padding-bottom:8px">${r.warnings.map((w) => `<div class="note ${r.status === 'insufficient' && /fit/.test(w) ? 'bad' : 'warn'}"><span class="body">${esc(w)}</span></div>`).join('')}</div>`;

    // KPIs
    const p = r.perf;
    const dash = '<span style="color:var(--faint)">—</span>';
    let k = '<div class="kpis">';
    if (diff) {
      const unit = m.task_type === 'video_generation' ? 'video' : 'image';
      k += kpi('Time per ' + unit, p ? `${p.seconds >= 100 ? Math.round(p.seconds) : p.seconds.toFixed(1)}<small>s</small>` : dash, p ? `${state.inferenceSteps} steps · batch ${state.diffBatch}` : 'does not fit');
      k += kpi('Denoising speed', p ? `${p.itPerSec >= 1 ? p.itPerSec.toFixed(2) : (1 / p.itPerSec).toFixed(2)}<small>${p.itPerSec >= 1 ? 'it/s' : 's/it'}</small>` : dash, p ? `${p.latentTokens.toLocaleString()} latent tokens` : '');
    } else if (r.mode === 'inference') {
      k += kpi('Generation speed', p ? `${fmtRate(p.tps)}<small>tok/s</small>` : dash, p ? `${state.users > 1 ? 'per user · ' : ''}${p.msPerToken.toFixed(1)} ms/token${p.spec ? ` · ${p.spec.speedup.toFixed(2)}× spec` : ''}` : 'does not fit');
      k += kpi('Time to first token', p ? fmtMs(p.ttft).replace(/ (ms|s|min)$/, '<small>$1</small>') : dash, p ? `${p.promptTok.toLocaleString()}-token prompt${state.users > 1 ? ` · ≤ ${fmtMs(p.ttftQueued)} queued` : ''}` : '');
      k += kpi('Total throughput', p ? `${fmtRate(p.throughput)}<small>tok/s</small>` : dash, p ? `${state.users * state.batch} concurrent sequence${state.users * state.batch > 1 ? 's' : ''}` : '');
      k += kpi('Full response', p ? fmtMs(p.genSeconds * 1000).replace(/ (ms|s|min)$/, '<small>$1</small>') : dash, p ? `${p.outTok.toLocaleString()} output tokens` : '');
    } else {
      k += kpi('Training speed', p ? `${fmtRate(p.trainTps)}<small>tok/s</small>` : dash, p ? `MFU ${Math.round(p.mfu * 100)}% · ${n} GPU${n > 1 ? 's' : ''}` : 'does not fit');
      k += kpi('Training time', p ? fmtDur(p.hours).replace(/ (s|min|h|days)$/, '<small>$1</small>') : dash, p ? `${fmtTok(p.dataTok)} tokens` : '');
      k += kpi('Samples / second', p ? `${p.samplesPerSec >= 10 ? p.samplesPerSec.toFixed(1) : p.samplesPerSec.toFixed(2)}` : dash, p ? `${p.steps.toLocaleString()} optimizer steps` : '');
      k += kpi('Step time', p ? `${(1 / p.stepsPerSec).toFixed(2)}<small>s</small>` : dash, p ? `${fmtTok(p.tokensPerStep)} tokens / step` : '');
    }
    h += k + '</div>';
    h += `<div class="tabs" role="tablist"><button data-tab="breakdown" class="${state.tab === 'breakdown' ? 'on' : ''}">Memory breakdown</button><button data-tab="cost" class="${state.tab === 'cost' ? 'on' : ''}">Cost &amp; energy</button></div>`;
    h += state.tab === 'breakdown' ? renderBreakdown(r, per, n) : renderCost(r, n);
    h += `<div class="res-sec" style="display:flex;gap:8px;flex-wrap:wrap"><button class="icon-btn" data-act="link"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M6.5 9.5l3-3M7 4.5l1-1a2.8 2.8 0 014 4l-1 1M9 11.5l-1 1a2.8 2.8 0 01-4-4l1-1"/></svg>Copy link</button><button class="icon-btn" data-act="json"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="5" y="5" width="8" height="8" rx="1.5"/><path d="M3 10.5V4a1 1 0 011-1h6.5"/></svg>Copy JSON</button></div>`;
    el.innerHTML = h;
  }
  const kpi = (k, v, s) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s || '&nbsp;'}</div></div>`;

  function renderBreakdown(r, per, n) {
    const mem = r.memory;
    const mult = per ? 1 : n;
    const tot = mem.segments.reduce((a, s) => a + s.gib, 0) || 1;
    let h = `<div class="res-sec"><h3>${per && n > 1 ? 'Per GPU' : 'Where the memory goes'}<span>${gb((per ? mem.perGpu : mem.total))}</span></h3><div class="legend">`;
    h += mem.segments.filter((s) => s.gib > 0.0005).map((s) => `<div class="leg"><i style="background:${SEG_COLORS[s.key] || '#999'}"></i><span title="${esc(s.label)}">${esc(s.label)}</span><b>${gb(s.gib * mult)}</b><small>${(s.gib / tot * 100).toFixed(1)}%</small></div>`).join('');
    h += '</div></div>';
    if (mem.offloaded?.length) {
      h += `<div class="res-sec"><h3>Held outside the GPU<span>${mem.offloadTarget === 'nvme' ? 'NVMe' : 'System RAM'}</span></h3><div class="legend">${mem.offloaded.map((s) => `<div class="leg"><i style="background:var(--line-2)"></i><span>${esc(s.label)}</span><b>${gb(s.gib)}</b><small></small></div>`).join('')}</div></div>`;
    }
    if (r.mode === 'inference' && !r.diffusion) {
      const G = r.geometry;
      h += `<div class="res-sec"><h3>Inference details</h3><dl class="kv">
        <dt>Engine</dt><dd>${esc(r.runtimeLabel)}</dd>
        <dt>KV cache per token</dt><dd>${mem.kvPerToken >= 1048576 ? `${(mem.kvPerToken / 1048576).toFixed(2)} MiB` : `${(mem.kvPerToken / 1024).toFixed(1)} KiB`}</dd>
        ${state.users > 1 ? `<dt>Shared (weights + runtime)</dt><dd>${gb(mem.sharedGiB)}</dd><dt>Added per user</dt><dd>${gb(mem.perUserGiB)}</dd>` : ''}
        ${r.perf ? `<dt>Prefill speed</dt><dd>${fmtRate(r.perf.prefillTps)} tok/s</dd><dt>Decode is</dt><dd>${r.perf.bound}-bound</dd>` : ''}
        ${G.moe ? `<dt>Active / total params</dt><dd>${(G.A / 1e9).toFixed(1)}B / ${(G.P / 1e9).toFixed(1)}B</dd>` : ''}
        ${r.layout.n > 1 ? `<dt>Layout</dt><dd>TP ${r.layout.tp} × PP ${r.layout.pp}${r.layout.dp > 1 ? ` × ${r.layout.dp} replicas` : ''}</dd>` : ''}
      </dl></div>`;
    }
    return h;
  }

  function renderCost(r, n) {
    const g = gpu();
    const e = r.energy;
    const hourly = g.hourly_price_usd ? g.hourly_price_usd * n : null;
    let h = '<div class="res-sec"><h3>Cloud rental<span>on-demand list price</span></h3>';
    if (hourly) {
      h += `<dl class="kv"><dt>${n} × ${esc(g.label)}</dt><dd>${usd(hourly)}/h</dd>`;
      if (r.mode === 'inference' && !r.diffusion) {
        h += `<dt>Per month (730 h)</dt><dd>${usd(hourly * 730, 0)}</dd><dt>Spot, ~35% lower</dt><dd>${usd(hourly * 730 * 0.65, 0)}</dd>`;
        if (r.perf) h += `<div class="sep"></div><dt>Per 1M output tokens</dt><dd>${usd(hourly / (r.perf.throughput * 3600) * 1e6, hourly / (r.perf.throughput * 3600) * 1e6 < 1 ? 3 : 2)}</dd><dt style="grid-column:1/-1;font-size:12px">at ${fmtRate(r.perf.throughput)} tok/s, fully utilized</dt>`;
      } else if (r.diffusion) {
        if (r.perf) h += `<dt>Per ${model().task_type === 'video_generation' ? 'video' : 'image'}</dt><dd>${usd(hourly * r.perf.seconds / 3600, 4)}</dd><dt>Per month (730 h)</dt><dd>${usd(hourly * 730, 0)}</dd>`;
      } else if (r.perf) {
        h += `<dt>Training run (${fmtDur(r.perf.hours)})</dt><dd>${usd(hourly * r.perf.hours)}</dd><dt>Spot, ~35% lower</dt><dd>${usd(hourly * r.perf.hours * 0.65)}</dd>`;
      }
      h += '</dl>';
    } else h += `<div class="help" style="margin:0">No rental price listed for ${esc(g.label)} — typically owned hardware.</div>`;
    h += '</div>';
    h += `<div class="res-sec"><h3>Electricity<span>${Math.round(e.watts).toLocaleString()} W estimated draw</span></h3><dl class="kv">
      <dt>Per hour</dt><dd>${usd(e.elecPerHour, 3)}</dd><dt>Per day</dt><dd>${usd(e.elecPerHour * 24)}</dd><dt>Per month</dt><dd>${usd(e.elecPerHour * 24 * 30)}</dd><dt>Per year</dt><dd>${usd(e.elecPerHour * 24 * 365, 0)}</dd></dl>
      <div class="field" style="margin:14px 0 0"><div class="lbl">Electricity price ($/kWh)</div>${number('elec', state.elec, 0, 5, 0.01)}<div class="chips">${ELEC.map((x) => `<button class="chip-btn${Math.abs(state.elec - x.v) < 1e-9 ? ' on' : ''}" data-elec="${x.v}">${x.k} $${x.v}</button>`).join('')}</div></div></div>`;
    h += `<div class="res-sec"><h3>Carbon<span>${e.kwhPerDay.toFixed(1)} kWh / day</span></h3><dl class="kv"><dt>CO₂ per day</dt><dd>${e.co2KgPerDay.toFixed(2)} kg</dd><dt>CO₂ per year</dt><dd>${e.co2KgPerYear >= 1000 ? `${(e.co2KgPerYear / 1000).toFixed(2)} t` : `${Math.round(e.co2KgPerYear)} kg`}</dd></dl>
      <div class="field" style="margin:14px 0 0"><div class="lbl">Grid carbon intensity</div>${select('carbon', opt(CARBON.map(([v, l]) => [v, `${l} — ${v} g/kWh`]), state.carbon), state.carbon)}</div></div>`;
    return h;
  }

  /* ---------------- token simulator ---------------- */
  const SAMPLE = `A graphics processor spends most of its time during text generation waiting on memory, not doing math. To produce each new token, the model has to stream every weight it uses through the chip once, and then read the key and value vectors it saved for every earlier token in the conversation. That is why memory bandwidth, measured in gigabytes per second, predicts generation speed better than raw compute does.

The first token is different. Before anything appears on screen, the whole prompt is processed in parallel, and that phase is limited by arithmetic throughput instead. Long prompts therefore cost time up front, while long answers cost time token by token.

Batching changes the picture again. When many requests are decoded together, the weights are read once and shared across all of them, so the total number of tokens per second climbs steeply even though each individual user sees a slightly slower stream. Serving engines exploit this by admitting new requests into the running batch as soon as there is room in the key-value cache.

Quantization helps in two ways at once: smaller weights take less memory, and fewer bytes per token means each decoding step finishes sooner. The trade-off is a small loss in accuracy that grows as the bit width shrinks, so most deployments settle somewhere between eight and four bits per weight.`;
  const TOKENS = (() => {
    const out = [];
    const parts = SAMPLE.match(/\s*[\w’'-]+|\s*[^\w\s]|\s+/g) || [];
    for (const p of parts) {
      const lead = p.match(/^\s*/)[0], word = p.slice(lead.length);
      if (word.length <= 6) { out.push(p); continue; }
      let first = true;
      for (let i = 0; i < word.length; i += (i === 0 ? 5 : 4)) { out.push((first ? lead : '') + word.slice(i, i + (i === 0 ? 5 : 4))); first = false; }
    }
    return out;
  })();
  const sim = { running: false, t0: 0, raf: 0, lanes: [], compare: 'reading', customRate: 20, lastSig: '', speed: 1 };

  function simLanes() {
    const r = result;
    const p = r?.perf;
    if (!p || r.mode !== 'inference' || r.diffusion) return null;
    const out = Math.min(state.outputTokens, 4000);
    const lanes = [{ id: 'main', name: `${model().name} on ${totalGpus() > 1 ? `${totalGpus()} × ` : ''}${gpu().label}`, sub: state.users > 1 ? `per-user speed with ${state.users} users` : 'single user', rate: p.tps, ttft: p.ttft / 1000, out }];
    if (sim.compare === 'reading') lanes.push({ id: 'cmp', name: 'Human reading speed', sub: '≈ 240 words/min', rate: 5.3, ttft: 0, out, ref: true });
    else if (sim.compare === 'custom') lanes.push({ id: 'cmp', name: 'Custom rate', sub: 'for comparison', rate: sim.customRate, ttft: 0.3, out, ref: true });
    else if (sim.compare === 'solo' && state.users > 1) {
      const solo = C.calculate({ ...engineInput(), users: 1 }, model(), gpu()).perf;
      if (solo) lanes.push({ id: 'cmp', name: 'Same setup, one user', sub: 'no batching contention', rate: solo.tps, ttft: solo.ttft / 1000, out, ref: true });
    } else if (sim.compare === 'int4' || sim.compare === 'fp8') {
      const q = sim.compare === 'int4' ? (C.QUANTS[state.quant]?.gguf ? 'q4_k_m' : 'int4') : 'fp8';
      const alt = C.calculate({ ...engineInput(), quant: q }, model(), gpu()).perf;
      if (alt) lanes.push({ id: 'cmp', name: `Same setup in ${C.QUANTS[q].label.split(' ')[0]}`, sub: 'weight precision only', rate: alt.tps, ttft: alt.ttft / 1000, out, ref: true });
    }
    return lanes;
  }

  function renderSimulator() {
    const el = $('#simulator');
    const lanes = simLanes();
    let h = `<div class="card-h"><h2>Token stream simulator</h2><span class="hint">Real-time playback of the speed this configuration produces</span></div><div class="card-b">`;
    if (!lanes) {
      const why = result?.diffusion ? 'The simulator plays back text generation. Pick a text model to use it.' : state.mode === 'finetuning' ? 'The simulator plays back text generation. Switch to Inference to use it.' : 'This configuration does not fit in memory, so there is nothing to play back yet.';
      el.innerHTML = h + `<div class="note info"><span class="body">${why}</span></div>${state.mode === 'finetuning' && !result?.diffusion ? '<button class="btn" data-mode="inference">Switch to inference</button>' : ''}</div>`;
      sim.lanes = []; sim.lastSig = ''; stopSim(); return;
    }
    sim.lanes = lanes;
    h += `<div class="sim-grid"><div class="lanes">${lanes.map((l, i) => `<div class="lane${i ? ' compact' : ''}" data-lane="${l.id}"><div class="lane-h"><span class="dot${l.ref ? ' ref' : ''}"></span><b>${esc(l.name)}</b><span class="muted" style="font-size:12px">${esc(l.sub)}</span><span class="rate">${fmtRate(l.rate)} tok/s${l.ttft ? ` · TTFT ${fmtMs(l.ttft * 1000)}` : ''}</span></div><div class="lane-text"><div class="inner"></div></div><div class="lane-foot"><span class="t">0.0 s</span><div class="prog"><div class="ttft"></div><div class="gen"></div></div><span class="c">0 / ${l.out.toLocaleString()}</span></div></div>`).join('')}</div>
      <div class="sim-side"><div class="btn-row"><button class="btn primary" data-sim="play"><svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>${sim.running ? 'Restart' : 'Play'}</button><button class="btn" data-sim="stop"${sim.running ? '' : ' disabled'}>Stop</button></div>${lanes[0].ttft > 3 ? `<button class="btn" data-sim="skip"${sim.running ? '' : ' disabled'}>Skip the ${fmtMs(lanes[0].ttft * 1000)} prompt wait</button>` : ''}
      <div class="sim-stats"><div><div class="k">Elapsed</div><div class="v" id="ss-el">0.0 s</div></div><div><div class="k">Tokens</div><div class="v" id="ss-tok">0</div></div><div><div class="k">Live rate</div><div class="v" id="ss-rate">–</div></div><div><div class="k">Finishes at</div><div class="v">${fmtMs((lanes[0].ttft + lanes[0].out / lanes[0].rate) * 1000)}</div></div></div>
      ${field('Compare against', select('simCompare', opt([['reading', 'Human reading speed'], ['fp8', 'This setup in FP8'], ['int4', 'This setup in 4-bit'], ...(state.users > 1 ? [['solo', 'This setup with one user']] : []), ['custom', 'A custom rate'], ['none', 'Nothing']], sim.compare), sim.compare))}
      ${field('Playback speed', `<div class="seg block sm">${[1, 2, 5, 20].map((x) => `<button data-speed="${x}" class="${sim.speed === x ? 'on' : ''}">${x}×</button>`).join('')}</div>`, sim.speed === 1 ? 'Real time' : `${sim.speed}× faster than real time; timers show simulated seconds`)}
      ${sim.compare === 'custom' ? slider('Custom rate', 'simRate', sim.customRate, 1, 300, 1, ['1', '150', '300'], null, `${sim.customRate} tok/s`) : ''}
      <div class="help">Plays ${lanes[0].out.toLocaleString()} output tokens (set in Workload). Text is a fixed sample tokenized at ≈1.3 tokens per word, like a typical BPE tokenizer.</div></div></div></div>`;
    el.innerHTML = h;
    sim.lastSig = lanes.map((l) => `${l.rate.toFixed(3)}|${l.ttft.toFixed(3)}|${l.out}`).join(';');
    if (sim.running) drawSim(performance.now()); else drawSim(null);
  }
  function startSim() { sim.running = true; sim.t0 = performance.now(); renderSimulator(); cancelAnimationFrame(sim.raf); sim.raf = requestAnimationFrame(tick); }
  function stopSim() { sim.running = false; cancelAnimationFrame(sim.raf); const b = $('[data-sim="stop"]'); if (b) b.disabled = true; const sk = $('[data-sim="skip"]'); if (sk) sk.disabled = true; const p = $('[data-sim="play"]'); if (p) p.lastChild.textContent = 'Play'; }
  function tick(now) { const done = drawSim(now); if (done) { stopSim(); } else sim.raf = requestAnimationFrame(tick); }
  function drawSim(now) {
    const el = now == null ? 0 : (now - sim.t0) / 1000 * sim.speed;
    let allDone = true, mainTok = 0;
    for (const l of sim.lanes) {
      const box = $(`[data-lane="${l.id}"]`);
      if (!box) continue;
      const n = el <= l.ttft ? 0 : Math.min(l.out, Math.floor((el - l.ttft) * l.rate));
      if (n < l.out) allDone = false;
      if (l.id === 'main') mainTok = n;
      const inner = $('.inner', box);
      if (box.dataset.n !== String(n) || now == null) {
        box.dataset.n = n;
        if (now != null && el <= l.ttft) inner.innerHTML = `<span class="wait"><span class="spinner"></span>Processing prompt… ${fmtMs((l.ttft - el) * 1000)} until first token</span>`;
        else if (now == null) inner.innerHTML = '<span class="wait">Press Play to stream the response at this speed.</span>';
        else {
          const tok = (i) => TOKENS[i % TOKENS.length] + (i % TOKENS.length === TOKENS.length - 1 ? '\n\n' : '');
          let text = '';
          for (let i = Math.max(0, n - 400); i < n - 1; i++) text += tok(i);
          const newest = n > 0 ? tok(n - 1) : '';
          inner.innerHTML = `${esc(text)}<span class="tok-new">${esc(newest)}</span>${n < l.out ? '<span class="cursor"></span>' : ''}`;
        }
      }
      const total = l.ttft + l.out / l.rate;
      $('.t', box).textContent = `${Math.min(el, total).toFixed(1)} s`;
      $('.c', box).textContent = `${n.toLocaleString()} / ${l.out.toLocaleString()}`;
      $('.ttft', box).style.width = `${Math.min(el, l.ttft) / total * 100}%`;
      $('.gen', box).style.width = `${Math.max(0, Math.min(el, total) - l.ttft) / total * 100}%`;
    }
    const main = sim.lanes[0];
    if (main && $('#ss-el')) {
      $('#ss-el').textContent = `${el.toFixed(1)} s`;
      $('#ss-tok').textContent = mainTok.toLocaleString();
      const sk = $('[data-sim="skip"]'); if (sk) sk.disabled = now == null || el >= main.ttft;
      $('#ss-rate').textContent = el > main.ttft && mainTok > 0 ? `${fmtRate(mainTok / (el - main.ttft))}` : '–';
    }
    return now != null && allDone;
  }

  /* ---------------- compute + wiring ---------------- */
  function compute() {
    try { result = C.calculate(engineInput(), model(), gpu()); }
    catch (e) { console.error(e); result = null; $('#results').innerHTML = `<div class="res-sec"><div class="note bad"><span class="body">Calculation error: ${esc(e.message)}</span></div></div>`; return; }
    renderResults();
    const lanes = simLanes();
    const sig = lanes ? lanes.map((l) => `${l.rate.toFixed(3)}|${l.ttft.toFixed(3)}|${l.out}`).join(';') : '';
    if (sig !== sim.lastSig || !lanes) { if (sim.running) stopSim(); renderSimulator(); }
    // labels that depend on the result
    const rt = $('[data-k="runtime"] option[value="auto"]');
    if (rt && result.runtimeLabel) rt.textContent = `Auto — ${result.runtimeLabel}`;
    syncUrl();
  }
  let rafId = 0;
  const scheduleCompute = () => { cancelAnimationFrame(rafId); rafId = requestAnimationFrame(compute); };
  function renderAll() { compute(); renderInputs(); }

  function setPath(path, v) {
    const ks = path.split('.');
    let o = state;
    for (let i = 0; i < ks.length - 1; i++) o = o[ks[i]];
    o[ks.at(-1)] = v;
  }
  function readValue(t) {
    if (t.type === 'checkbox') return t.checked;
    if (t.type === 'number' || t.type === 'range') return Number(t.value);
    if (/^(opt\.zero_stage|carbon)$/.test(t.dataset.k)) return Number(t.value);
    return t.value;
  }
  function onInput(e, committed) {
    const t = e.target;
    const k = t.dataset.k;
    if (!k) return;
    let v = readValue(t);
    if (t.type === 'number') {
      if (!Number.isFinite(v) || t.value === '') return;
      if (committed) { v = clamp(v, t.min !== '' ? Number(t.min) : -Infinity, t.max !== '' ? Number(t.max) : Infinity); if (t.step === '1' || !t.step) v = Math.round(v); t.value = v; }
      else if ((t.min !== '' && v < Number(t.min)) || (t.max !== '' && v > Number(t.max))) return;
    }
    if (k === 'simCompare') { sim.compare = v; return renderSimulator(); }
    if (k === 'simRate') { sim.customRate = v; $('#v-simRate').textContent = `${v} tok/s`; t.style.setProperty('--p', `${(v - 1) / 299 * 100}%`); return committed ? renderSimulator() : undefined; }
    if (k === 'seqIdx' || k === 'ftSeqIdx') {
      const steps = ctxSteps(k === 'seqIdx' ? maxCtx() : Math.min(maxCtx(), 131072));
      const val = steps[v];
      if (k === 'seqIdx') { state.seq = val; state.outputTokens = Math.min(state.outputTokens, val - 1); } else state.ftSeq = val;
      $(`#v-${k}`).textContent = `${fmtCtx(val)} tokens`;
    } else if (k === 'opt.gradient_checkpointing_ratio') { state.opt.gradient_checkpointing_ratio = v / 100; $('#v-opt-gradient_checkpointing_ratio').textContent = `${v}%`; }
    else if (k === 'prefixShare' || k === 'offloadPct') { state[k] = v; $(`#v-${k}`).textContent = `${v}%`; }
    else if (k === 'specAccept') { state.specAccept = v; $('#v-specAccept').textContent = v.toFixed(2); }
    else if (k === 'users') { state.users = v; $('#v-users').textContent = v; }
    else setPath(k, v);
    if (k === 'opt.sequence_packing' && v) state.opt.dynamic_padding = false;
    if (k === 'numGpus' || k === 'numNodes' || k === 'gpusPerNode') { const g = gpu(); if (!linkOptions(g).some(([x]) => x === state.interconnect)) state.interconnect = defaultLink(g); }
    if (t.type === 'range') t.style.setProperty('--p', `${(v - t.min) / (t.max - t.min) * 100}%`);
    if (t.type === 'range' && !committed) return scheduleCompute();
    if (t.type === 'number' && !committed) return scheduleCompute();
    if (t.closest('#results')) return compute();
    renderAll();
  }

  function wire() {
    const inputs = $('#inputs');
    inputs.addEventListener('input', (e) => onInput(e, false));
    inputs.addEventListener('change', (e) => onInput(e, true));
    inputs.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.dataset.pick) return openPicker(b);
      if (b.dataset.set) {
        const k = b.dataset.set; let v = b.dataset.v;
        if (k === 'cluster') { state.cluster = v === '1'; if (state.cluster) { const g = gpu(); state.interNode = linkOptions(g, true)[0]?.[0]; } }
        else state[k] = v;
        return renderAll();
      }
    });
    inputs.addEventListener('toggle', (e) => { const d = e.target; if (d.dataset?.open) state.open[d.dataset.open] = d.open; }, true);
    $('#mode-seg').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b && !b.disabled) { state.mode = b.dataset.mode; renderAll(); } });
    const res = $('#results');
    res.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.dataset.tab) { state.tab = b.dataset.tab; return renderResults(); }
      if (b.dataset.view) { state.view = b.dataset.view; return renderResults(); }
      if (b.dataset.elec) { state.elec = Number(b.dataset.elec); return compute(); }
      if (b.dataset.act === 'json' || b.dataset.act === 'link') {
        const txt = b.dataset.act === 'link' ? location.href : JSON.stringify({ model: state.model, gpu: gpu(), input: engineInput(), result: { ...result, trace: result.trace } }, (k, v) => (k === 'geometry' ? undefined : v), 2);
        navigator.clipboard?.writeText(txt).then(() => { const o = b.innerHTML; b.textContent = 'Copied'; setTimeout(() => { b.innerHTML = o; }, 1200); }).catch(() => {});
      }
    });
    res.addEventListener('change', (e) => onInput(e, true));
    const simEl = $('#simulator');
    simEl.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.dataset.sim === 'play') startSim();
      if (b.dataset.sim === 'stop') { stopSim(); }
      if (b.dataset.sim === 'skip' && sim.running && sim.lanes[0]) { const el = (performance.now() - sim.t0) / 1000 * sim.speed; if (el < sim.lanes[0].ttft) sim.t0 = performance.now() - ((sim.lanes[0].ttft + 0.01) / sim.speed) * 1000; }
      if (b.dataset.speed) {
        const elapsed = sim.running ? (performance.now() - sim.t0) / 1000 * sim.speed : 0;
        sim.speed = Number(b.dataset.speed);
        if (sim.running) sim.t0 = performance.now() - (elapsed / sim.speed) * 1000;
        renderSimulator();
      }
      if (b.dataset.mode) { state.mode = b.dataset.mode; renderAll(); }
    });
    simEl.addEventListener('input', (e) => onInput(e, false));
    simEl.addEventListener('change', (e) => onInput(e, true));
    // theme
    const applyTheme = (t) => { if (t === 'system') document.documentElement.removeAttribute('data-theme'); else document.documentElement.dataset.theme = t; $$('[data-theme-set]').forEach((b) => b.classList.toggle('on', b.dataset.themeSet === t)); try { localStorage.setItem('gpusim-theme', t); } catch {} };
    $$('[data-theme-set]').forEach((b) => b.addEventListener('click', () => applyTheme(b.dataset.themeSet)));
    let t0 = 'system'; try { t0 = localStorage.getItem('gpusim-theme') || 'system'; } catch {}
    applyTheme(t0);
  }

  /* ---------------- URL state ---------------- */
  const URL_KEYS = ['mode', 'model', 'quant', 'kvQuant', 'runtime', 'gpu', 'numGpus', 'parallelism', 'interconnect', 'seq', 'outputTokens', 'batch', 'users', 'prefixShare', 'ftMethod', 'ftQuant', 'loraRank', 'ftSeq', 'ftBatch', 'gradSteps', 'numSamples', 'tokensPerSample', 'numEpochs', 'cluster', 'numNodes', 'gpusPerNode', 'tpDegree', 'ppDegree', 'interNode', 'spec', 'specDraft', 'specAccept', 'specTokens', 'specType', 'offload', 'offloadTarget', 'offloadPct', 'offloadKv', 'elec', 'carbon'];
  const DEFAULTS = JSON.parse(JSON.stringify(state));
  let urlTimer = 0;
  function syncUrl() {
    clearTimeout(urlTimer);
    urlTimer = setTimeout(() => {
      const p = new URLSearchParams();
      for (const k of URL_KEYS) if (JSON.stringify(state[k]) !== JSON.stringify(DEFAULTS[k])) p.set(k, String(state[k]));
      if (state.gpu === 'custom') { p.set('custom', [state.custom.memory, state.custom.bandwidth_gbs, state.custom.fp16_tflops, state.custom.tdp_watts, state.customApple ? 1 : 0].join(',')); }
      history.replaceState(null, '', p.toString() ? `?${p}${location.hash}` : location.pathname + location.hash);
    }, 250);
  }
  function loadUrl() {
    const p = new URLSearchParams(location.search);
    for (const k of URL_KEYS) {
      if (!p.has(k)) continue;
      const raw = p.get(k), d = DEFAULTS[k];
      const v = typeof d === 'number' ? Number(raw) : typeof d === 'boolean' ? raw === 'true' : raw;
      if (typeof v === 'number' && !Number.isFinite(v)) continue;
      state[k] = v;
    }
    if (p.has('custom')) { const [a, b, c, d2, e] = p.get('custom').split(',').map(Number); Object.assign(state.custom, { memory: a || 24, bandwidth_gbs: b || 1000, fp16_tflops: c || 165, tdp_watts: d2 || 350 }); state.customApple = e === 1; }
    if (!modelBySlug[state.model]) state.model = DEFAULTS.model;
    if (state.gpu !== 'custom' && !gpuByKey[state.gpu]) state.gpu = DEFAULTS.gpu;
    if (!C.QUANTS[state.quant]) state.quant = nativeQuant(model());
    state.seq = clamp(state.seq, 512, maxCtx());
  }

  // boot
  loadUrl();
  wire();
  compute();
  renderInputs();
  window.__gpusim = { state, engineInput, get result() { return result; } };
})();
