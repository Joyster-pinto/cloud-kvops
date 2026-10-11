/* =========================================================
   LLM-KVOps — dual chat frontend (vanilla JS, no dependencies)
   ========================================================= */
(() => {
  'use strict';

  const MODES = ['kvops', 'baseline'];
  const NAMES = { kvops: 'LLM-KVOps', baseline: 'Standard LLM' };
  const COLORS = { kvops: '#81c995', baseline: '#f28b82' };
  const STORAGE_KEY = 'llm_kvops_sessions_v2';
  const SUGGESTIONS = [
    'Tell me a fun and engaging short story about an offline robot on a mission.',
    'Explain how a KV cache works in transformers in simple words.',
    'Write a short poem about the ocean at night.',
  ];

  const $ = (sel, root = document) => root.querySelector(sel);

  // ---------------------------------------------------------
  // State
  // ---------------------------------------------------------
  const state = {
    sessions: [],
    currentId: null,
    streams: { kvops: null, baseline: null },
    series: { kvops: [], baseline: [] },   // per-token telemetry of the latest run
    summary: { kvops: null, baseline: null },
  };

  const panes = {};
  MODES.forEach((mode) => {
    const root = document.getElementById(`pane-${mode}`);
    panes[mode] = {
      root,
      messages: $('[data-role=messages]', root),
      input: $('[data-role=input]', root),
      send: $('[data-role=send]', root),
      liveKv: $('[data-live=kv]', root),
      liveCtx: $('[data-live=ctx]', root),
      attachBtn: $('[data-role=attach]', root),
      fileInput: $('[data-role=fileInput]', root),
      attachmentsEl: $('[data-role=attachments]', root),
      attachedFiles: [],
    };
  });

  const settings = {
    model: $('#modelSelect'),
    budget: $('#budget'),
    temp: $('#temp'),
  };

  // ---------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  function cleanText(t) {
    return (t || '')
      .replace(/<\|[^|>]*\|>/g, '')
      .replace(/<\/?s>/g, '')
      .replace(/^\s*(assistant|Assistant)\s*:?\s*\n/, '')
      .replace(/^\s+/, '');
  }

  function inline(s) {
    return s
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  }

  // Tiny markdown renderer: code fences, headings, lists, paragraphs, inline styles
  function renderMarkdown(src) {
    const parts = escapeHtml(src).split(/```/);
    let html = '';
    parts.forEach((part, i) => {
      if (i % 2 === 1) {
        const code = part.replace(/^[\w+-]*\n/, '');
        html += `<pre><code>${code}</code></pre>`;
        return;
      }
      const blocks = part.split(/\n{2,}/);
      blocks.forEach((block) => {
        const b = block.trim();
        if (!b) return;
        const lines = b.split('\n');
        if (lines.every((l) => /^\s*([-*•]|\d+[.)])\s+/.test(l))) {
          const ordered = /^\s*\d+[.)]/.test(lines[0]);
          const items = lines.map((l) => `<li>${inline(l.replace(/^\s*([-*•]|\d+[.)])\s+/, ''))}</li>`).join('');
          html += ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
        } else if (/^#{1,6}\s/.test(b) && lines.length === 1) {
          html += `<h3>${inline(b.replace(/^#{1,6}\s/, ''))}</h3>`;
        } else {
          html += `<p>${lines.map((l) => inline(l.replace(/^#{1,6}\s/, ''))).join('<br>')}</p>`;
        }
      });
    });
    return html;
  }

  let toastTimer;
  function toast(msg) {
    const el = $('#toast');
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 2200);
  }

  function autoGrow(ta) {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
  }

  function icon(id) {
    return `<svg><use href="#${id}"/></svg>`;
  }

  function fmt(n, d = 2) {
    return Number.isFinite(n) ? n.toFixed(d) : '—';
  }

  // ---------------------------------------------------------
  // Sessions (localStorage + server sync)
  // ---------------------------------------------------------
  function normalizeSession(s) {
    if (s && s.kvops && s.baseline && Array.isArray(s.kvops)) return s;
    // Legacy format: { prompt, kvops_response, base_response }
    if (s && s.prompt) {
      return {
        id: s.id || 'sess_' + Date.now(),
        title: s.title || s.prompt.slice(0, 40),
        updated: Date.now(),
        kvops: s.kvops_response ? [{ role: 'user', text: s.prompt }, { role: 'bot', text: s.kvops_response }] : [],
        baseline: s.base_response ? [{ role: 'user', text: s.prompt }, { role: 'bot', text: s.base_response }] : [],
      };
    }
    return null;
  }

  function loadSessions() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) state.sessions = JSON.parse(raw).map(normalizeSession).filter(Boolean);
    } catch (_) { /* ignore */ }
    renderHistory();

    fetch('/api/sessions')
      .then((r) => r.json())
      .then((data) => {
        if (!Array.isArray(data)) return;
        const known = new Set(state.sessions.map((s) => s.id));
        data.map(normalizeSession).filter(Boolean).forEach((s) => {
          if (!known.has(s.id)) state.sessions.push(s);
        });
        state.sessions.sort((a, b) => (b.updated || 0) - (a.updated || 0));
        renderHistory();
      })
      .catch(() => {});
  }

  function persist(session) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.sessions.slice(0, 50))); } catch (_) {}
    if (session) {
      fetch('/api/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(session),
      }).catch(() => {});
    }
  }

  function currentSession() {
    return state.sessions.find((s) => s.id === state.currentId) || null;
  }

  function ensureSession(firstPrompt) {
    let s = currentSession();
    if (!s) {
      s = {
        id: 'sess_' + Date.now(),
        title: firstPrompt.length > 42 ? firstPrompt.slice(0, 42) + '…' : firstPrompt,
        updated: Date.now(),
        kvops: [],
        baseline: [],
      };
      state.sessions.unshift(s);
      state.currentId = s.id;
      renderHistory();
    }
    return s;
  }

  function renderHistory() {
    const el = $('#history');
    el.innerHTML = '';
    if (!state.sessions.length) {
      el.innerHTML = '<div class="history-empty">No chats yet</div>';
      return;
    }
    state.sessions.forEach((s) => {
      const item = document.createElement('div');
      item.className = 'history-item' + (s.id === state.currentId ? ' active' : '');
      item.innerHTML = `<span class="title"></span><button class="del" title="Delete">${icon('i-trash')}</button>`;
      $('.title', item).textContent = s.title;
      item.addEventListener('click', () => openSession(s.id));
      $('.del', item).addEventListener('click', (e) => {
        e.stopPropagation();
        deleteSession(s.id);
      });
      el.appendChild(item);
    });
  }

  function deleteSession(id) {
    state.sessions = state.sessions.filter((s) => s.id !== id);
    persist();
    fetch(`/api/sessions/${id}`, { method: 'DELETE' }).catch(() => {});
    if (state.currentId === id) newChat();
    else renderHistory();
  }

  function openSession(id) {
    if (isBusy()) return toast('Wait for the current response to finish');
    state.currentId = id;
    const s = currentSession();
    MODES.forEach((mode) => {
      renderPane(mode, s[mode]);
      state.series[mode] = (s.series && s.series[mode]) || [];
      state.summary[mode] = (s.summary && s.summary[mode]) || null;
      updateLive(mode, state.series[mode][state.series[mode].length - 1]);
    });
    renderHistory();
    drawCharts();
    renderCompare();
  }

  function newChat() {
    if (isBusy()) return toast('Wait for the current response to finish');
    state.currentId = null;
    MODES.forEach((mode) => {
      state.series[mode] = [];
      state.summary[mode] = null;
      renderPane(mode, []);
      updateLive(mode, null);
      panes[mode].input.value = '';
      autoGrow(panes[mode].input);
    });
    renderHistory();
    drawCharts();
    renderCompare();
    panes.baseline.input.focus();
  }

  // ---------------------------------------------------------
  // Pane rendering
  // ---------------------------------------------------------
  function renderWelcome(mode) {
    const wrap = document.createElement('div');
    wrap.className = 'welcome';
    wrap.innerHTML = `
      <h2>Hello, I'm ${NAMES[mode]}</h2>
      <p>How can I help you today?</p>
      <div class="chips"></div>`;
    const chips = $('.chips', wrap);
    SUGGESTIONS.forEach((text) => {
      const c = document.createElement('button');
      c.className = 'chip';
      c.textContent = text;
      c.addEventListener('click', () => {
        panes[mode].input.value = text;
        autoGrow(panes[mode].input);
        updateSendState(mode);
        panes[mode].input.focus();
      });
      chips.appendChild(c);
    });
    return wrap;
  }

  function renderPane(mode, msgs) {
    const box = panes[mode].messages;
    box.innerHTML = '';
    if (!msgs.length) {
      box.appendChild(renderWelcome(mode));
      return;
    }
    msgs.forEach((m, i) => {
      if (m.role === 'user') box.appendChild(userEl(m.text));
      else {
        const prompt = findPrompt(msgs, i);
        const el = botEl(mode, prompt);
        finalizeBot(el, mode, m.text, m.stats, prompt);
        box.appendChild(el);
      }
    });
    box.scrollTop = box.scrollHeight;
  }

  function findPrompt(msgs, idx) {
    for (let i = idx - 1; i >= 0; i--) if (msgs[i].role === 'user') return msgs[i].text;
    return '';
  }

  function userEl(text) {
    const el = document.createElement('div');
    el.className = 'msg-user';
    el.innerHTML = '<div class="bubble"></div>';
    $('.bubble', el).textContent = text;
    return el;
  }

  function botEl(mode) {
    const el = document.createElement('div');
    el.className = 'msg-bot';
    el.innerHTML = `
      <div class="avatar" style="color:${COLORS[mode]}">${icon('i-spark')}</div>
      <div class="bot-body">
        <div class="content"></div>
        <div class="metrics"></div>
        <div class="actions"></div>
      </div>`;
    return el;
  }

  function metricsHtml(mode, st) {
    if (!st) return '';
    const items = [
      `<span class="metric"><b>${st.tokens}</b> tokens</span>`,
      `<span class="metric">KV <b>${fmt(st.kvMb)} MB</b></span>`,
    ];
    if (mode === 'kvops') {
      items.push(`<span class="metric good">saved <b>${fmt(st.savedPct, 1)}%</b></span>`);
      items.push(`<span class="metric">evicted <b>${st.evicted}</b></span>`);
    }
    items.push(`<span class="metric"><b>${fmt(st.avgLatency, 1)}</b> ms/tok</span>`);
    items.push(`<span class="metric">GPU <b>${fmt(st.peakGpu, 0)} MB</b></span>`);
    items.push(`<span class="metric"><b>${fmt(st.seconds, 1)}</b> s</span>`);
    if (st.stopped) items.push('<span class="metric">stopped</span>');
    return items.join('');
  }

  function finalizeBot(el, mode, text, stats, prompt) {
    el.classList.remove('generating');
    const content = $('.content', el);
    content.classList.remove('cursor');
    content.innerHTML = text ? renderMarkdown(text) : '<span class="thinking">No response.</span>';
    $('.metrics', el).innerHTML = metricsHtml(mode, stats);

    const other = mode === 'kvops' ? 'baseline' : 'kvops';
    const actions = $('.actions', el);
    actions.innerHTML = `
      <button class="icon-btn" data-act="copy" title="Copy">${icon('i-copy')}</button>
      <button class="icon-btn" data-act="redo" title="Regenerate">${icon('i-redo')}</button>
      <button class="label-btn" data-act="other" title="Send the same prompt to ${NAMES[other]}">${icon('i-swap')}Ask ${NAMES[other]}</button>`;

    $('[data-act=copy]', actions).addEventListener('click', (e) => {
      const btn = e.currentTarget;
      navigator.clipboard.writeText(text).then(() => {
        btn.innerHTML = icon('i-check');
        toast('Copied to clipboard');
        setTimeout(() => (btn.innerHTML = icon('i-copy')), 1500);
      });
    });
    $('[data-act=redo]', actions).addEventListener('click', () => prompt && send(mode, prompt));
    $('[data-act=other]', actions).addEventListener('click', () => prompt && send(other, prompt));
  }

  function updateLive(mode, point) {
    panes[mode].liveKv.textContent = point ? `${fmt(point.kv)} MB` : '0.00 MB';
    panes[mode].liveCtx.textContent = point ? point.ctx : '0';
  }

  function renderAttachments(mode) {
    const p = panes[mode];
    p.attachmentsEl.innerHTML = '';
    p.attachedFiles.forEach((file, idx) => {
      const chip = document.createElement('div');
      chip.className = 'file-chip';
      const meta = file.pages && file.pages > 1 ? ` · ${file.pages} pages` : (file.size ? ` · ${Math.round(file.size / 1024)} KB` : '');
      chip.innerHTML = `<span>📄 ${escapeHtml(file.name)}${meta}</span><span class="remove-file" title="Remove">✕</span>`;
      chip.querySelector('.remove-file').addEventListener('click', (e) => {
        e.stopPropagation();
        p.attachedFiles.splice(idx, 1);
        renderAttachments(mode);
        updateSendState(mode);
      });
      p.attachmentsEl.appendChild(chip);
    });
  }

  function updateSendState(mode) {
    const p = panes[mode];
    const streaming = !!state.streams[mode];
    const hasContent = p.input.value.trim().length > 0 || (p.attachedFiles && p.attachedFiles.length > 0);
    p.send.classList.toggle('stop', streaming);
    p.send.classList.toggle('ready', !streaming && hasContent);
    p.send.innerHTML = icon(streaming ? 'i-stop' : 'i-send');
    p.send.title = streaming ? 'Stop' : 'Send';
  }

  function isBusy() {
    return MODES.some((m) => state.streams[m]);
  }

  // ---------------------------------------------------------
  // Generation
  // ---------------------------------------------------------
  function send(mode, prompt) {
    const p = panes[mode];
    prompt = (prompt || '').trim();
    if (!prompt && (!p.attachedFiles || !p.attachedFiles.length)) return;

    let displayPrompt = prompt;
    let fullPromptText = prompt;
    if (p.attachedFiles && p.attachedFiles.length > 0) {
      const fileNames = p.attachedFiles.map((f) => f.name).join(', ');
      const fileBlocks = p.attachedFiles
        .map((f) => `[Attached File: ${f.name}]\n\`\`\`\n${f.content}\n\`\`\``)
        .join('\n\n');
      fullPromptText = prompt ? `${fileBlocks}\n\nUser Question:\n${prompt}` : fileBlocks;
      displayPrompt = prompt ? `📄 [${fileNames}]\n${prompt}` : `📄 [${fileNames}]`;
      p.attachedFiles = [];
      renderAttachments(mode);
    }

    const other = mode === 'kvops' ? 'baseline' : 'kvops';
    if (state.streams[other]) return toast(`Wait for ${NAMES[other]} to finish first`);
    if (state.streams[mode]) return;

    const session = ensureSession(displayPrompt);
    if (!session[mode].length) p.messages.innerHTML = '';

    session[mode].push({ role: 'user', text: fullPromptText });
    p.messages.appendChild(userEl(displayPrompt));

    const el = botEl(mode);
    el.classList.add('generating');
    const content = $('.content', el);
    content.innerHTML = '<span class="thinking">Thinking</span>';
    content.classList.add('cursor');
    p.messages.appendChild(el);
    p.messages.scrollTop = p.messages.scrollHeight;

    p.input.value = '';
    autoGrow(p.input);

    state.series[mode] = [];
    drawCharts();

    const history = session[mode].map((m) => ({
      role: m.role === 'bot' ? 'assistant' : 'user',
      content: m.text,
    }));

    const controller = new AbortController();
    state.streams[mode] = controller;
    updateSendState(mode);

    const started = performance.now();
    let text = '';
    let last = null;
    let finished = false;
    let renderQueued = false;

    const flush = () => {
      renderQueued = false;
      const nearBottom = p.messages.scrollHeight - p.messages.scrollTop - p.messages.clientHeight < 120;
      content.innerHTML = renderMarkdown(text);
      if (nearBottom) p.messages.scrollTop = p.messages.scrollHeight;
      drawCharts();
    };

    const finish = (stopped = false, error = null) => {
      if (finished) return;
      finished = true;
      try { controller.abort(); } catch (_) {}
      state.streams[mode] = null;
      updateSendState(mode);

      const series = state.series[mode];
      const stats = last ? {
        tokens: last.step,
        kvMb: last.kv_cache_mb,
        peakKv: Math.max(...series.map((s) => s.kv)),
        peakGpu: Math.max(...series.map((s) => s.gpu)),
        avgLatency: series.reduce((a, s) => a + s.lat, 0) / (series.length || 1),
        evicted: last.tokens_evicted || 0,
        savedPct: last.saved_pct || 0,
        ctx: last.total_tokens,
        seconds: (performance.now() - started) / 1000,
        stopped,
      } : null;

      const finalText = error ? `Error: ${error}` : text;
      finalizeBot(el, mode, finalText, stats, prompt);
      session[mode].push({ role: 'bot', text: finalText, stats });
      session.updated = Date.now();
      session.series = session.series || {};
      session.summary = session.summary || {};
      session.series[mode] = series;
      session.summary[mode] = stats;
      state.summary[mode] = stats;

      state.sessions = [session, ...state.sessions.filter((s) => s.id !== session.id)];
      persist(session);
      renderHistory();
      drawCharts();
      renderCompare();
    };

    el._stop = () => finish(true);
    p._active = el;

    fetch('/api/stream_single', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: history,
        mode,
        max_tokens: 1024,
        budget_slots: mode === 'kvops' ? parseInt(settings.budget.value) : 0,
        temperature: parseFloat(settings.temp.value),
        model_name: settings.model.value,
      }),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          const err = await response.text();
          return finish(false, `HTTP ${response.status}: ${err}`);
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split('\n\n');
          buffer = parts.pop();
          for (const part of parts) {
            const lines = part.split('\n');
            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed.startsWith('data: ')) {
                try {
                  const data = JSON.parse(trimmed.slice(6));
                  if (data.error) return finish(false, data.error);
                  if (data.is_complete) return finish(false);

                  last = data;
                  text = cleanText(data.full_text);
                  const point = {
                    step: data.step,
                    kv: data.kv_cache_mb,
                    gpu: data.gpu_allocated_mb,
                    lat: data.step_time_ms,
                    ctx: data.total_tokens,
                  };
                  state.series[mode].push(point);
                  updateLive(mode, point);
                  if (!renderQueued) {
                    renderQueued = true;
                    requestAnimationFrame(flush);
                  }
                } catch (_) {}
              }
            }
          }
        }
        finish(false);
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        finish(false, last ? null : 'Could not reach the server.');
      });
  }

  function stop(mode) {
    const el = panes[mode]._active;
    if (el && el._stop) el._stop();
  }

  // ---------------------------------------------------------
  // Charts (lightweight SVG line charts with hover tooltips)
  // ---------------------------------------------------------
  function niceMax(v) {
    if (v <= 0) return 1;
    const exp = Math.pow(10, Math.floor(Math.log10(v)));
    const f = v / exp;
    const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
    return nf * exp;
  }

  function drawChart(container, key, opts = {}) {
    const W = container.clientWidth || 340;
    const H = container.clientHeight || 170;
    const pad = { l: 40, r: 8, t: 8, b: 22 };
    const iw = W - pad.l - pad.r;
    const ih = H - pad.t - pad.b;

    const series = MODES.map((m) => ({ mode: m, pts: state.series[m].map((p) => ({ x: p.step, y: p[key] })) }))
      .filter((s) => s.pts.length);

    const svgNS = 'http://www.w3.org/2000/svg';
    container.innerHTML = '';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    container.appendChild(svg);

    if (!series.length) {
      svg.innerHTML = `<text class="empty" x="${W / 2}" y="${H / 2}" text-anchor="middle">Send a prompt to see live data</text>`;
      return;
    }

    const all = series.flatMap((s) => s.pts);
    const xMax = Math.max(10, ...all.map((p) => p.x));
    let yMin = opts.zero ? 0 : Math.min(...all.map((p) => p.y));
    let yMaxRaw = Math.max(...all.map((p) => p.y));
    if (!opts.zero) {
      const span = Math.max(yMaxRaw - yMin, 1);
      yMin = Math.max(0, Math.floor(yMin - span * 0.15));
      yMaxRaw = yMaxRaw + span * 0.15;
    }
    const yMax = opts.zero ? niceMax(yMaxRaw * 1.08) : Math.ceil(yMaxRaw);
    const X = (x) => pad.l + (x / xMax) * iw;
    const Y = (y) => pad.t + ih - ((y - yMin) / (yMax - yMin || 1)) * ih;

    let html = '<defs>';
    series.forEach((s) => {
      html += `<linearGradient id="g-${key}-${s.mode}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="${COLORS[s.mode]}" stop-opacity=".22"/>
        <stop offset="1" stop-color="${COLORS[s.mode]}" stop-opacity="0"/></linearGradient>`;
    });
    html += '</defs><g class="grid">';
    const ticks = 4;
    let axis = '<g class="axis">';
    for (let i = 0; i <= ticks; i++) {
      const v = yMin + ((yMax - yMin) * i) / ticks;
      const y = Y(v);
      html += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${y}" y2="${y}"/>`;
      const label = (yMax - yMin) < 5 ? v.toFixed(1) : Math.round(v);
      axis += `<text x="${pad.l - 6}" y="${y + 3}" text-anchor="end">${label}</text>`;
    }
    axis += `<text x="${pad.l}" y="${H - 4}">0</text><text x="${W - pad.r}" y="${H - 4}" text-anchor="end">${xMax} tok</text></g>`;
    html += '</g>' + axis;

    series.forEach((s) => {
      const d = s.pts.map((p, i) => `${i ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join('');
      const lastP = s.pts[s.pts.length - 1];
      const area = `${d}L${X(lastP.x).toFixed(1)},${Y(yMin)}L${X(s.pts[0].x).toFixed(1)},${Y(yMin)}Z`;
      html += `<path d="${area}" fill="url(#g-${key}-${s.mode})"/>`;
      html += `<path d="${d}" fill="none" stroke="${COLORS[s.mode]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
      html += `<circle cx="${X(lastP.x)}" cy="${Y(lastP.y)}" r="3" fill="${COLORS[s.mode]}"/>`;
    });

    html += `<line class="hover-line" x1="0" x2="0" y1="${pad.t}" y2="${pad.t + ih}" stroke="#5f6368" stroke-dasharray="3 3" opacity="0"/>`;
    html += `<rect x="${pad.l}" y="${pad.t}" width="${iw}" height="${ih}" fill="transparent" class="hit"/>`;
    svg.innerHTML = html;

    // Hover tooltip
    const tip = document.createElement('div');
    tip.className = 'chart-tip';
    container.appendChild(tip);
    const hl = svg.querySelector('.hover-line');
    const hit = svg.querySelector('.hit');
    hit.addEventListener('mousemove', (e) => {
      const rect = svg.getBoundingClientRect();
      const sx = ((e.clientX - rect.left) / rect.width) * W;
      const step = Math.round(((sx - pad.l) / iw) * xMax);
      const rows = series.map((s) => {
        const p = s.pts.reduce((best, q) => (Math.abs(q.x - step) < Math.abs(best.x - step) ? q : best), s.pts[0]);
        return `<div><span style="color:${COLORS[s.mode]}">●</span> ${NAMES[s.mode]}: ${fmt(p.y, opts.digits ?? 2)} ${opts.unit || ''}</div>`;
      });
      tip.innerHTML = `<div style="color:#8e918f">token ${step}</div>${rows.join('')}`;
      tip.style.left = `${(sx / W) * 100}%`;
      tip.style.top = `${(pad.t / H) * 100 + 10}%`;
      tip.style.opacity = 1;
      hl.setAttribute('x1', sx);
      hl.setAttribute('x2', sx);
      hl.setAttribute('opacity', 1);
    });
    hit.addEventListener('mouseleave', () => {
      tip.style.opacity = 0;
      hl.setAttribute('opacity', 0);
    });
  }

  function drawCharts() {
    if (!$('#stats').classList.contains('open')) return;
    drawChart($('#chartKv'), 'kv', { zero: true, unit: 'MB' });
    drawChart($('#chartGpu'), 'gpu', { unit: 'MB', digits: 0 });
    drawChart($('#chartLat'), 'lat', { zero: true, unit: 'ms', digits: 1 });
  }

  function renderCompare() {
    const k = state.summary.kvops;
    const b = state.summary.baseline;
    const row = (label, fn, d = 2, suffix = '') => {
      const kv = k ? fmt(fn(k), d) + suffix : '—';
      const bv = b ? fmt(fn(b), d) + suffix : '—';
      return `<tr><td>${label}</td><td class="k">${kv}</td><td class="b">${bv}</td></tr>`;
    };
    $('#compareBody').innerHTML = [
      row('Tokens generated', (s) => s.tokens, 0),
      row('Peak KV cache', (s) => s.peakKv ?? s.kvMb, 2, ' MB'),
      row('Final KV cache', (s) => s.kvMb, 2, ' MB'),
      row('Tokens evicted', (s) => s.evicted, 0),
      row('Avg latency', (s) => s.avgLatency, 1, ' ms'),
      row('Total GPU VRAM (Model + KV)', (s) => s.peakGpu, 0, ' MB'),
      row('Speed', (s) => s.tokens / s.seconds, 1, ' tok/s'),
    ].join('');
  }

  // ---------------------------------------------------------
  // Hardware info
  // ---------------------------------------------------------
  function refreshHardware() {
    fetch('/api/hardware_info')
      .then((r) => r.json())
      .then((hw) => {
        const name = hw.has_cuda ? hw.device_name.replace('NVIDIA GeForce ', '') : 'CPU';
        $('#deviceName').textContent = hw.has_cuda ? `${name} · CUDA` : 'CPU mode';
        $('#gpuName').textContent = name;
        const used = hw.reserved_vram_mb || hw.allocated_vram_mb || 0;
        const total = hw.total_vram_mb || 0;
        $('#vramUsed').textContent = `${Math.round(used)} MB reserved by PyTorch`;
        $('#vramTotal').textContent = `${Math.round(total)} MB`;
        $('#vramFill').style.width = total ? `${Math.min(100, (used / total) * 100)}%` : '0';

        if (Array.isArray(hw.cached_models) && !settings.model.dataset.loaded) {
          settings.model.dataset.loaded = 'true';
          settings.model.innerHTML = '';
          hw.cached_models.forEach((m) => {
            const o = document.createElement('option');
            o.value = m.id;
            o.textContent = m.name.split('•')[0].trim();
            if (m.id === hw.current_model) o.selected = true;
            settings.model.appendChild(o);
          });
        }
      })
      .catch(() => { $('#deviceName').textContent = 'Server offline'; });
  }

  // ---------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------
  MODES.forEach((mode) => {
    const p = panes[mode];
    p.input.addEventListener('input', () => {
      autoGrow(p.input);
      updateSendState(mode);
    });
    p.input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        send(mode, p.input.value);
      }
    });
    p.send.addEventListener('click', () => (state.streams[mode] ? stop(mode) : send(mode, p.input.value)));

    p.attachBtn.addEventListener('click', () => p.fileInput.click());
    p.fileInput.addEventListener('change', async (e) => {
      const files = Array.from(e.target.files || []);
      if (!files.length) return;
      for (const file of files) {
        const ext = (file.name.split('.').pop() || '').toLowerCase();
        if (['pdf', 'docx', 'doc'].includes(ext)) {
          try {
            const formData = new FormData();
            formData.append('file', file);
            const res = await fetch('/api/extract_text', { method: 'POST', body: formData });
            if (res.ok) {
              const data = await res.json();
              p.attachedFiles.push({
                name: data.filename || file.name,
                size: data.size || file.size,
                pages: data.pages || 1,
                content: data.content || '',
                type: ext,
              });
            } else {
              toast(`Failed to read ${file.name}`);
            }
          } catch (err) {
            console.error('PDF/DOCX extraction error:', err);
            toast(`Error reading ${file.name}`);
          }
        } else {
          await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (evt) => {
              p.attachedFiles.push({
                name: file.name,
                size: file.size,
                content: evt.target.result,
                type: ext,
              });
              resolve();
            };
            reader.readAsText(file);
          });
        }
      }
      renderAttachments(mode);
      updateSendState(mode);
      p.fileInput.value = '';
      p.input.focus();
    });

    updateSendState(mode);
  });

  $('#newChatBtn').addEventListener('click', newChat);
  $('#toggleSidebar').addEventListener('click', () => $('#sidebar').classList.toggle('collapsed'));

  const toggleStats = (open) => {
    const el = $('#stats');
    el.classList.toggle('open', open ?? !el.classList.contains('open'));
    $('#toggleStats').classList.toggle('active', el.classList.contains('open'));
    if (el.classList.contains('open')) {
      setTimeout(drawCharts, 220); // after width transition
      refreshHardware();
    }
  };
  $('#toggleStats').addEventListener('click', () => toggleStats());
  $('#closeStats').addEventListener('click', () => toggleStats(false));

  settings.budget.addEventListener('input', () => ($('#budgetVal').textContent = settings.budget.value));
  settings.temp.addEventListener('input', () => ($('#tempVal').textContent = Number(settings.temp.value).toFixed(2)));

  window.addEventListener('resize', () => drawCharts());
  setInterval(() => {
    if ($('#stats').classList.contains('open')) refreshHardware();
  }, 2000);

  // Init
  loadSessions();
  newChat();
  renderCompare();
  refreshHardware();
})();
