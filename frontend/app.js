const $ = selector => document.querySelector(selector);
let mode = 'demo';
let busy = false;
let lastRun = null;
let runEvents = [];
let activeQuery = '';

function text(tag, value, className) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

async function refreshStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Status unavailable');
    const s = await response.json();
    const rows = [
      ['Model credentials', s.openai_configured ? 'Configured' : 'Not configured', s.openai_configured],
      ['Source PDFs', s.pdfs.policy && s.pdfs.cases ? 'Both available' : 'Add PDFs to data/', s.pdfs.policy && s.pdfs.cases],
      ['Chroma index', s.index_ready ? 'Ready' : 'Not built', s.index_ready],
      ['LangSmith', s.tracing_enabled ? `Enabled · ${s.project}` : 'Off · enable in .env', s.tracing_enabled]
    ];
    $('#config-status').replaceChildren(...rows.map(([label, value, ok]) => {
      const row = text('div', '', 'status-row');
      row.append(text('span', label), text('strong', value, ok ? '' : 'off'));
      return row;
    }));
  } catch {
    $('#config-status').textContent = 'Server unavailable. Start the app using the README instructions.';
  }
}

function setBusy(value) {
  busy = value;
  $('#run-button').disabled = value;
  $('#index-button').disabled = value;
  document.querySelectorAll('.mode').forEach(button => { button.disabled = value; });
  $('#run-state').textContent = value ? 'RUNNING' : 'READY';
  $('#run-state').className = value ? 'pill running' : 'pill';
}

function addEvent(event) {
  runEvents.push(event);
  const item = text('div', '', 'event');
  const head = text('div', '', 'event-head');
  head.append(text('span', event.title), text('span', event.elapsed_ms != null ? `${(event.elapsed_ms / 1000).toFixed(1)}s` : '', 'event-time'));
  item.append(head, text('div', event.step.replaceAll('_', ' '), 'event-sub'));
  const details = document.createElement('details');
  details.append(text('summary', 'Inspect inputs & outputs'), text('pre', JSON.stringify(event, null, 2)));
  item.append(details);
  $('#journal').append(item);
  $('#journal').scrollTop = $('#journal').scrollHeight;
  const node = event.step === 'orchestrator' ? 'orchestrator' : event.step.includes('parse') ? 'parse_agent' : event.step === 'reply_tool' ? 'reply_tool' : 'response_agent';
  document.querySelectorAll('.road-step').forEach(el => el.classList.toggle('active', el.dataset.node === node));
}

function finish(result) {
  lastRun = { ...result, query: activeQuery, events: runEvents };
  $('#answer-section').hidden = false;
  $('#answer-label').textContent = result.mode === 'demo' ? 'SIMULATED DEMO RESPONSE' : 'FINAL DOCUMENT-BASED RESPONSE';
  $('#answer').textContent = result.answer;
  $('#run-meta').textContent = `${result.mode.toUpperCase()} · ${result.attempts} retrieval attempt(s) · ${(result.duration_ms / 1000).toFixed(1)}s`;
  $('#run-state').textContent = 'COMPLETE';
  $('#sources').replaceChildren();
  for (const chunk of result.sources) {
    const card = document.createElement('details');
    card.className = 'source';
    card.append(text('summary', `${chunk.source} · p. ${chunk.page}`), text('small', `Chunk ${chunk.id.slice(0, 16)} · distance ${Number(chunk.distance).toFixed(3)} (not a confidence score)`), text('p', chunk.text));
    $('#sources').append(card);
  }
  $('#sources-section').hidden = !result.sources.length;
  $('#source-count').textContent = `${result.sources.length} retrieved chunks`;
  if (result.trace_url) {
    const url = new URL(result.trace_url);
    if (url.protocol === 'https:' && (url.hostname === 'smith.langchain.com' || url.hostname.endsWith('.smith.langchain.com'))) {
      $('#trace-link').href = url.href;
      $('#trace-link').hidden = false;
    }
  }
}

async function stream(url, body) {
  const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { const data = await response.json(); message = typeof data.detail === 'string' ? data.detail : message; } catch { /* keep status */ }
    throw new Error(message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let terminal = false;
  function process(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === 'step') addEvent(event);
    if (event.type === 'error') throw new Error(event.message);
    if (event.type === 'result') { finish(event); terminal = true; }
    if (event.type === 'indexed') {
      addEvent({ step: 'index', title: `${event.reused ? 'Reused' : 'Built'} index: ${event.policy} policy + ${event.cases} case chunks` });
      terminal = true;
    }
  }
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    lines.forEach(process);
    if (done) break;
  }
  if (buffer.trim()) process(buffer);
  if (!terminal) throw new Error('Connection ended before completion. Check the server terminal.');
}

async function start(url, body) {
  if (busy) return;
  setBusy(true);
  activeQuery = body?.query || '';
  runEvents = [];
  lastRun = null;
  $('#journal').replaceChildren();
  $('#answer-section').hidden = true;
  $('#sources-section').hidden = true;
  $('#trace-link').hidden = true;
  $('#run-meta').textContent = url.includes('index') ? 'Building document embeddings…' : `${mode.toUpperCase()} · running`;
  let failed = false;
  try {
    await stream(url, body);
  } catch (error) {
    failed = true;
    addEvent({ step: 'error', title: error.message });
  } finally {
    setBusy(false);
    $('#run-state').textContent = failed ? 'FAILED' : 'COMPLETE';
    $('#run-state').className = failed ? 'pill error' : 'pill';
    document.querySelectorAll('.road-step').forEach(el => el.classList.remove('active'));
    await refreshStatus();
  }
}

$('#query-form').addEventListener('submit', event => {
  event.preventDefault();
  start('/api/run', { query: $('#query').value.trim(), mode, demo_retry: $('#demo-retry').checked });
});
$('#index-button').addEventListener('click', () => start('/api/index'));
document.querySelectorAll('.mode').forEach(button => button.addEventListener('click', () => {
  mode = button.dataset.mode;
  document.querySelectorAll('.mode').forEach(el => {
    el.classList.toggle('active', el === button);
    el.setAttribute('aria-pressed', String(el === button));
  });
  $('#mode-description').textContent = mode === 'demo' ? 'Simulated evidence. No API key, model calls, or legal advice.' : 'Uses your PDFs and paid model calls. Enable tracing in .env to inspect live runs in LangSmith.';
  $('#retry-option').hidden = mode !== 'demo';
}));
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => { $('#query').value = button.dataset.example; $('#query').focus(); }));
$('#explain-toggle').addEventListener('click', () => {
  const expanded = $('#explain-toggle').getAttribute('aria-expanded') === 'true';
  $('#explain-toggle').setAttribute('aria-expanded', String(!expanded));
  $('#workflow-detail').hidden = expanded;
});
$('#download-button').addEventListener('click', () => {
  if (!lastRun) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastRun, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url; a.download = 'tenantflow-run.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
refreshStatus();
