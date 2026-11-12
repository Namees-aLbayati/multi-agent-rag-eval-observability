/* SVG workflow diagram. All labels are static; event data uses textContent. */
const Pipeline = (() => {
  const nodes = [
    ['start', 10, 20, 'User question', 'POST /api/run'],
    ['orchestrator', 205, 20, 'Orchestrator', 'route shared state'],
    ['parse_agent', 400, 20, 'Parse Agent', 'keyword parser'],
    ['scope', 595, 20, 'Scope validation', 'refine / reject'],
    ['rejected', 790, 20, 'Out of scope', 'return rejection'],
    ['response_agent', 205, 170, 'Response Agent', 'model / tool loop'],
    ['policy_search_tool', 400, 170, 'Policy search', 'k = 5 × attempt'],
    ['cases_search_tool', 595, 170, 'Case search', 'up to 4 chunks; once'],
    ['validate_chunks_tool', 790, 170, 'Evidence check', 'policy evidence only'],
    ['end', 10, 340, 'Orchestrator → END', 'return final output'],
    ['reply_tool', 400, 340, 'Grounded answer', 'reply_tool + citations'],
    ['retry', 595, 340, 'Expand search', 'attempt + 1; max 3'],
    ['insufficient', 790, 340, 'Insufficient evidence', 'third failed check']
  ];
  const edges = [
    'M180 50 H205', 'M375 50 H400', 'M570 50 H595', 'M765 50 H790',
    'M680 80 V120 H290 V80', 'M290 80 V170', 'M375 200 H400',
    'M570 200 H595', 'M765 200 H790',
    'M875 230 V280 H485 V340', 'M845 230 V260 H680 V340',
    'M680 340 V290 H485 V230', 'M925 230 V340',
    'M400 370 H180', 'M875 400 V440 H95 V400',
    'M960 50 H980 V460 H95 V400'
  ];
  let visits = {};
  let current = null;
  function initialize() {
    const root = document.querySelector('#pipeline-diagram');
    root.innerHTML = `<svg viewBox="0 0 1000 480" role="img" aria-labelledby="pipe-title pipe-desc"><title id="pipe-title">Policy-first execution workflow</title><desc id="pipe-desc">Question to orchestrator, parsing and scope validation. Accepted questions enter the response agent, policy search, case search, evidence validation, and grounded reply. Failed evidence validation retries policy search up to three attempts. Scope rejection and insufficient evidence terminate early.</desc><defs><marker id="pipe-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#a5b597"/></marker></defs>${edges.map((path, i) => `<path class="pipe-edge ${i === 10 ? 'retry-edge' : ''}" d="${path}"/>`).join('')}<text x="430" y="113" class="pipe-label">accepted → return to orchestrator</text><text x="785" y="40" class="pipe-label"></text><text x="500" y="272" class="pipe-label">RELEVANT</text><text x="615" y="254" class="pipe-label">NOT_RELEVANT</text><text x="490" y="307" class="pipe-label">retry policy only</text>${nodes.map(([id,x,y,title,sub]) => `<g class="pipe-node" data-pipe="${id}" transform="translate(${x},${y})"><rect width="170" height="60" rx="9"/><text x="11" y="25">${title}</text><text x="11" y="44" class="pipe-sub">${sub}</text><text x="145" y="15" class="visits"></text></g>`).join('')}</svg>`;
  }
  function reset(indexing = false) {
    visits = {};
    current = null;
    document.querySelectorAll('[data-pipe]').forEach(el => {
      el.classList.remove('active', 'visited', 'failed');
      el.querySelector('.visits').textContent = '';
    });
    document.querySelector('#execution-path').replaceChildren();
    document.querySelector('#pipeline-status').textContent = indexing ? 'INDEXING' : 'RUNNING';
    if (!indexing) mark('start', 'Question submitted');
  }
  function mark(id, title) {
    const el = document.querySelector(`[data-pipe="${id}"]`);
    if (!el) return;
    document.querySelectorAll('[data-pipe]').forEach(node => node.classList.remove('active'));
    el.classList.add('visited', 'active');
    visits[id] = (visits[id] || 0) + 1;
    el.querySelector('.visits').textContent = `×${visits[id]}`;
    current = id;
    const item = document.createElement('li');
    item.textContent = title;
    document.querySelector('#execution-path').append(item);
  }
  function update(event) {
    document.querySelector('#pipeline-current').textContent = event.title;
    if (event.step === 'error') {
      if (current) document.querySelector(`[data-pipe="${current}"]`)?.classList.add('failed');
      document.querySelector('#pipeline-status').textContent = 'FAILED';
      return;
    }
    if (event.step === 'orchestrator') {
      mark('orchestrator', event.title);
    } else if (event.step === 'parse_agent') {
      mark('parse_agent', 'Parse Agent');
    } else if (event.step === 'response_agent') {
      mark('response_agent', 'Response Agent');
    } else if (event.step === 'retry') {
      mark('retry', `Retry: attempt ${event.attempt}, ${event.candidates} candidates`);
    } else if (event.phase === 'running') {
      const id = event.step === 'validate_and_refine_parsed_query' ? 'scope' : event.step;
      if (id !== 'parse_query') mark(id, event.step.replaceAll('_', ' '));
    }
    if (event.step === 'validate_and_refine_parsed_query' && event.phase === 'done' && event.output.startsWith('OUT_OF_SCOPE')) {
      mark('rejected', 'Scope rejected');
    }
    if (event.step === 'validate_chunks_tool' && event.phase === 'done' && event.output.status === 'NOT_RELEVANT') {
      document.querySelector('#pipeline-current').textContent += ' — evidence insufficient';
    }
  }
  function finish(result) {
    if (result.validation_status === 'NOT_RELEVANT' && result.attempts === 3) mark('insufficient', 'Insufficient evidence after 3 attempts');
    mark('end', 'Final output returned');
    document.querySelectorAll('[data-pipe]').forEach(el => el.classList.remove('active'));
    document.querySelector('#pipeline-status').textContent = 'COMPLETE';
    document.querySelector('#pipeline-current').textContent = `${result.mode === 'demo' ? 'Simulated' : 'Live'} execution complete · ${result.attempts} retrieval attempt(s)`;
  }
  return { initialize, reset, update, finish };
})();
Pipeline.initialize();
