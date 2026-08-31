"""Single-page browser client for the multiplayer table (no build step, no CDN)."""

CLIENT_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>holdem-ml</title>
<style>
  :root {
    --felt: #10603f; --felt-dark: #0a3f2a; --ink: #10151a; --paper: #f6f5f0;
    --muted: #7d8894; --accent: #f0b429; --danger: #c8402f;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
         background: #11171d; color: var(--paper); }
  header { display: flex; align-items: baseline; gap: 12px; padding: 14px 20px;
           border-bottom: 1px solid #222c35; }
  h1 { font-size: 16px; margin: 0; letter-spacing: .06em; text-transform: uppercase; }
  .muted { color: var(--muted); font-size: 13px; }
  main { display: grid; grid-template-columns: minmax(0,1fr) 300px; gap: 18px;
         padding: 18px 20px; max-width: 1100px; margin: 0 auto; }
  @media (max-width: 820px) { main { grid-template-columns: 1fr; } }
  .table { background: radial-gradient(ellipse at 50% 40%, var(--felt), var(--felt-dark));
           border-radius: 18px; padding: 20px; min-height: 340px; }
  .seats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .seat { background: rgba(0,0,0,.28); border: 1px solid rgba(255,255,255,.10);
          border-radius: 10px; padding: 9px 11px; }
  .seat.turn { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .seat.folded { opacity: .42; }
  .seat .name { font-weight: 600; display: flex; justify-content: space-between; gap: 6px; }
  .seat .sub { font-size: 12px; color: #cfe4d8; }
  .tag { font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
         color: var(--accent); }
  .board { display: flex; gap: 8px; justify-content: center; margin: 22px 0 12px; min-height: 74px; }
  .card { width: 50px; height: 72px; border-radius: 7px; background: var(--paper);
          color: var(--ink); display: flex; flex-direction: column; align-items: center;
          justify-content: center; font-weight: 700; font-size: 19px; line-height: 1;
          box-shadow: 0 2px 6px rgba(0,0,0,.35); }
  .card.red { color: var(--danger); }
  .card .s { font-size: 17px; }
  .pot { text-align: center; font-size: 15px; letter-spacing: .04em; }
  .panel { background: #161d24; border: 1px solid #222c35; border-radius: 12px; padding: 14px; }
  .panel h2 { font-size: 12px; letter-spacing: .1em; text-transform: uppercase;
              color: var(--muted); margin: 0 0 8px; }
  button { font: inherit; border: 1px solid #2c3a46; background: #1d262f; color: var(--paper);
           border-radius: 8px; padding: 8px 12px; cursor: pointer; }
  button:hover:not(:disabled) { border-color: var(--accent); }
  button:disabled { opacity: .4; cursor: default; }
  button.primary { background: var(--accent); color: #1a1205; border-color: var(--accent);
                   font-weight: 600; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  input[type=text], input[type=number] { font: inherit; background: #0e141a; color: var(--paper);
    border: 1px solid #2c3a46; border-radius: 8px; padding: 8px 10px; width: 100%; }
  input[type=range] { width: 100%; }
  #log { font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; color: #9fb0bd;
         max-height: 260px; overflow-y: auto; white-space: pre-wrap; }
  .hole { display: flex; gap: 8px; }
  .hint { font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>holdem&#8209;ml</h1>
  <span class="muted" id="tagline">a table of machine-learning bots — sit down and play</span>
</header>
<main>
  <section>
    <div class="table">
      <div class="seats" id="seats"></div>
      <div class="board" id="board"></div>
      <div class="pot" id="pot"></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h2>your action</h2>
      <div id="joinBox" class="row">
        <input type="text" id="name" placeholder="your name" style="max-width:200px">
        <button class="primary" id="joinBtn">take a seat</button>
      </div>
      <div id="actionBox" hidden>
        <div class="row" id="hole"></div>
        <div class="hint" id="handInfo"></div>
        <div class="row" style="margin-top:10px" id="buttons"></div>
        <div id="raiseBox" hidden style="margin-top:10px">
          <input type="range" id="raiseRange">
          <div class="row">
            <input type="number" id="raiseAmount" style="max-width:120px">
            <button id="raiseBtn" class="primary">raise to</button>
            <span class="hint" id="raiseHint"></span>
          </div>
        </div>
        <div class="hint" id="waiting">waiting for the next hand…</div>
      </div>
    </div>
  </section>
  <aside>
    <div class="panel">
      <h2>hand log</h2>
      <div id="log"></div>
    </div>
  </aside>
</main>
<script>
const state = { playerId: localStorage.getItem('holdemPlayerId') || null, view: null };

function cardEl(text) {
  const el = document.createElement('div');
  el.className = 'card';
  if (!text) { el.textContent = '?'; return el; }
  const rank = text[0], suit = text[1];
  const glyph = { c: '♣', d: '♦', h: '♥', s: '♠' }[suit] || suit;
  if (suit === 'd' || suit === 'h') el.classList.add('red');
  el.innerHTML = '<span>' + rank + '</span><span class="s">' + glyph + '</span>';
  return el;
}

function render(view) {
  state.view = view;
  const seats = document.getElementById('seats');
  seats.innerHTML = '';
  view.players.forEach(p => {
    const el = document.createElement('div');
    el.className = 'seat' + (p.folded ? ' folded' : '') + (view.toAct === p.seat ? ' turn' : '');
    const tags = [];
    if (p.seat === view.button) tags.push('btn');
    if (p.human) tags.push('human');
    if (p.allIn) tags.push('all-in');
    el.innerHTML = '<div class="name"><span>' + p.name + '</span><span class="tag">' +
      tags.join(' ') + '</span></div><div class="sub">' + p.stack +
      (p.committed ? ' &middot; bet ' + p.committed : '') + '</div>';
    seats.appendChild(el);
  });

  const board = document.getElementById('board');
  board.innerHTML = '';
  view.board.forEach(c => board.appendChild(cardEl(c)));
  document.getElementById('pot').textContent =
    'pot ' + view.pot + '  ·  hand #' + view.handNumber +
    '  ·  blinds ' + view.blinds[0] + '/' + view.blinds[1];
  document.getElementById('log').textContent = (view.log || []).join('\n');

  document.getElementById('joinBox').hidden = !!view.you;
  document.getElementById('actionBox').hidden = !view.you;
  if (view.you) renderYou(view.you);
}

function renderYou(you) {
  const hole = document.getElementById('hole');
  hole.innerHTML = '';
  (you.hole || []).forEach(c => hole.appendChild(cardEl(c)));
  const buttons = document.getElementById('buttons');
  buttons.innerHTML = '';
  const raiseBox = document.getElementById('raiseBox');
  const waiting = document.getElementById('waiting');

  if (!you.yourTurn) {
    raiseBox.hidden = true;
    waiting.hidden = false;
    waiting.textContent = 'seat ' + (you.seat + 1) + ' · stack ' + you.stack +
      ' · waiting for your turn…';
    document.getElementById('handInfo').textContent = '';
    return;
  }
  waiting.hidden = true;
  let info = 'equity ' + Math.round((you.equity || 0) * 100) + '%';
  if (you.madeHand) info += ' · ' + you.madeHand;
  if (you.toCall) info += ' · to call ' + you.toCall +
    ' (pot odds ' + Math.round((you.potOdds || 0) * 100) + '%)';
  document.getElementById('handInfo').textContent = info;

  let raiseSpec = null;
  (you.legal || []).forEach(la => {
    if (la.type === 'bet' || la.type === 'raise') { raiseSpec = la; return; }
    const b = document.createElement('button');
    b.textContent = la.type === 'call' ? 'call ' + la.min : la.type;
    if (la.type === 'call' || la.type === 'check') b.className = 'primary';
    b.onclick = () => act(la.type, la.min || 0);
    buttons.appendChild(b);
  });
  if (raiseSpec) {
    const allIn = document.createElement('button');
    allIn.textContent = 'all-in ' + raiseSpec.max;
    allIn.onclick = () => act(raiseSpec.type, raiseSpec.max);
    buttons.appendChild(allIn);
    raiseBox.hidden = false;
    const range = document.getElementById('raiseRange');
    const amount = document.getElementById('raiseAmount');
    range.min = amount.min = raiseSpec.min;
    range.max = amount.max = raiseSpec.max;
    if (!(+amount.value >= raiseSpec.min && +amount.value <= raiseSpec.max)) {
      range.value = amount.value = raiseSpec.min;
    }
    range.oninput = () => { amount.value = range.value; };
    amount.oninput = () => { range.value = amount.value; };
    document.getElementById('raiseHint').textContent =
      'between ' + raiseSpec.min + ' and ' + raiseSpec.max;
    document.getElementById('raiseBtn').onclick =
      () => act(raiseSpec.type, +amount.value);
  } else {
    raiseBox.hidden = true;
  }
}

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  return res.json();
}

async function act(type, amount) {
  document.querySelectorAll('#buttons button, #raiseBtn').forEach(b => b.disabled = true);
  await post('/api/action', { playerId: state.playerId, type, amount });
  await poll();
}

async function poll() {
  const q = state.playerId ? '?playerId=' + encodeURIComponent(state.playerId) : '';
  try {
    const view = await fetch('/api/state' + q).then(r => r.json());
    render(view);
  } catch (err) { /* server restarting; try again on the next tick */ }
}

document.getElementById('joinBtn').onclick = async () => {
  const name = document.getElementById('name').value.trim() || 'Player';
  const res = await post('/api/join', { name });
  if (res.playerId) {
    state.playerId = res.playerId;
    localStorage.setItem('holdemPlayerId', res.playerId);
    await poll();
  } else {
    alert(res.error || 'could not join');
  }
};

poll();
setInterval(poll, 900);
</script>
</body>
</html>
"""
