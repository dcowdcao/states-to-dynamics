// UI glue: board interaction + feature bars, all inference in-browser.

import { MOVE_TO_TOKEN, label, newBoard, apply, legalMoves } from "./othello.js";
import { OthelloGPT } from "./model.js";
import { makeEncoder, topFeatures } from "./sae.js";

const ALPHA = "ABCDEFGH", TOP_K = 20, CTX_LEN = 59;
const $ = id => document.getElementById(id);

let manifest, gpt, moves = [];
const encoders = new Map();

async function fetchProgress(url, onPct) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: ${resp.status}`);
  const total = +resp.headers.get("Content-Length") || 0;
  const reader = resp.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onPct(100 * got / total);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  return buf.buffer;
}

async function encoder(id) {
  if (!encoders.has(id)) {
    const m = manifest.saes.find(s => s.id === id);
    $("busy").textContent = `downloading ${m.display}…`;
    $("busy").style.visibility = "visible";
    const buf = await fetchProgress(`weights/${m.file}`,
      pct => $("busy").textContent = `downloading ${m.display}… ${pct.toFixed(0)}%`);
    encoders.set(id, makeEncoder(m, buf, manifest.cos_eps));
  }
  return encoders.get(id);
}

for (let c = 0; c < 8; c++) $("col-labels").insertAdjacentHTML("beforeend", `<div>${ALPHA[c]}</div>`);
for (let r = 0; r < 8; r++) $("row-labels").insertAdjacentHTML("beforeend", `<div>${r}</div>`);
for (let m = 0; m < 64; m++) {
  const sq = document.createElement("div");
  sq.className = "sq";
  sq.dataset.m = m;
  sq.addEventListener("click", () => {
    if (sq.classList.contains("legal")) { moves.push(m); refresh(); }
  });
  $("board").appendChild(sq);
}
$("undo").onclick = () => { if (moves.length) { moves.pop(); refresh(); } };
$("reset").onclick = () => { moves = []; refresh(); };
$("sae-select").onchange = refresh;

const fmt = v => v >= 100 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toFixed(4);

function renderBoard() {
  const board = newBoard();
  for (const m of moves) apply(board, m);
  const { moves: legal, mover } = legalMoves(board);
  const legalSet = new Set(legal), last = moves.length ? moves[moves.length - 1] : -1;
  const player = mover === 1 ? "black" : "white";
  document.querySelectorAll(".sq").forEach(sq => {
    const m = +sq.dataset.m, v = board.state[m];
    sq.classList.toggle("legal", legalSet.has(m));
    sq.innerHTML = v !== 0
      ? `<div class="disc ${v === 1 ? "black" : "white"}${m === last ? " last" : ""}"></div>`
      : (legalSet.has(m) ? `<div class="hint ${player}"></div>` : "");
  });
  let nb = 0, nw = 0;
  for (const v of board.state) { if (v === 1) nb++; else if (v === -1) nw++; }
  $("turn").textContent = legal.length ? `${player} to move` : "game over";
  $("counts").textContent = `● ${nb}  ○ ${nw}`;
  $("movelist").textContent = moves.map((m, i) => `${i + 1}.${label(m)}`).join("  ");
}

function renderFeatures(result) {
  $("empty").style.display = result ? "none" : "block";
  $("l0").textContent = result ? `L0 = ${result.l0} / ${manifest.saes[0].dict_size}` : "";
  const feats = result ? result.top : [];
  const max = feats.length ? feats[0].activation : 1;
  $("bars").innerHTML = feats.map(f => `
    <div class="frow" title="feature ${f.index} — activation ${f.activation}">
      <span class="fid">F${f.index}</span>
      <div class="track"><div class="fill" style="width:${100 * f.activation / max}%"></div></div>
      <span class="val">${fmt(f.activation)}</span>
    </div>`).join("");
}

let seq = 0;
async function refresh() {
  const s = ++seq;
  renderBoard();
  if (!moves.length) {
    renderFeatures(null);
    $("feat-sub").textContent = "at the latest move";
    $("busy").style.visibility = "hidden";
    return;
  }
  const encode = await encoder($("sae-select").value);
  if (s !== seq) return;
  $("busy").textContent = "running model…";
  $("busy").style.visibility = "visible";
  await new Promise(r => setTimeout(r));  // paint before the compute blocks
  // n_ctx is 59: on the 60th move of a finished game, drop the oldest token
  const resid = gpt.run(moves.slice(-CTX_LEN).map(m => MOVE_TO_TOKEN[m]));
  const result = topFeatures(encode(resid), TOP_K);
  if (s !== seq) return;
  $("busy").style.visibility = "hidden";
  renderFeatures(result);
  $("feat-sub").textContent =
    `${$("sae-select").selectedOptions[0].text} · after ${moves.length}. ${label(moves[moves.length - 1])}`;
}

(async () => {
  manifest = await (await fetch("weights/manifest.json")).json();
  $("sae-select").innerHTML = manifest.saes
    .map(s => `<option value="${s.id}">${s.display}</option>`).join("");
  $("busy").style.visibility = "visible";
  const buf = await fetchProgress(`weights/${manifest.model.file}`,
    pct => $("busy").textContent = `downloading model… ${pct.toFixed(0)}%`);
  gpt = new OthelloGPT(manifest.model, buf);
  $("busy").style.visibility = "hidden";
  refresh();
})();
