// Port of circuits/othello_engine_utils.py OthelloBoardState (1 black, -1 white,
// black first), including the forfeit rule: if the mover has no flips, the move
// is played by the opponent instead.

const DIRS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
const CENTERS = new Set([27, 28, 35, 36]);

export const MOVE_TO_TOKEN = (() => {
  const t = new Array(64).fill(0);
  let tok = 1;
  for (let m = 0; m < 64; m++) if (!CENTERS.has(m)) t[m] = tok++;
  return t;
})();

export const label = m => "ABCDEFGH"[m % 8] + Math.floor(m / 8);

export function newBoard() {
  const state = new Int8Array(64);
  state[3 * 8 + 4] = 1; state[3 * 8 + 3] = -1;
  state[4 * 8 + 3] = 1; state[4 * 8 + 4] = -1;
  return { state, next: 1, history: [] };
}

function flips(state, move, color) {
  const r0 = Math.floor(move / 8), c0 = move % 8, out = [];
  for (const [dr, dc] of DIRS) {
    const buf = [];
    let r = r0 + dr, c = c0 + dc;
    while (r >= 0 && r < 8 && c >= 0 && c < 8) {
      const v = state[r * 8 + c];
      if (v === 0) break;
      if (v === color) { out.push(...buf); break; }
      buf.push(r * 8 + c);
      r += dr; c += dc;
    }
  }
  return out;
}

// 0 illegal, 1 regular move, 2 forfeit (opponent plays here)
export function tentative(board, move) {
  if (board.state[move] !== 0) return 0;
  if (flips(board.state, move, board.next).length) return 1;
  if (flips(board.state, move, -board.next).length) return 2;
  return 0;
}

export function legalMoves(board) {
  const regular = [], forfeit = [];
  for (let m = 0; m < 64; m++) {
    const t = tentative(board, m);
    if (t === 1) regular.push(m);
    else if (t === 2) forfeit.push(m);
  }
  return regular.length ? { moves: regular, mover: board.next }
                        : { moves: forfeit, mover: -board.next };
}

export function apply(board, move) {
  if (board.state[move] !== 0) throw new Error(`${label(move)} is occupied`);
  let color = board.next;
  let f = flips(board.state, move, color);
  if (!f.length) { color = -color; f = flips(board.state, move, color); }
  if (!f.length) throw new Error(`illegal move ${label(move)}`);
  for (const i of f) board.state[i] = color;
  board.state[move] = color;
  board.next = -color;
  board.history.push(move);
}
