// 벤치마크 게임: 3×4 판의 작은 장기. 잡은 기물을 내 것으로 다시 놓을 수 있습니다 (십이장기와 같은 규칙).
// 기물을 옮기는 게임이라 "체스 기반" 묶음에 씁니다 (docs/plan.md 2단계). 판이 작아 학습이 빠릅니다.
//
// 기물 (보드마당의 체스 그림을 빌려 씁니다)
//   왕 = K: 8방향으로 한 칸        장 = R: 앞뒤좌우로 한 칸      상 = B: 대각선으로 한 칸
//   자 = P: 앞으로 한 칸           후 = Q: 자가 상대 끝줄에 닿으면 바뀜. 대각선 뒤 두 방향만 빼고 한 칸
// 잡은 기물은 내 손에 들어오고(후는 자로 돌아옴), 내 차례에 기물을 옮기는 대신 빈칸에 놓을 수 있습니다. 상대 끝줄에는 놓을 수 없습니다.
// 이기는 방법: 상대 왕을 잡거나, 내 왕이 상대 끝줄에 들어가 상대 차례가 지나도록 잡히지 않음.
// 무승부: 같은 판(판의 기물, 양쪽 손, 둘 차례)이 세 번째 나오거나, 300수 안에 끝나지 않음.
//
// 판은 6행 × 3열입니다. 0행은 후공의 손, 1~4행이 판, 5행은 선공의 손입니다. 칸 번호는 i = row * 3 + col.
// 손의 칸은 열마다 기물 하나 (0열 자, 1열 상, 2열 장)이고, 둘 이상이면 개수를 적습니다.
// 선공(0) = 흰 기물, 4행에서 시작해 위로 / 후공(1) = 검은 기물, 1행에서 시작해 아래로.

const ROWS = 6;
const COLS = 3;
const SIDE = ['w', 'b'];
const HOME = [4, 1]; // 플레이어의 끝줄 (상대가 들어오면 왕은 버티기, 자는 후)
const HAND_ROW = [5, 0];
const HAND = ['P', 'B', 'R']; // 손 칸의 열 순서
const FORWARD = [-1, 1];

const ORTHO = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const DIAG = [[-1, -1], [-1, 1], [1, -1], [1, 1]];

/** 기물이 갈 수 있는 방향. f는 그 기물 주인의 앞쪽 (행 방향) */
function steps(type, f) {
  if (type === 'K') return [...ORTHO, ...DIAG];
  if (type === 'R') return ORTHO;
  if (type === 'B') return DIAG;
  if (type === 'P') return [[f, 0]];
  return [...ORTHO, [f, -1], [f, 1]]; // Q (후)
}

const TEXT = {
  ko: {
    reason: { capture: '왕을 잡음', try: '왕이 상대 끝줄에서 버팀', 'no-moves': '상대가 둘 수 없음', repetition: '같은 판이 세 번 나옴' },
    won: (why) => `이겼습니다! (${why})`,
    lost: (why) => `졌습니다 (${why})`,
    draw: (why) => `무승부 (${why})`,
    theirTurn: '상대 차례',
    pick: '움직일 기물이나 놓을 포로를 고르세요',
    move: '갈 곳을 고르세요',
    drop: '놓을 빈칸을 고르세요 (상대 끝줄은 안 됨)',
  },
  en: {
    reason: {
      capture: 'King captured',
      try: 'King held the far row',
      'no-moves': 'Opponent has no move',
      repetition: 'Same position three times',
    },
    won: (why) => `You won! (${why})`,
    lost: (why) => `You lost (${why})`,
    draw: (why) => `Draw (${why})`,
    theirTurn: "Opponent's turn",
    pick: 'Choose a piece to move or a captured piece to drop',
    move: 'Choose a square',
    drop: "Choose an empty square to drop on (not your opponent's back row)",
  },
};

const idx = (r, c) => r * COLS + c;
const rowOf = (i) => Math.floor(i / COLS);
const colOf = (i) => i % COLS;
const onBoard = (r, c) => r >= 1 && r <= 4 && c >= 0 && c < COLS;
const click = (i) => ({ kind: 'click', row: rowOf(i), col: colOf(i) });

/**
 * @typedef {{
 *   board: (string | null)[],   // 18칸. 판(1~4행)에만 기물이 있고 손의 행은 늘 null. 'wK', 'bP' 같은 이름
 *   hand: number[][],           // [플레이어][자, 상, 장] 손에 든 개수
 *   turn: 0 | 1,
 *   selected: number | null,    // 고른 칸. 손의 행이면 그 기물을 놓으려는 것
 *   last: number[],
 *   over: boolean,
 *   winner: 0 | 1 | null,
 *   reason: 'capture' | 'try' | 'no-moves' | 'repetition' | null,
 *   seen: Record<string, number>, // 판(positionKey)마다 나온 횟수. 세 번째면 무승부
 * }} State
 */

/** 같은 판인지 가리는 열쇠: 둘 차례 + 판의 기물(선공은 대문자, 후공은 소문자, 빈칸은 .) + 양쪽 손의 개수 */
function positionKey(board, hand, turn) {
  let key = String(turn);
  for (const p of board) key += p ? (p[0] === 'w' ? p[1] : p[1].toLowerCase()) : '.';
  return `${key}${hand[0].join('')}${hand[1].join('')}`;
}

function setup() {
  const board = Array(ROWS * COLS).fill(null);
  // 내 쪽에서 볼 때 왼쪽에 상, 가운데 왕, 오른쪽에 장, 왕 앞에 자
  board[idx(4, 0)] = 'wB';
  board[idx(4, 1)] = 'wK';
  board[idx(4, 2)] = 'wR';
  board[idx(3, 1)] = 'wP';
  board[idx(1, 2)] = 'bB';
  board[idx(1, 1)] = 'bK';
  board[idx(1, 0)] = 'bR';
  board[idx(2, 1)] = 'bP';
  const hand = [[0, 0, 0], [0, 0, 0]];
  const seen = { [positionKey(board, hand, 0)]: 1 };
  return { board, hand, turn: 0, selected: null, last: [], over: false, winner: null, reason: null, seen };
}

/** player가 둘 수 있는 수 { from, to }. from이 손의 칸이면 놓기 */
function movesOf(state, player) {
  const out = [];
  const me = SIDE[player];
  for (let i = 0; i < ROWS * COLS; i++) {
    const p = state.board[i];
    if (!p || p[0] !== me) continue;
    for (const [dr, dc] of steps(p[1], FORWARD[player])) {
      const r = rowOf(i) + dr;
      const c = colOf(i) + dc;
      if (!onBoard(r, c)) continue;
      const q = state.board[idx(r, c)];
      if (!q || q[0] !== me) out.push({ from: i, to: idx(r, c) });
    }
  }
  HAND.forEach((_, k) => {
    if (!state.hand[player][k]) return;
    for (let i = 0; i < ROWS * COLS; i++) {
      const r = rowOf(i);
      if (onBoard(r, colOf(i)) && r !== HOME[1 - player] && !state.board[i]) out.push({ from: idx(HAND_ROW[player], k), to: i });
    }
  });
  return out;
}

/** 수를 두고 차례를 넘깁니다. 끝났는지도 여기서 정합니다 */
function apply(state, m) {
  const player = state.turn;
  const opp = 1 - player;
  const me = SIDE[player];
  const board = state.board.slice();
  const hand = state.hand.map((h) => h.slice());
  let captured = null;
  if (rowOf(m.from) === HAND_ROW[player]) {
    const k = colOf(m.from);
    hand[player][k]--;
    board[m.to] = me + HAND[k];
  } else {
    captured = board[m.to];
    let piece = board[m.from];
    if (captured && captured[1] !== 'K') hand[player][HAND.indexOf(captured[1] === 'Q' ? 'P' : captured[1])]++;
    if (piece[1] === 'P' && rowOf(m.to) === HOME[opp]) piece = `${me}Q`;
    board[m.to] = piece;
    board[m.from] = null;
  }
  const s = { ...state, board, hand, turn: opp, selected: null, last: [m.from, m.to] };
  const end = (winner, reason) => Object.assign(s, { over: true, winner, reason });
  if (captured?.[1] === 'K') return end(player, 'capture');
  // 상대 왕이 지난 차례에 내 끝줄에 들어왔는데 이번에 잡지 못했으면 상대가 이깁니다
  const theirKing = board.indexOf(`${SIDE[opp]}K`);
  if (rowOf(theirKing) === HOME[player]) return end(opp, 'try');
  if (movesOf(s, opp).length === 0) return end(player, 'no-moves');
  // 같은 판(판의 기물, 양쪽 손, 둘 차례)이 세 번째 나오면 무승부입니다. 잡은 기물이 다시 들어오는 게임이라
  // 양쪽이 같은 수를 오가면 끝나지 않기 때문입니다
  const key = positionKey(board, hand, opp);
  const times = (state.seen?.[key] ?? 0) + 1;
  s.seen = { ...state.seen, [key]: times };
  if (times >= 3) return end(null, 'repetition');
  return s;
}

function play(state, player, action) {
  if (state.over || player !== state.turn || action?.kind !== 'click') return state;
  const { row, col } = action;
  if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || row >= ROWS || col < 0 || col >= COLS) return state;
  const i = idx(row, col);
  const moves = movesOf(state, player);
  if (state.selected !== null) {
    const m = moves.find((x) => x.from === state.selected && x.to === i);
    if (m) return apply(state, m);
  }
  if (i !== state.selected && moves.some((x) => x.from === i)) return { ...state, selected: i };
  if (state.selected !== null) return { ...state, selected: null };
  return state;
}

function legal(state, player) {
  if (state.over || player !== state.turn) return [];
  const moves = movesOf(state, player);
  const out = new Set();
  if (state.selected !== null) for (const m of moves) if (m.from === state.selected) out.add(m.to);
  for (const m of moves) if (m.from !== state.selected) out.add(m.from);
  return [...out].map(click);
}

function result(state, lang = 'ko') {
  return state.over ? { winner: state.winner, reason: TEXT[lang].reason[state.reason] } : null;
}

function view(state, player, lang = 'ko') {
  const text = TEXT[lang];
  const at = (i) => ({ row: rowOf(i), col: colOf(i) });
  const pieces = [];
  const counts = [];
  const tint = [];
  for (let i = 0; i < ROWS * COLS; i++) if (state.board[i]) pieces.push({ ...at(i), piece: state.board[i] });
  for (const p of [0, 1]) {
    HAND.forEach((t, k) => {
      const cell = { row: HAND_ROW[p], col: k };
      tint.push({ ...cell, color: 'gray' });
      const n = state.hand[p][k];
      if (n > 0) pieces.push({ ...cell, piece: SIDE[p] + t });
      if (n > 1) counts.push({ ...cell, text: String(n) });
    });
  }
  const marks = state.last.map((i) => ({ ...at(i), type: 'last-move' }));
  const myTurn = !state.over && player === state.turn;
  if (myTurn && state.selected !== null) {
    marks.push({ ...at(state.selected), type: 'ring' });
    for (const m of movesOf(state, player)) if (m.from === state.selected) marks.push({ ...at(m.to), type: 'dot' });
  }

  let message;
  if (state.over) {
    const why = text.reason[state.reason];
    if (state.winner === null) message = text.draw(why);
    else message = state.winner === player ? text.won(why) : text.lost(why);
  } else if (!myTurn) message = text.theirTurn;
  else if (state.selected === null) message = text.pick;
  else message = rowOf(state.selected) === HAND_ROW[player] ? text.drop : text.move;
  return { pieces, tint, text: counts, marks, message };
}

/** @type {import('../../../boardmadang/sdk/game').Game<State>} */
export default {
  id: 'shogi-3x4',
  board: { places: 'cells', rows: ROWS, cols: COLS, look: 'wood', flipForPlayer: 1 },
  pieces: 'chess',
  maxActions: 300,
  setup,
  play,
  result,
  view,
  turn: (state) => (state.over ? null : state.turn),
  legal,
};
