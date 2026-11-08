// 벤치마크 게임: 작은 오델로. 상대 돌을 내 돌 사이에 끼우면 뒤집습니다.
// 중력 N목과 달리 놓은 돌의 색이 바뀌고, 판 전체의 돌 수로 승패가 나서 "덜 닮은 것"으로 씁니다 (docs/plan.md).
//
// 흑(0)이 먼저 둡니다. 가운데 네 칸에 백·흑 / 흑·백으로 놓고 시작합니다.
// 상대 돌을 하나 이상 뒤집는 칸에만 둘 수 있고, 둘 곳이 없으면 차례가 저절로 넘어갑니다 (특수 행동 없음).
// 둘 다 둘 곳이 없으면 끝나고, 돌이 많은 쪽이 이깁니다. 같으면 무승부.
// 칸 번호는 i = row * n + col 입니다.

const DIRS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
const STONE = ['black', 'white'];

const TEXT = {
  ko: {
    more: '돌이 더 많음', even: '돌 수가 같음',
    turn: (s) => `흑 ${s[0]} · 백 ${s[1]}`,
    passed: (s) => `상대가 둘 곳이 없어 한 번 더 둡니다 · 흑 ${s[0]} · 백 ${s[1]}`,
  },
  en: {
    more: 'More stones', even: 'Same number of stones',
    turn: (s) => `Black ${s[0]} · White ${s[1]}`,
    passed: (s) => `Your opponent has no move, so you play again · Black ${s[0]} · White ${s[1]}`,
  },
};

/**
 * @typedef {{ board: (0 | 1 | null)[], turn: 0 | 1, last: number | null, passed: boolean, over: boolean }} State
 *   passed: 상대가 둘 곳이 없어 같은 쪽이 이어서 두는 중 (화면 안내용)
 */

/** @param {string} id @param {number} n 판 크기 (짝수) */
export function othello(id, n) {
  const idx = (r, c) => r * n + c;

  /** player가 i에 두면 뒤집히는 칸들 */
  function flips(board, i, player) {
    if (board[i] !== null) return [];
    const r0 = Math.floor(i / n);
    const c0 = i % n;
    const out = [];
    for (const [dr, dc] of DIRS) {
      const line = [];
      let r = r0 + dr;
      let c = c0 + dc;
      while (r >= 0 && r < n && c >= 0 && c < n && board[idx(r, c)] === 1 - player) {
        line.push(idx(r, c));
        r += dr;
        c += dc;
      }
      if (line.length && r >= 0 && r < n && c >= 0 && c < n && board[idx(r, c)] === player) out.push(...line);
    }
    return out;
  }

  const movesOf = (board, player) => board.flatMap((_, i) => (flips(board, i, player).length ? [i] : []));
  const count = (board) => [board.filter((x) => x === 0).length, board.filter((x) => x === 1).length];

  function setup() {
    const board = Array(n * n).fill(null);
    const m = n / 2;
    board[idx(m - 1, m - 1)] = 1;
    board[idx(m - 1, m)] = 0;
    board[idx(m, m - 1)] = 0;
    board[idx(m, m)] = 1;
    return { board, turn: 0, last: null, passed: false, over: false };
  }

  function play(state, player, action) {
    if (state.over || player !== state.turn || action.kind !== 'click') return state;
    const { row, col } = action;
    if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || row >= n || col < 0 || col >= n) return state;
    const i = idx(row, col);
    const f = flips(state.board, i, player);
    if (!f.length) return state;
    const board = state.board.slice();
    board[i] = player;
    for (const j of f) board[j] = player;
    // 상대가 둘 곳이 있으면 상대 차례, 없으면 내가 한 번 더, 둘 다 없으면 끝
    if (movesOf(board, 1 - player).length) return { board, turn: 1 - player, last: i, passed: false, over: false };
    if (movesOf(board, player).length) return { board, turn: player, last: i, passed: true, over: false };
    return { board, turn: player, last: i, passed: false, over: true };
  }

  function result(state, lang = 'ko') {
    if (!state.over) return null;
    const [b, w] = count(state.board);
    if (b === w) return { winner: null, reason: TEXT[lang].even };
    return { winner: b > w ? 0 : 1, reason: TEXT[lang].more };
  }

  function legal(state, player) {
    if (state.over || player !== state.turn) return [];
    return movesOf(state.board, player).map((i) => ({ kind: 'click', row: Math.floor(i / n), col: i % n }));
  }

  function view(state, player, lang = 'ko') {
    const pieces = [];
    state.board.forEach((x, i) => {
      if (x !== null) pieces.push({ row: Math.floor(i / n), col: i % n, piece: STONE[x] });
    });
    const marks = [];
    if (state.last !== null) marks.push({ row: Math.floor(state.last / n), col: state.last % n, type: 'last-move' });
    // 둘 차례면 둘 수 있는 칸에 점 (오델로 화면의 보통 표시. AI에게도 보입니다)
    for (const a of legal(state, player)) marks.push({ row: a.row, col: a.col, type: 'dot' });
    const s = count(state.board);
    const message = state.over ? result(state, lang).reason : (state.passed && player === state.turn ? TEXT[lang].passed(s) : TEXT[lang].turn(s));
    return { pieces, marks, message };
  }

  return {
    id,
    board: { places: 'cells', rows: n, cols: n, look: 'plain' },
    pieces: 'go-stones',
    // 둘 때마다 빈칸이 하나 줄므로 이보다 길어질 수 없습니다
    maxActions: n * n,
    setup,
    play,
    result,
    turn: (state) => (state.over ? null : state.turn),
    legal,
    view,
  };
}
