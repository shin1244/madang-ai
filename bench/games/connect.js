// 벤치마크 게임: 중력 N목. 열의 아무 칸이나 누르면 그 열의 가장 아래 빈칸에 놓입니다.
// 판이 작아서 모든 국면의 정답을 완전 탐색으로 구할 수 있고, 그 정답으로 학습한 신경망을 채점합니다 (bench/grade.py).
// 보드마당 게임 API(boardmadang/sdk/game.d.ts)를 그대로 따르므로 사이트에 올리는 게임과 똑같이 학습됩니다.
// misere면 N개를 이은 쪽이 집니다 (미제르). 판과 관측이 원래 게임과 똑같고 뜻만 반대라, 공통 모델이
// 게임 벡터로 둘을 가려내는지 봅니다 (docs/plan.md의 "반대인 것").

const DIRS = [[0, 1], [1, 0], [1, 1], [1, -1]];

/**
 * @param {string} id
 * @param {number} rows
 * @param {number} cols
 * @param {number} n 이만큼 이으면 승리 (misere면 패배)
 * @param {{ misere?: boolean }} [options]
 */
export function connect(id, rows, cols, n, { misere = false } = {}) {
  const R = rows;
  const C = cols;

  /** i에 놓인 돌로 n개가 이어졌는지 */
  function lineAt(board, i) {
    const p = board[i];
    if (p === null) return false;
    const r = Math.floor(i / C);
    const c = i % C;
    const same = (rr, cc) => rr >= 0 && rr < R && cc >= 0 && cc < C && board[rr * C + cc] === p;
    for (const [dr, dc] of DIRS) {
      let k = 1;
      for (let s = 1; s < n && same(r + dr * s, c + dc * s); s++) k++;
      for (let s = 1; s < n && same(r - dr * s, c - dc * s); s++) k++;
      if (k >= n) return true;
    }
    return false;
  }

  /** 열 c에 놓으면 들어갈 칸. 가득 찼으면 -1 */
  function drop(board, c) {
    for (let r = R - 1; r >= 0; r--) if (board[r * C + c] === null) return r * C + c;
    return -1;
  }

  function result(state) {
    if (state.last !== null && lineAt(state.board, state.last)) {
      const p = state.board[state.last];
      return misere ? { winner: 1 - p, reason: 'misere' } : { winner: p, reason: 'line' };
    }
    if (state.board.every((x) => x !== null)) return { winner: null, reason: 'full' };
    return null;
  }

  return {
    id,
    board: { places: 'cells', rows: R, cols: C },
    pieces: ['red', 'blue'],
    maxActions: R * C,
    setup() {
      return { board: Array(R * C).fill(null), turn: 0, last: null };
    },
    play(state, player, action) {
      if (result(state) || player !== state.turn || action.kind !== 'click') return state;
      if (!Number.isInteger(action.col) || action.col < 0 || action.col >= C) return state;
      const i = drop(state.board, action.col);
      if (i < 0) return state;
      const board = state.board.slice();
      board[i] = player;
      return { board, turn: 1 - player, last: i };
    },
    turn(state) {
      return result(state) ? null : state.turn;
    },
    result,
    legal(state, player) {
      if (result(state) || player !== state.turn) return [];
      const out = [];
      for (let c = 0; c < C; c++) {
        const i = drop(state.board, c);
        if (i >= 0) out.push({ kind: 'click', row: Math.floor(i / C), col: c });
      }
      return out;
    },
    view(state) {
      const pieces = [];
      state.board.forEach((x, i) => {
        if (x !== null) pieces.push({ row: Math.floor(i / C), col: i % C, piece: x === 0 ? 'red' : 'blue' });
      });
      const marks = state.last === null ? [] : [{ row: Math.floor(state.last / C), col: state.last % C, type: 'last-move' }];
      return { pieces, marks };
    },
  };
}
