// 벤치마크 게임: 작은 바둑. 판이 커서 완전 탐색은 안 되고, 고정한 상대와 모델끼리의 대국으로 잽니다.
// 돌을 잡고(활로) 집을 세는 게임이라 중력 N목, 오델로와 가장 덜 닮았습니다 (docs/plan.md).
//
// 규칙은 AI 학습에서 흔히 쓰는 Tromp-Taylor 규칙입니다.
//   - 흑(0)이 먼저 둡니다. 빈 교차점에 두거나 통과(pass)합니다.
//   - 활로(맞닿은 빈 점)가 없어진 상대 돌무더기는 들어냅니다. 내 돌무더기의 활로가 없어지는 수(자살수)는 못 둡니다.
//   - 패: 돌 하나를 잡아 바로 되잡히는 모양이면, 상대는 바로 다음 수에 그 점에 못 둡니다 (단순 패. 화면에 어둡게).
//   - 둘이 연달아 통과하면 끝납니다. 판 위의 돌은 모두 산 것으로 보고, 내 돌 수 + 내 돌에만 닿은 빈 점 수가 내 점수입니다.
//   - 백은 덤(komi)을 더합니다. 덤에 반이 있어 비기지 않습니다.
// 5×5는 흑이 완벽하게 두면 25집을 다 차지한다고 알려져 있어서, 덤이 25보다 작으면 완벽한 수읽기로는 흑이 이깁니다.
// 학습용 판이 한쪽으로 너무 치우치지 않게 덤은 7.5로 둡니다.
// 점 번호는 i = row * n + col 입니다.

const STONE = ['black', 'white'];

const TEXT = {
  ko: {
    score: (b, w, komi) => `흑 ${b} · 백 ${w} + 덤 ${komi}`,
    turn: (who) => `${who === 0 ? '흑' : '백'} 차례`,
    passed: (who) => `${who === 0 ? '흑' : '백'} 차례 · 상대가 통과했습니다 (통과하면 끝납니다)`,
  },
  en: {
    score: (b, w, komi) => `Black ${b} · White ${w} + komi ${komi}`,
    turn: (who) => `${who === 0 ? 'Black' : 'White'} to play`,
    passed: (who) => `${who === 0 ? 'Black' : 'White'} to play · Your opponent passed (passing ends the game)`,
  },
};

/**
 * @typedef {{ board: (0 | 1 | null)[], turn: 0 | 1, last: number | null, passes: number, ko: number | null, over: boolean }} State
 *   passes: 연달아 통과한 횟수, ko: 둘 차례인 쪽이 이번에 못 두는 점 (단순 패)
 */

/** @param {string} id @param {number} n 판 줄 수 @param {{ komi?: number }} [options] */
export function go(id, n, { komi = 7.5 } = {}) {
  const N = n * n;
  const neighbors = Array.from({ length: N }, (_, i) => {
    const r = Math.floor(i / n);
    const c = i % n;
    return [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]
      .filter(([a, b]) => a >= 0 && a < n && b >= 0 && b < n)
      .map(([a, b]) => a * n + b);
  });

  /** i의 돌무더기와 활로 수 */
  function group(board, i) {
    const color = board[i];
    const stones = [i];
    const seen = new Set(stones);
    const liberties = new Set();
    for (let k = 0; k < stones.length; k++) {
      for (const j of neighbors[stones[k]]) {
        if (board[j] === null) liberties.add(j);
        else if (board[j] === color && !seen.has(j)) {
          seen.add(j);
          stones.push(j);
        }
      }
    }
    return { stones, liberties: liberties.size };
  }

  /** player가 i에 둔 뒤의 { board, ko }. 둘 수 없으면 null */
  function place(state, i, player) {
    if (state.board[i] !== null || i === state.ko) return null;
    const board = state.board.slice();
    board[i] = player;
    const captured = [];
    for (const j of neighbors[i]) {
      if (board[j] !== 1 - player) continue;
      const g = group(board, j);
      if (g.liberties === 0) {
        for (const s of g.stones) board[s] = null;
        captured.push(...g.stones);
      }
    }
    const mine = group(board, i);
    if (mine.liberties === 0) return null; // 자살수
    // 돌 하나를 잡았고, 둔 돌이 혼자 활로 하나(잡은 점)만 남았으면 상대가 바로 되잡을 수 없습니다
    const ko = captured.length === 1 && mine.stones.length === 1 && mine.liberties === 1 ? captured[0] : null;
    return { board, ko };
  }

  /** Tromp-Taylor 점수: [흑, 백] (돌 + 한쪽 돌에만 닿은 빈 점) */
  function score(board) {
    const s = [0, 0];
    for (const x of board) if (x !== null) s[x]++;
    const seen = new Set();
    for (let i = 0; i < N; i++) {
      if (board[i] !== null || seen.has(i)) continue;
      const region = [i];
      const touches = new Set();
      seen.add(i);
      for (let k = 0; k < region.length; k++) {
        for (const j of neighbors[region[k]]) {
          if (board[j] === null) {
            if (!seen.has(j)) {
              seen.add(j);
              region.push(j);
            }
          } else touches.add(board[j]);
        }
      }
      if (touches.size === 1) s[[...touches][0]] += region.length;
    }
    return s;
  }

  function setup() {
    return { board: Array(N).fill(null), turn: 0, last: null, passes: 0, ko: null, over: false };
  }

  function play(state, player, action) {
    if (state.over || player !== state.turn) return state;
    if (action.kind === 'pass') {
      const passes = state.passes + 1;
      return { ...state, turn: 1 - player, last: null, passes, ko: null, over: passes >= 2 };
    }
    if (action.kind !== 'click') return state;
    const { row, col } = action;
    if (!Number.isInteger(row) || !Number.isInteger(col) || row < 0 || row >= n || col < 0 || col >= n) return state;
    const i = row * n + col;
    const next = place(state, i, player);
    if (!next) return state;
    return { board: next.board, turn: 1 - player, last: i, passes: 0, ko: next.ko, over: false };
  }

  function result(state, lang = 'ko') {
    if (!state.over) return null;
    const [b, w] = score(state.board);
    return { winner: b > w + komi ? 0 : 1, reason: TEXT[lang].score(b, w, komi) };
  }

  function legal(state, player) {
    if (state.over || player !== state.turn) return [];
    const out = [];
    for (let i = 0; i < N; i++) {
      if (place(state, i, player)) out.push({ kind: 'click', row: Math.floor(i / n), col: i % n });
    }
    out.push({ kind: 'pass' });
    return out;
  }

  function view(state, player, lang = 'ko') {
    const pieces = [];
    state.board.forEach((x, i) => {
      if (x !== null) pieces.push({ row: Math.floor(i / n), col: i % n, piece: STONE[x] });
    });
    const marks = [];
    if (state.last !== null) marks.push({ row: Math.floor(state.last / n), col: state.last % n, type: 'last-move' });
    if (state.ko !== null && player === state.turn) marks.push({ row: Math.floor(state.ko / n), col: state.ko % n, type: 'blocked' });
    let message;
    if (state.over) message = result(state, lang).reason;
    else message = (state.passes === 1 ? TEXT[lang].passed : TEXT[lang].turn)(state.turn);
    return { pieces, marks, message };
  }

  /** 판에 안 보이는 것: 상대가 방금 통과했는지 (지금 통과하면 끝나서 점수대로 승패가 남) */
  function observe(state) {
    const v = state.passes === 1 ? 1 : 0;
    return [Array.from({ length: n }, () => Array(n).fill(v))];
  }

  return {
    id,
    board: { places: 'points', rows: n, cols: n, look: 'wood' },
    pieces: 'go-stones',
    specialActions: ['pass'],
    // 잡고 되잡으며 길어질 수 있어 넉넉히. 이르면 무승부입니다
    maxActions: n * n * 4,
    setup,
    play,
    result,
    turn: (state) => (state.over ? null : state.turn),
    legal,
    view,
    observe,
  };
}
