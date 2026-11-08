// 벤치마크 게임 규칙 테스트: node --test bench
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { loadEnv } from '../../boardmadang/trainer/load.js';
import { solve } from '../../boardmadang/trainer/solve.js';
import c3 from './c3-4x4/index.js';
import go5 from './go-5x5/index.js';
import m3 from './m3-4x4/index.js';
import oth from './othello-4x4/index.js';
import sh from './shogi-3x4/index.js';

const click = (row, col) => ({ kind: 'click', row, col });
const PASS = { kind: 'pass' };
const at = (dir) => fileURLToPath(new URL(dir, import.meta.url));

/** 차례대로 열을 눌러 둔 판 (중력 N목) */
function drops(game, cols) {
  let s = game.setup();
  for (const c of cols) s = game.play(s, s.turn, click(0, c));
  return s;
}

test('misère: the same three in a row that wins c3-4x4 loses m3-4x4', () => {
  // 선공이 0, 1, 2열 맨 아랫줄, 후공은 그 위
  const cols = [0, 0, 1, 1, 2];
  assert.equal(c3.result(drops(c3, cols)).winner, 0);
  assert.equal(m3.result(drops(m3, cols)).winner, 1);
});

test('othello: flips, legal moves and the start', () => {
  let s = oth.setup();
  assert.deepEqual(oth.legal(s, 0).map((a) => [a.row, a.col]), [[0, 1], [1, 0], [2, 3], [3, 2]]);
  s = oth.play(s, 0, click(0, 1)); // (1,1) 백을 뒤집음
  assert.equal(s.board[1 * 4 + 1], 0);
  assert.equal(s.turn, 1);
  assert.equal(oth.play(s, 1, click(3, 3)), s, 'a square that flips nothing is ignored');
});

test('othello: no move for the opponent means the same player plays again, none for both ends it', () => {
  // 흑: (0,0) (0,1), 백: (0,2) 하나. 흑이 (0,3)에 두면 백이 다 뒤집혀 아무도 둘 곳이 없음
  const board = Array(16).fill(null);
  board[0] = 0; board[1] = 0; board[2] = 1;
  let s = { board, turn: 0, last: null, passed: false, over: false };
  s = oth.play(s, 0, click(0, 3));
  assert.equal(s.over, true);
  assert.equal(oth.result(s, 'ko').winner, 0);
  // 흑 (0,0) (3,3), 백 (0,1) (3,2). 흑이 (0,2)에 두면 백은 둘 곳이 없고 흑은 (3,1)에 둘 수 있어 흑이 이어서 둠
  const b2 = Array(16).fill(null);
  b2[0] = 0; b2[1] = 1; b2[14] = 1; b2[15] = 0;
  s = oth.play({ board: b2, turn: 0, last: null, passed: false, over: false }, 0, click(0, 2));
  assert.deepEqual([s.turn, s.passed, s.over], [0, true, false]);
  assert.equal(oth.turn(s), 0);
  assert.deepEqual(oth.legal(s, 0).map((a) => [a.row, a.col]), [[3, 1]]);
  assert.match(oth.view(s, 0, 'ko').message, /한 번 더/);
});

test('othello 4x4: white wins with perfect play (known result)', async () => {
  const env = await loadEnv(at('./othello-4x4'));
  const solved = solve(env, env.start(0), { limit: 1_000_000 });
  assert.equal(solved.value, -1);
});

/** 바둑 판을 글자로: b 흑, w 백, . 빈 점 */
function goState(rows, turn = 0, extra = {}) {
  const board = rows.join('').split('').map((ch) => (ch === 'b' ? 0 : ch === 'w' ? 1 : null));
  return { board, turn, last: null, passes: 0, ko: null, over: false, ...extra };
}

test('go: capture, and suicide is not allowed', () => {
  // 백 (0,0)을 흑이 (1,0)으로 잡음
  let s = goState(['wb...', '.....', '.....', '.....', '.....']);
  s = go5.play(s, 0, click(1, 0));
  assert.equal(s.board[0], null);
  // 흑이 백에 둘러싸인 (0,0)에 두는 것은 자살수
  const t = goState(['.w...', 'w....', '.....', '.....', '.....']);
  assert.equal(go5.play(t, 0, click(0, 0)), t);
  assert.ok(!go5.legal(t, 0).some((a) => a.kind === 'click' && a.row === 0 && a.col === 0));
});

test('go: ko cannot be retaken right away', () => {
  // 흑이 (1,2)에 두어 백 (1,1)을 잡으면, 백은 바로 (1,1)에 되잡을 수 없음
  let s = goState([
    '.bw..',
    'bw.w.',
    '.bw..',
    '.....',
    '.....',
  ]);
  s = go5.play(s, 0, click(1, 2));
  assert.equal(s.board[1 * 5 + 1], null);
  assert.equal(s.ko, 1 * 5 + 1);
  assert.equal(go5.play(s, 1, click(1, 1)), s, 'retaking the ko is ignored');
  assert.ok(go5.view(s, 1, 'ko').marks.some((m) => m.type === 'blocked' && m.row === 1 && m.col === 1));
  // 다른 곳에 한 수씩 두면 다시 됩니다
  s = go5.play(go5.play(s, 1, click(4, 4)), 0, click(4, 0));
  assert.notEqual(go5.play(s, 1, click(1, 1)), s);
});

test('go: two passes end it, area scoring with komi', () => {
  // 흑이 왼쪽 두 줄 + 빈 점, 백이 오른쪽. 흑 10점(돌 5 + 집 5), 백 15점 + 7.5
  let s = goState([
    '.bw..',
    '.bw..',
    '.bw..',
    '.bw..',
    '.bw..',
  ]);
  s = go5.play(s, 0, PASS);
  assert.equal(go5.observe(s)[0][0][0], 1, 'the next player sees that the opponent passed');
  s = go5.play(s, 1, PASS);
  assert.equal(s.over, true);
  assert.deepEqual(go5.result(s, 'en'), { winner: 1, reason: 'Black 10 · White 15 + komi 7.5' });
  // 흑이 3줄을 가지면 15 대 10 + 7.5 → 백 승, 4줄이면 20 대 5 + 7.5 → 흑 승
  const four = goState(['...bw', '...bw', '...bw', '...bw', '...bw']);
  assert.equal(go5.result({ ...four, over: true }, 'en').winner, 0);
});

/** 3×4 장기 판을 글자로: 0행 후공의 손, 1~4행 판, 5행 선공의 손. 대문자 선공(흰), 소문자 후공(검은), . 빈칸 */
function shogiState(rows, turn = 0, hand = [[0, 0, 0], [0, 0, 0]]) {
  const board = rows.join('').split('').map((ch) => (ch === '.' ? null : (ch === ch.toUpperCase() ? 'w' : 'b') + ch.toUpperCase()));
  return { board, hand, turn, selected: null, last: [], over: false, winner: null, reason: null };
}

/** 고르고 옮기기 (두 번 누름) */
const shMove = (s, from, to) => sh.play(sh.play(s, s.turn, click(...from)), s.turn, click(...to));

test('shogi-3x4: capture puts the piece in hand, and a captured promoted pawn comes back as a pawn', () => {
  let s = sh.setup();
  s = shMove(s, [3, 1], [2, 1]); // 자가 자를 잡음
  assert.equal(s.board[2 * 3 + 1], 'wP');
  assert.deepEqual(s.hand, [[1, 0, 0], [0, 0, 0]]);
  assert.equal(s.turn, 1);
  // 후(Q)를 상(B)이 대각선으로 잡으면 손에는 자로
  const u = shMove(shogiState(['...', 'rkb', '.Q.', '...', 'BKR', '...'], 1), [1, 2], [2, 1]);
  assert.deepEqual(u.hand, [[0, 0, 0], [1, 0, 0]]);
});

test('shogi-3x4: a pawn reaching the far row promotes, drops cannot go on the far row', () => {
  let s = shogiState(['...', '..k', 'P..', '...', '.K.', '...'], 0, [[0, 0, 1], [0, 0, 0]]);
  s = shMove(s, [2, 0], [1, 0]);
  assert.equal(s.board[1 * 3 + 0], 'wQ');
  // 손의 장(5행 2열)을 고르면 상대 끝줄(1행)은 갈 곳에 없음
  const d = sh.play(shogiState(['...', '..k', '...', '...', '.K.', '...'], 0, [[0, 0, 1], [0, 0, 0]]), 0, click(5, 2));
  const targets = sh.legal(d, 0).filter((a) => a.row >= 1 && a.row <= 4).map((a) => a.row);
  assert.ok(targets.length > 0 && !targets.includes(1));
  const dropped = sh.play(d, 0, click(2, 0));
  assert.equal(dropped.board[2 * 3 + 0], 'wR');
  assert.deepEqual(dropped.hand[0], [0, 0, 0]);
});

test('shogi-3x4: a king that reaches the far row and survives a turn wins, a captured king loses', () => {
  // 흰 왕이 1행 0열로 들어감. 검은 왕(1행 2열)은 닿지 않음 → 검은 쪽이 아무 수나 두면 흰 쪽 승
  let s = shMove(shogiState(['...', '..k', 'K..', '...', '...', '...']), [2, 0], [1, 0]);
  assert.equal(s.over, false);
  s = shMove(s, [1, 2], [2, 2]);
  assert.deepEqual(sh.result(s, 'en'), { winner: 0, reason: 'King held the far row' });
  // 들어간 칸을 검은 왕이 잡을 수 있으면 잡고 이김
  let t = shMove(shogiState(['...', '.k.', 'K..', '...', '...', '...']), [2, 0], [1, 0]);
  t = shMove(t, [1, 1], [1, 0]);
  assert.deepEqual(sh.result(t, 'en'), { winner: 1, reason: 'King captured' });
});
