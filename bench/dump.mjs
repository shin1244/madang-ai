// 벤치마크 게임을 완전 탐색해서 채점용 파일로 씁니다.
//
//   node bench/dump.mjs bench/games/c4-4x4 runs/c4-4x4.bin 60000
//
// 국면마다 신경망 입력(planes), 둘 수 있는 행동, 가장 좋은 행동, 둘 차례인 쪽 기준 정답 가치(+1 승, 0 무, −1 패)를 남깁니다.
// 국면이 많으면 고정 시드로 섞어 n개만 남깁니다. 우리가 만든 벤치마크 게임에만 씁니다 (게임 코드를 이 프로세스에서 그대로 돌림).
//
// 파일: [u32 머리 길이][머리 JSON][planes f32][legal u8][best u8][value f32][choice u8]
//   머리: { n, channels, rows, cols, A, game, channelNames }
//   choice는 둔 수에 따라 결과가 갈리는 국면 (가장 좋은 수가 따로 있음)

import { writeFileSync } from 'node:fs';
import { loadEnv } from '../boardmadang/trainer/load.js';
import { solve } from '../boardmadang/trainer/solve.js';

const [dir, out, count = '60000'] = process.argv.slice(2);
if (!dir || !out) {
  console.error('usage: node bench/dump.mjs <game folder> <out.bin> [positions]');
  process.exit(2);
}
const env = await loadEnv(dir);
const solved = solve(env, env.start(0), { limit: 5_000_000 });
if (!solved) throw new Error('too big to solve');
let cases = [...solved.table.values()].filter((e) => e.toPlay !== null);

let seed = 12345;
const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
for (let i = cases.length - 1; i > 0; i--) {
  const j = Math.floor(random() * (i + 1));
  [cases[i], cases[j]] = [cases[j], cases[i]];
}
cases = cases.slice(0, Number(count));

const size = env.channels * env.rows * env.cols;
const A = env.nActions;
const planes = new Float32Array(cases.length * size);
const legal = new Uint8Array(cases.length * A);
const best = new Uint8Array(cases.length * A);
const value = new Float32Array(cases.length);
const choice = new Uint8Array(cases.length);
cases.forEach((e, i) => {
  env.encode(e.pos, planes.subarray(i * size, (i + 1) * size));
  for (const m of env.children(e.pos)) legal[i * A + m.action] = 1;
  for (const a of e.best) best[i * A + a] = 1;
  value[i] = e.toPlay === 0 ? e.value : -e.value;
  choice[i] = e.best.length < e.moves ? 1 : 0;
});

// game과 channelNames는 채점할 때 모델이 이 게임의 것인지 보는 데 씁니다 (공통 모델은 game으로 경로를 고름)
const head = Buffer.from(JSON.stringify({
  n: cases.length, channels: env.channels, rows: env.rows, cols: env.cols, A, game: env.game.id, channelNames: env.channelNames(),
}));
const len = Buffer.alloc(4);
len.writeUInt32LE(head.length);
const bytes = (a) => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
writeFileSync(out, Buffer.concat([len, head, bytes(planes), bytes(legal), bytes(best), bytes(value), bytes(choice)]));
const matters = choice.reduce((s, x) => s + x, 0);
console.log(`${cases.length} positions of ${solved.table.size} (${matters} where the move matters): ${out}`);
