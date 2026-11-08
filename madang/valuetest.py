"""처음 보는 판에서 모델들의 가치 오차를 잽니다. 판 외우기(학습 판의 가치 손실만 낮고 새 판에서는 높음)를 비교할 때 씁니다.

    python -m madang.valuetest ../splat-chess runs/player/model.pt runs/a/model.pt runs/b/model.pt --games 200

첫 모델(player)이 학습 때와 같은 방식으로 자기대국을 두고 (탐색, 처음 몇 수 뽑기, 디리클레 잡음),
그 판의 모든 국면에서 나머지 모델들의 가치를 실제 승패(차례인 쪽 기준 +1/0/−1)와 비교합니다.
학습 로그의 "new games v"와 달리 --value-mix로 섞인 정답이 아니라 늘 실제 승패로 잽니다.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

from madang.versus import bm, network


@torch.no_grad()
def values(net, planes, device, batch=4096):
    out = []
    for i in range(0, len(planes), batch):
        _, v = net(torch.from_numpy(planes[i : i + batch]).to(device))
        out.append(v.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description="Value error of models on fresh self-play games of a player model.")
    ap.add_argument("game", help="game folder with index.js")
    ap.add_argument("player", help="model.pt that plays the test games")
    ap.add_argument("models", nargs="+", help="model.pt files to measure")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--parallel", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--temp-moves", type=int, default=None, help="default: board cells, up to 30 (as in training)")
    ap.add_argument("--max-moves", type=int, default=512)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.monotonic()
    pool = bm.Pool(args.game, args.workers)
    info = pool.info
    temp = args.temp_moves if args.temp_moves is not None else min(30, info["rows"] * info["cols"])
    games = []
    try:
        player = network(args.player, info, args.device)
        cmd = {
            "cmd": "selfplay",
            "games": args.games,
            "parallel": args.parallel,
            "sims": args.sims,
            "tempMoves": temp,
            "maxMoves": args.max_moves,
            "noise": {"eps": 0.25},
            "seed": args.seed * 1_000_000,
            "teacher": None,
            "method": "puct",
            "valueMix": 0,
            "fastSims": 0,
            "fullProb": 0.25,
        }
        stats = pool.split(cmd, bm.make_infer(player, args.device), on_game=lambda o, d: games.append(bm.unpack_game(info, o, d)))
    finally:
        pool.close()
    planes = np.concatenate([g[0] for g in games])
    zs = np.concatenate([g[2] for g in games])
    decided = zs != 0
    print(
        f"{info['id']}: {stats['games']} games by {args.player} ({args.sims} sims), first/second/draw "
        f"{stats['wins'][0]}/{stats['wins'][1]}/{stats['draws']}, {len(zs)} positions, "
        f"{time.monotonic() - t0:.0f}s"
    )
    print(f"  always 0: value mse {float((zs**2).mean()):.3f}")
    for path in args.models:
        v = values(network(path, info, args.device), planes, args.device)
        mse = float(((v - zs) ** 2).mean())
        sign = float((np.sign(v[decided]) == zs[decided]).mean()) if decided.any() else float("nan")
        print(f"  {path}: value mse {mse:.3f}  sign right {sign:.1%} (decided games)  mean |v| {float(np.abs(v).mean()):.2f}")


if __name__ == "__main__":
    sys.exit(main())
