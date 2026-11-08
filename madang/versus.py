"""두 모델을 한 게임에서 서로 두게 합니다. 정답을 모르는 게임(바둑 5×5 등)에서 혼자 학습과 함께 학습을 비교할 때 씁니다.

    python -m madang.versus bench/games/go-5x5 runs/alone/model.pt runs/joint/model.pt --games 200 --sims 50

모델은 보드마당 학습기의 model.pt나 공통 모델의 model.pt입니다 (공통 모델은 이 게임의 경로로 둠).
보드마당 학습기의 --versus와 같은 방식입니다: 판 둘이 한 쌍으로, 같은 시작과 같은 무작위 첫 수(--opening-moves)로 차례만 바꿔 둡니다.
결과는 첫 모델(A) 기준 승·무·패, 점수와 95% 구간, Elo 차이입니다.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "boardmadang" / "trainer"))
import train as bm  # noqa: E402  boardmadang/trainer/train.py

from madang.net import GamePath  # noqa: E402
from madang.net import load as load_shared  # noqa: E402


def network(path, info, device):
    """model.pt → 이 게임을 두는 신경망 (forward(x))"""
    saved = torch.load(path, map_location=device)
    if "games" in saved["config"]:
        shared = load_shared(saved, device)
        g = shared.index(info["id"])
        names = shared.config["games"][g]["channelNames"]
        net = GamePath(shared, g).eval()
    else:
        net = bm.load_net(saved, device)
        names = (saved.get("info") or info)["channelNames"]
    if names != info["channelNames"]:
        raise SystemExit(f"{path} was trained on other observation channels than {info['id']}")
    return net


def main():
    ap = argparse.ArgumentParser(description="Play two models against each other in one game.")
    ap.add_argument("game", help="game folder")
    ap.add_argument("a", help="model A (results are A's)")
    ap.add_argument("b", help="model B")
    ap.add_argument("--games", type=int, default=200, help="games (pairs that swap sides, so it is made even)")
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--opening-moves", type=int, default=2, help="random first moves of each pair, so the games differ")
    ap.add_argument("--max-moves", type=int, default=512)
    ap.add_argument("--parallel", type=int, default=64)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pool = bm.Pool(args.game, args.workers)
    try:
        info = pool.info
        infers = [bm.make_infer(network(p, info, args.device), args.device) for p in (args.a, args.b)]
        cmd = {
            "cmd": "match",
            "opponent": "network",
            "games": max(2, args.games + args.games % 2),
            "parallel": args.parallel,
            "sims": args.sims,
            "maxMoves": args.max_moves,
            "seed": args.seed,
            "openingMoves": args.opening_moves,
        }
        res = pool.split(cmd, infers, unit=2)
    finally:
        pool.close()
    v = bm.versus_summary(res)
    cut = f", {res['cut']} cut by the training draw rule" if res["cut"] else ""
    print(
        f"{info['id']}: {args.a} vs {args.b}: {v['games']} games ({v['distinct']} distinct), {args.sims} sims, "
        f"first {args.opening_moves} moves random{cut}\n"
        f"  A as first W/D/L {bm.wdl(res['first'])}  second {bm.wdl(res['second'])}\n"
        f"  A scores {v['score']:.1%} ({v['low']:.1%}-{v['high']:.1%}), "
        f"Elo {v['elo']:+.0f} ({v['eloLow']:+.0f} to {v['eloHigh']:+.0f}, 95%)",
        flush=True,
    )
    return v


if __name__ == "__main__":
    main()
