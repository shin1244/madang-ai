"""여러 게임을 함께 학습합니다 (docs/plan.md의 2단계).

    python -m madang.train bench/games/c3-4x4 bench/games/c4-4x4 bench/games/c3-4x5 --minutes 15 --out runs/shared

게임마다 보드마당의 자기대국 워커 묶음(Pool)을 띄우고, 신경망 하나(madang/net.py)를 같이 씁니다.
반복마다 게임을 차례로 자기대국한 뒤(게임마다 --games판), 학습 한 걸음에 게임마다 같은 크기의 묶음을 뽑아
손실을 평균해 갱신합니다. 데이터는 게임마다 따로 쌓아서 한 게임이 모델을 차지하지 않게 합니다.
관측 인코딩, 탐색, 워커와 주고받는 메시지는 보드마당 학습기(boardmadang/trainer/train.py)의 것을 그대로 씁니다.
게임 하나만 주면 보드마당 학습기와 같은 자기대국 시드와 설정으로 돕니다 (구현이 결과를 바꾸지 않았는지 비교할 때).

모듈로 실행합니다 (python -m). 보드마당의 train.py, net.py와 이름이 같아서, 스크립트로 실행하면 이 폴더의 것을 잘못 불러옵니다.

결과 (--out)
    model.pt   공통 신경망과 설정(게임 목록). bench/grade.py로 게임마다 채점합니다
    log.jsonl  반복마다 게임별 자기대국 통계, 손실(학습 중, 새 판)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "boardmadang" / "trainer"))
import train as bm  # noqa: E402  boardmadang/trainer/train.py

from madang.net import SharedNet, init_from  # noqa: E402


def make_infer(net, g, device):
    """g번째 게임의 신경망 계산 (보드마당 Pool이 부르는 모양)"""

    @torch.no_grad()
    def infer(x):
        net.eval()
        logits, v = net(torch.from_numpy(x).to(device), g)
        return logits.float().cpu().numpy(), v.float().cpu().numpy()

    return infer


def joined(buffer):
    """반복마다 쌓은 (국면, 탐색 분포, 가치 정답)을 하나로"""
    return tuple(np.concatenate([b[k] for b in buffer]) for k in range(3))


def train(net, opt, buffers, steps, batch, device, rng):
    """학습 한 걸음마다 게임마다 batch개씩 뽑아 손실을 평균합니다. 게임마다의 평균 손실을 돌려줍니다"""
    data = [joined(b) for b in buffers]
    net.train()
    total = np.zeros((len(data), 2))
    for _ in range(steps):
        loss = 0
        for g, (planes, pis, zs) in enumerate(data):
            idx = rng.integers(0, len(zs), size=min(batch, len(zs)))
            logits, v = net(torch.from_numpy(planes[idx]).to(device), g)
            lp = -(torch.from_numpy(pis[idx]).to(device) * F.log_softmax(logits, 1)).sum(1).mean()
            lv = F.mse_loss(v, torch.from_numpy(zs[idx]).to(device))
            loss = loss + lp + lv
            total[g] += [lp.item(), lv.item()]
        opt.zero_grad()
        (loss / len(data)).backward()
        opt.step()
    return [{"policy": p / steps, "value": v / steps} for p, v in total]


@torch.no_grad()
def fresh_loss(net, g, data, device, batch=4096):
    """아직 학습에 안 쓴 새 판에서 잰 손실 (보드마당 train.py의 fresh_loss와 같음)"""
    planes, pis, zs = data
    if len(zs) == 0:
        return None
    net.eval()
    lp = lv = 0.0
    for i in range(0, len(zs), batch):
        logits, v = net(torch.from_numpy(planes[i : i + batch]).to(device), g)
        pi = torch.from_numpy(pis[i : i + batch]).to(device)
        z = torch.from_numpy(zs[i : i + batch]).to(device)
        lp += -(pi * F.log_softmax(logits.float(), 1)).sum().item()
        lv += ((v.float() - z) ** 2).sum().item()
    return {"policy": lp / len(zs), "value": lv / len(zs), "zero": float((zs**2).mean())}


def main():
    ap = argparse.ArgumentParser(description="Train one network on several boardmadang games together.")
    ap.add_argument("games", nargs="+", help="game folders with index.js")
    ap.add_argument("--minutes", type=float, default=15, help="training budget for all games together (default 15)")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--games", dest="per_game", type=int, default=100, help="self-play games per game per iteration")
    ap.add_argument("--sims", type=int, default=50, help="MCTS simulations per move")
    ap.add_argument("--parallel", type=int, default=64, help="games played at once in each worker")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="self-play workers per game")
    ap.add_argument("--temp-moves", type=int, default=None, help="default: board cells, up to 30 (per game)")
    ap.add_argument("--max-moves", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=None, help="residual blocks (default 4, or the --init-from model's)")
    ap.add_argument("--filters", type=int, default=None, help="channels per layer (default 64, or the --init-from model's)")
    ap.add_argument("--dim", type=int, default=None, help="size of each game's vector (default 16, or the --init-from model's)")
    ap.add_argument(
        "--init-from",
        default=None,
        help="start from another shared model.pt: games in both keep their parts, new games start from the average game, "
        "games not given here are dropped",
    )
    ap.add_argument("--steps", type=int, default=100, help="training steps per iteration")
    ap.add_argument("--batch", type=int, default=256, help="positions per game in each training step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=20, help="train on the last N iterations of each game's self-play")
    ap.add_argument("--search", choices=["puct", "gumbel"], default="puct")
    ap.add_argument("--value-mix", type=float, default=0)
    ap.add_argument("--eval-every", type=int, default=0, help="play the random opponent every N iterations (0 = only at the end)")
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--rollout-sims", type=int, default=0, help="the final evaluation also plays an untrained rollout AI")
    ap.add_argument("--solve-limit", type=int, default=0, help="also play a perfect opponent in games with fewer positions")
    ap.add_argument("--out", default="runs/shared")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.monotonic()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pools = []
    try:
        for d in args.games:
            pools.append(bm.Pool(d, args.workers))
        infos = [p.info for p in pools]
        init = torch.load(args.init_from, map_location="cpu") if args.init_from else None
        size = init["config"] if init else {}
        net = SharedNet(
            infos,
            args.blocks or size.get("blocks", 4),
            args.filters or size.get("filters", 64),
            args.dim or size.get("dim", 16),
        )
        if init:
            started = init_from(net, init)
            print(
                f"started from {args.init_from} (iteration {init.get('iteration')}): {started['layers']} shared layers copied, "
                f"kept {', '.join(started['kept'])}"
                + (f"; new {', '.join(started['fresh'])}" if started["fresh"] else "")
                + (f"; dropped {', '.join(started['dropped'])}" if started["dropped"] else ""),
                flush=True,
            )
        net = net.to(args.device)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        infers = [make_infer(net, g, args.device) for g in range(len(pools))]
        solvable = []
        for p, info in zip(pools, infos):
            solved = p.one({"cmd": "solve", "limit": args.solve_limit}) if args.solve_limit > 0 else {"tooBig": True}
            solvable.append(not solved.get("tooBig"))
            temp = args.temp_moves if args.temp_moves is not None else min(30, info["rows"] * info["cols"])
            info["tempMoves"] = temp
            print(
                f"{info['id']}: {info['rows']}x{info['cols']} board, {info['nActions']} actions, "
                f"{info['channels']} input channels"
                + (f", solved exactly ({solved['positions']} positions)" if solvable[-1] else ""),
                flush=True,
            )
        params = sum(p.numel() for p in net.parameters())
        print(
            f"{len(pools)} games, {net.config['blocks']}x{net.config['filters']} network ({params:,} parameters), device {args.device}, "
            f"{args.workers} workers per game",
            flush=True,
        )

        buffers = [[] for _ in pools]
        log = open(out / "log.jsonl", "a", encoding="utf-8")
        if init:
            log.write(json.dumps({"initFrom": str(args.init_from), "init": started}) + "\n")
        last = 0
        try:
            for it in range(1, args.iterations + 1):
                if time.monotonic() - t0 > args.minutes * 60:
                    break
                entry = {"iteration": it, "games": {}}
                fresh = []
                for g, (pool, info) in enumerate(zip(pools, infos)):
                    games = []
                    cmd = {
                        "cmd": "selfplay",
                        "games": args.per_game,
                        "parallel": args.parallel,
                        "sims": args.sims,
                        "tempMoves": info["tempMoves"],
                        "maxMoves": args.max_moves,
                        "noise": {"eps": 0.25},
                        "seed": args.seed * 1_000_000 + it * args.per_game,
                        "method": args.search,
                        "valueMix": args.value_mix,
                    }
                    stats = pool.split(cmd, infers[g], on_game=lambda obj, data, i=info: games.append(bm.unpack_game(i, obj, data)))
                    new = joined(games)
                    fresh.append(fresh_loss(net, g, new, args.device))  # 버퍼에 넣기 전에 잽니다
                    buffers[g] = (buffers[g] + [new])[-args.window :]
                    entry["games"][info["id"]] = {"selfplay": stats, "fresh": fresh[-1]}
                losses = train(net, opt, buffers, args.steps, args.batch, args.device, rng)
                for g, info in enumerate(infos):
                    e = entry["games"][info["id"]]
                    e["loss"] = losses[g]
                    e["buffer"] = int(sum(len(b[2]) for b in buffers[g]))
                    s, f = e["selfplay"], fresh[g]
                    print(
                        f"iter {it:3d}  {info['id']:<10} first/second/draw {s['wins'][0]}/{s['wins'][1]}/{s['draws']}"
                        + (f" (cut {s['cut']})" if s["cut"] else "")
                        + f"  moves {s['moves'] / args.per_game:.1f}"
                        + f"  loss p {losses[g]['policy']:.3f} v {losses[g]['value']:.3f}"
                        + (f"  new games p {f['policy']:.3f} v {f['value']:.3f} (always 0: {f['zero']:.3f})" if f else "")
                        + f"  buffer {e['buffer']}",
                        flush=True,
                    )
                entry["seconds"] = time.monotonic() - t0
                print(f"          {entry['seconds']:.0f}s", flush=True)
                if args.eval_every and it % args.eval_every == 0:
                    entry["eval"] = evaluate_all(pools, infos, infers, args, solvable)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                torch.save(
                    {"config": net.config, "state": net.state_dict(), "iteration": it, "minutes": entry["seconds"] / 60},
                    out / "model.pt",
                )
                last = it
            if last:
                print("final evaluation", flush=True)
                res = evaluate_all(pools, infos, infers, args, solvable, rollout=True)
                log.write(json.dumps({"iteration": last, "eval": res}) + "\n")
        finally:
            log.close()
    finally:
        for p in pools:
            p.close()
    print(f"done in {time.monotonic() - t0:.0f}s: {out / 'model.pt'}")


def evaluate_all(pools, infos, infers, args, solvable, rollout=False):
    """게임마다 보드마당 학습기의 평가 (아무 수나 두는 상대, 완전 탐색이 되면 완벽한 상대)"""
    res = {}
    for pool, info, infer, ok in zip(pools, infos, infers, solvable):
        print(f"{info['id']}:", flush=True)
        res[info["id"]] = bm.evaluate(pool, infer, args, ok, rollout=rollout)
    return res


if __name__ == "__main__":
    main()
