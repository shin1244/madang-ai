"""model.pt → 사이트 AI 파일 (model.onnx, model.json). 운영자 화면의 "AI 모델"에서 두 파일을 같이 올립니다.

    python -m madang.export ../cellmoku runs/cellmoku-vmix/model.pt exports/cellmoku
    python -m madang.export ../splat-chess runs/chess2-vmix-s6/model.pt exports/splat-chess
    python -m madang.export ../twelve-janggi runs/shogi-vmix-s6/model.pt exports/twelve-janggi --zero tint:0,tint:1

보드마당 학습기의 model.pt와 공통 모델(madang.train)의 model.pt를 모두 받습니다. 공통 모델이면 그 게임의 경로만 내보냅니다.
보드마당 학습기의 --export-onnx와 같은 모양(입력 x, 출력 logits, value)과 같은 model.json을 씁니다.

학습한 게임과 내보낼 게임의 관측 채널이 다르면 이름으로 맞춥니다 (벤치 shogi-3x4 → 사이트 twelve-janggi):
- 판 크기, 특수 행동, 행동 수는 같아야 합니다
- 모델에만 있는 채널은 0을 넣습니다. 학습한 게임에서도 늘 비어 있던 채널이어야 같은 입력이 됩니다
- 내보낼 게임에만 있는 채널은 버립니다
- --zero의 채널은 이름이 같아도 버립니다. 학습한 게임에서는 늘 비어 있었는데 내보낼 게임에서 그림이 더해진 채널입니다
  (twelve-janggi의 진영 칠하기 tint:0, tint:1)
끝나면 내보낸 ONNX가 PyTorch와 같은 값을 내는지 보고, --check면 내보낼 게임의 코드로 무작위 상대·롤아웃 AI와 둬 봅니다.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from madang.net import GamePath
from madang.net import load as load_shared
from madang.versus import bm


class Remap(nn.Module):
    """내보낼 게임의 관측 → 모델이 학습한 관측 (채널을 이름으로 맞추고 없는 채널은 0)"""

    def __init__(self, net, src):
        super().__init__()
        self.net = net
        self.src = src  # 모델의 채널마다 내보낼 게임의 채널 번호, 없으면 -1

    def forward(self, x):
        zero = torch.zeros_like(x[:, :1])
        return self.net(torch.cat([x[:, i : i + 1] if i >= 0 else zero for i in self.src], 1))


def balance(path, game_id, upto):
    """공통 모델의 log.jsonl에서 이 게임의 학습 마지막 5분의 1 자기대국 합계 (보드마당 학습기의 balance_from_log와 같은 칸)"""
    rows = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        sp = (e.get("games") or {}).get(game_id, {}).get("selfplay")
        it = e.get("iteration")
        if isinstance(it, int) and sp and (upto is None or it <= upto):
            rows[it] = sp
    if not rows:
        return None
    its = sorted(rows)[-max(1, round(len(rows) * bm.BALANCE_SHARE)) :]
    total = {"games": 0, "wins": [0, 0], "draws": 0, "cut": 0}
    for i in its:
        sp = rows[i]
        total["games"] += sp["games"]
        total["wins"] = [total["wins"][0] + sp["wins"][0], total["wins"][1] + sp["wins"][1]]
        total["draws"] += sp["draws"]
        total["cut"] += sp.get("cut", 0)
    return {**total, "from": its[0], "to": its[-1]} if total["games"] else None


def main():
    ap = argparse.ArgumentParser(description="Export a model.pt (boardmadang trainer or shared model) as the site's AI files.")
    ap.add_argument("game", help="the site game's folder with index.js")
    ap.add_argument("model", help="model.pt")
    ap.add_argument("out", help="output folder for model.onnx and model.json")
    ap.add_argument("--trained-as", default=None, help="the game id inside a shared model (default: the game's id)")
    ap.add_argument("--zero", default="", help="comma-separated channels of the game to feed as zero (see above)")
    ap.add_argument("--sims", type=int, default=100, help="search per move of the browser AI (model.json sims, default 100)")
    ap.add_argument("--check", action="store_true", help="also play the random opponent and the rollout AI in the game")
    ap.add_argument("--check-games", type=int, default=40)
    ap.add_argument("--check-sims", type=int, default=50, help="search per move of the AI while checking (default 50)")
    args = ap.parse_args()

    worker = bm.Worker(args.game)
    worker.close()
    info = worker.info
    saved = torch.load(args.model, map_location="cpu")
    model_dir = Path(args.model).parent
    if "games" in saved["config"]:
        shared = load_shared(saved)
        g = shared.index(args.trained_as or info["id"])
        trained = dict(shared.config["games"][g])
        net = GamePath(shared, g).eval()
        bal = balance(model_dir / "log.jsonl", trained["id"], saved.get("iteration"))
        version = lang = None  # 공통 모델은 학습한 코드의 버전과 언어를 남기지 않습니다
    else:
        trained = saved.get("info") or info
        net = bm.load_net(saved)
        bal = bm.balance_from_log(model_dir / "log.jsonl", saved.get("iteration"))
        version, lang = trained.get("version"), trained.get("lang")
    for k in ("rows", "cols", "specials", "nActions"):
        if trained[k] != info[k]:
            raise SystemExit(f"{k} differ: the model has {trained[k]}, {info['id']} has {info[k]}")

    zero = {c for c in args.zero.split(",") if c}
    unknown = zero - set(info["channelNames"])
    if unknown:
        raise SystemExit(f"--zero names channels {info['id']} does not have: {', '.join(sorted(unknown))}")
    names = info["channelNames"]
    src = [names.index(c) if c in names and c not in zero else -1 for c in trained["channelNames"]]
    if trained["channelNames"] != names or zero:
        filled = [c for c, i in zip(trained["channelNames"], src) if i < 0]
        dropped = [c for i, c in enumerate(names) if i not in src]
        print(f"channels matched by name: model-only or zeroed {filled or '-'}; dropped from {info['id']} {dropped or '-'}")
        net = Remap(net, src).eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(2, info["channels"], info["rows"], info["cols"])
    torch.onnx.export(
        net,
        (x,),
        str(out / "model.onnx"),
        dynamo=True,
        verbose=False,
        input_names=["x"],
        output_names=["logits", "value"],
        dynamic_shapes={"x": {0: torch.export.Dim("n")}},
        external_data=False,
    )
    # 같은 trained version일 때만 버전을 남깁니다. 다른 게임 코드에서 학습했으면 null (알 수 없음)
    if trained.get("id") != info["id"]:
        version = None
    meta = {
        "format": bm.MODEL_FORMAT,
        "game": info["id"],
        "version": version,
        "lang": lang or info["lang"],
        **{k: info[k] for k in bm.SHAPE},
        "channels": info["channels"],
        "sims": args.sims,
        "iteration": saved.get("iteration"),
        "net": {"blocks": saved["config"]["blocks"], "filters": saved["config"]["filters"]},
    }
    if bal:
        meta["balance"] = bal
    (out / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    import onnxruntime as ort

    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    probe = (rng.random((16, info["channels"], info["rows"], info["cols"])) < 0.3).astype(np.float32)
    with torch.no_grad():
        want = [t.numpy() for t in net(torch.from_numpy(probe))]
    got = sess.run(None, {"x": probe})
    diff = max(float(np.abs(a - b).max()) for a, b in zip(want, got))
    print(f"{out / 'model.onnx'}: {(out / 'model.onnx').stat().st_size / 1e6:.1f} MB, onnx vs torch max diff {diff:.1e}")
    if diff > 1e-3:
        raise SystemExit("the exported model gives different values")
    print(json.dumps({k: meta[k] for k in ("game", "version", "lang", "channels", "sims", "iteration", "net")}, ensure_ascii=False))

    if args.check:
        # 사이트 게임 코드 그대로, 내보낸 ONNX로 둡니다
        def infer(xb):
            logits, v = sess.run(None, {"x": xb})
            return logits, v

        pool = bm.Pool(args.game, 8)
        try:
            opts = SimpleNamespace(
                rollout_sims=200, eval_games=args.check_games, sims=args.check_sims, max_moves=512
            )
            bm.evaluate(pool, infer, opts, solvable=False, rollout=True)
        finally:
            pool.close()


if __name__ == "__main__":
    sys.exit(main())
