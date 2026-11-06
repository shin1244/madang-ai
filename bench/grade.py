"""학습한 신경망(model.pt)을 완전 탐색한 정답으로 채점합니다. 탐색 없이 신경망만 봅니다.

    python bench/grade.py runs/c4-4x4.bin runs/a/model.pt runs/b/model.pt

  policy  둔 수에 따라 결과가 갈리는 국면에서, 신경망이 가장 높게 본 수가 가장 좋은 수 중 하나인 비율
  value   가치 예측과 정답(+1, 0, −1)의 평균 제곱 오차. "always 0"(늘 0이라고 답함)보다 낮아야 승패를 맞히는 것입니다
  class   가치 예측을 승(> 1/3), 무, 패(< −1/3)로 나눴을 때 정답과 같은 비율
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "boardmadang" / "trainer"))
from net import Net  # noqa: E402


def load(path):
    b = Path(path).read_bytes()
    (hl,) = struct.unpack("<I", b[:4])
    h = json.loads(b[4 : 4 + hl])
    at = 4 + hl

    def take(dtype, count):
        nonlocal at
        a = np.frombuffer(b, dtype, count, at)
        at += a.nbytes
        return a

    n, a = h["n"], h["A"]
    planes = take(np.float32, n * h["channels"] * h["rows"] * h["cols"]).reshape(n, h["channels"], h["rows"], h["cols"])
    legal = take(np.uint8, n * a).reshape(n, a).astype(bool)
    best = take(np.uint8, n * a).reshape(n, a).astype(bool)
    value = take(np.float32, n)
    choice = take(np.uint8, n).astype(bool)
    return planes, legal, best, value, choice


@torch.no_grad()
def grade(model, data):
    planes, legal, best, value, choice = data
    saved = torch.load(model, map_location="cpu")
    net = Net(**saved["config"]).eval()
    net.load_state_dict(saved["state"])
    logits, values = [], []
    for i in range(0, len(planes), 8192):
        lg, v = net(torch.from_numpy(planes[i : i + 8192].copy()))
        logits.append(lg.numpy())
        values.append(v.numpy())
    logits = np.where(legal, np.concatenate(logits), -np.inf)
    v = np.concatenate(values)
    top = logits.argmax(1)
    hit = best[np.arange(len(top)), top]
    cls = np.select([v > 1 / 3, v < -1 / 3], [1.0, -1.0], 0.0)
    return {
        "iteration": saved["iteration"],
        "policy": float(hit[choice].mean()),
        "value_mse": float(((v - value) ** 2).mean()),
        "value_class": float((cls == value).mean()),
    }


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    data = load(sys.argv[1])
    value = data[3]
    print(f"{len(value)} positions, always 0: value mse {(value**2).mean():.3f}, draws {(value == 0).mean():.1%}")
    for model in sys.argv[2:]:
        g = grade(model, data)
        print(
            f"{model}: iteration {g['iteration']}  policy {g['policy']:.1%}  "
            f"value mse {g['value_mse']:.3f}  class {g['value_class']:.1%}"
        )


if __name__ == "__main__":
    main()
