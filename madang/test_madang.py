"""공통 신경망 모양, 게임별 정규화, 저장과 불러오기, 함께 학습 → 채점까지.

    python -m unittest discover -s madang -t .
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from madang import train as mt
from madang.net import COMMON, GamePath, SharedNet, init_from, load, split_channels

ROOT = Path(__file__).resolve().parent.parent
MARKS = [n for n in COMMON if n.startswith("mark:")]
# 자식 프로세스의 출력은 UTF-8로 주고받습니다 (Windows 기본 코드 페이지로는 한글 경로가 깨짐)
UTF8 = {"env": {**os.environ, "PYTHONUTF8": "1"}, "encoding": "utf-8", "capture_output": True, "cwd": ROOT}


def game(gid, rows, cols, pieces, observe=0, specials=()):
    """env.js의 channelNames와 같은 순서로 만든 게임 정보"""
    names = [f"piece:{p}" for p in pieces] + MARKS + ["tint:0", "tint:1", "tint:other", "text", "toPlay"]
    names += [f"observe:{i}" for i in range(observe)]
    return {
        "id": gid,
        "rows": rows,
        "cols": cols,
        "channelNames": names,
        "channels": len(names),
        "specials": list(specials),
        "nActions": rows * cols + len(specials),
    }


A = game("a", 4, 4, ["red", "blue"])
B = game("b", 5, 6, ["wK", "bK", "wR"], observe=2, specials=["pass"])


class NetTest(unittest.TestCase):
    def test_split_channels(self):
        common, own = split_channels(B["channelNames"])
        self.assertEqual([B["channelNames"][i] for i in common], COMMON)
        self.assertEqual([B["channelNames"][i] for i in own], ["piece:wK", "piece:bK", "piece:wR", "observe:0", "observe:1"])
        with self.assertRaises(ValueError):
            split_channels(["piece:x", "toPlay"])

    def test_shapes_per_game(self):
        net = SharedNet([A, B], blocks=2, filters=8, dim=4).eval()
        logits, v = net(torch.zeros(3, A["channels"], 4, 4), 0)
        self.assertEqual((tuple(logits.shape), tuple(v.shape)), ((3, 16), (3,)))
        # 특수 행동은 칸 클릭 뒤에 붙습니다
        logits, v = net(torch.zeros(2, B["channels"], 5, 6), 1)
        self.assertEqual((tuple(logits.shape), tuple(v.shape)), ((2, 31), (2,)))
        self.assertEqual(net.index("b"), 1)
        with self.assertRaises(KeyError):
            net.index("c")
        with self.assertRaises(ValueError):
            SharedNet([A, A])

    def test_norm_statistics_are_per_game(self):
        # 한 게임으로 학습해도 다른 게임의 평균·분산은 그대로입니다 (평가 때 마지막에 본 게임 쪽으로 쏠리지 않게)
        net = SharedNet([A, B], blocks=1, filters=8, dim=4).train()
        net(torch.rand(16, A["channels"], 4, 4), 0)
        for norm in (net.stem, net.trunk[0].n2, net.value_norm):
            self.assertGreater(norm.mean[0].abs().sum().item(), 0)
            self.assertNotEqual(norm.var[0].tolist(), [1.0] * norm.filters)
            self.assertEqual(norm.mean[1].abs().sum().item(), 0)
            self.assertEqual(norm.var[1].tolist(), [1.0] * norm.filters)

    def test_save_and_load(self):
        torch.manual_seed(0)
        net = SharedNet([A, B], blocks=2, filters=8, dim=4)
        net.train()(torch.rand(8, B["channels"], 5, 6), 1)  # 평균·분산도 저장되는지 보려고
        saved = {"config": net.config, "state": net.state_dict()}
        again = load(saved)
        x = torch.rand(4, B["channels"], 5, 6)
        with torch.no_grad():
            want = net.eval()(x, 1)
            got = again(x, 1)
            path = GamePath(again, 1)(x)
        for a, b, c in zip(want, got, path):
            self.assertTrue(torch.allclose(a, b))
            self.assertTrue(torch.allclose(a, c))


class InitFromTest(unittest.TestCase):
    def test_keeps_shared_games_and_starts_new_ones_from_the_average(self):
        torch.manual_seed(0)
        old = SharedNet([A, B], blocks=2, filters=8, dim=4)
        old.train()(torch.rand(8, A["channels"], 4, 4), 0)
        old(torch.rand(8, B["channels"], 5, 6), 1)
        saved = {"config": old.config, "state": old.state_dict()}
        C = game("c", 6, 6, ["x", "o"], observe=1, specials=["pass"])
        new = SharedNet([B, C], blocks=2, filters=8, dim=4)  # A는 빠지고 C가 새로 들어옴
        got = init_from(new, saved)
        self.assertEqual((got["kept"], got["fresh"], got["dropped"]), (["b"], ["c"], ["a"]))
        x = torch.rand(4, B["channels"], 5, 6)
        with torch.no_grad():
            want = old.eval()(x, 1)
            have = new.eval()(x, 0)
        for a, b in zip(want, have):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        # 새 게임은 옮긴 게임들의 평균에서 (여기서는 B 하나)
        self.assertTrue(torch.equal(new.embed.weight[1], old.embed.weight[1]))
        self.assertTrue(torch.equal(new.stem.mean[1], old.stem.mean[1]))
        logits, v = new(torch.rand(2, C["channels"], 6, 6), 1)
        self.assertEqual((tuple(logits.shape), tuple(v.shape)), ((2, 37), (2,)))
        with self.assertRaises(ValueError):
            init_from(SharedNet([B], blocks=3, filters=8, dim=4), saved)  # 크기가 다름
        with self.assertRaises(ValueError):
            init_from(SharedNet([C], blocks=2, filters=8, dim=4), saved)  # 같은 게임이 없음


class TrainTest(unittest.TestCase):
    def test_train_and_grade(self):
        """벤치마크 게임 둘을 함께 아주 조금 학습 → 한 게임을 채점. 모델에 없는 게임은 채점하지 않습니다"""
        small = ["--blocks", "1", "--filters", "8", "--games", "4", "--sims", "4", "--steps", "2", "--workers", "1"]
        with tempfile.TemporaryDirectory() as d:
            games = [str(ROOT / "bench" / "games" / g) for g in ("c3-4x4", "c4-4x4")]
            run = subprocess.run(
                [sys.executable, "-m", "madang.train", *games, "--iterations", "2", "--eval-games", "2", "--out", d, *small],
                **UTF8,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("iter   2  c4-4x4", run.stdout)
            saved = torch.load(Path(d) / "model.pt")
            self.assertEqual([g["id"] for g in saved["config"]["games"]], ["c3-4x4", "c4-4x4"])
            self.assertEqual(saved["iteration"], 2)

            data = Path(d) / "c3.bin"
            dump = subprocess.run(
                ["node", "bench/dump.mjs", "bench/games/c3-4x4", str(data), "500"], **UTF8
            )
            self.assertEqual(dump.returncode, 0, dump.stderr)
            grade = subprocess.run(
                [sys.executable, "bench/grade.py", str(data), str(Path(d) / "model.pt")], **UTF8
            )
            self.assertEqual(grade.returncode, 0, grade.stderr)
            self.assertIn("c3-4x4: 500 positions", grade.stdout)
            self.assertIn("iteration 2  policy", grade.stdout)

            sys.path.insert(0, str(ROOT / "bench"))
            import grade as gr

            with self.assertRaises(SystemExit) as e:
                gr.network(Path(d) / "model.pt", {"game": "c3-4x5"})
            self.assertIn("not one of this model's games", str(e.exception))

            # 모델끼리 대국 (정답이 없는 게임의 비교)
            vs = subprocess.run(
                [sys.executable, "-m", "madang.versus", "bench/games/c4-4x4", str(Path(d) / "model.pt"),
                 str(Path(d) / "model.pt"), "--games", "2", "--sims", "2", "--workers", "1"],
                **UTF8,
            )
            self.assertEqual(vs.returncode, 0, vs.stderr)
            self.assertIn("c4-4x4: ", vs.stdout)
            self.assertIn("2 games", vs.stdout)


if __name__ == "__main__":
    unittest.main()
