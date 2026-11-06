# 실험 기록

날짜순으로 쌓습니다. 실험마다 무엇을, 어떤 명령으로, 어디서 쟀는지 남겨서 다시 돌릴 수 있게 합니다.

## 2026-11-06 탐색 횟수 섞기 (playout cap randomization)

보드마당 [PR #32](https://github.com/shin1244/boardmadang/pull/32)의 `--fast-sims`. KataGo처럼 수의 25%만 깊게 탐색해서 학습 데이터로 남기고,
나머지는 얕게 탐색해 판을 진행하면 같은 시간에 더 나은 신경망이 나오는지 봤습니다.

- **게임**: `bench/games/c4-4x4` (중력 4목 4×4, 304,575국면, 완벽하게 두면 무승부)
- **예산**: 5분씩, 시드 1과 2. CPU 4코어 (클라우드 컨테이너, GPU 없음), 신경망은 기본 크기(4블록 × 64채널), 워커 4개
- **비교**: 반복마다 새 국면 수가 비슷하도록 섞는 쪽은 `--games`를 4배(400)로
- **채점**: `bench/grade.py`, 정답 국면 6만 개 (`bench/dump.mjs ... 60000`). 탐색 없이 신경망만

```bash
common="--minutes 5 --snapshot-every 0 --rollout-sims 0 --solve-limit 0 --eval-every 1000"
python boardmadang/trainer/train.py bench/games/c4-4x4 $common --seed 1 --out runs/puct-base-1
python boardmadang/trainer/train.py bench/games/c4-4x4 $common --seed 1 --fast-sims 10 --games 400 --out runs/puct-pcr-1
python boardmadang/trainer/train.py bench/games/c4-4x4 $common --seed 1 --search gumbel --sims 16 --out runs/gumbel-base-1
python boardmadang/trainer/train.py bench/games/c4-4x4 $common --seed 1 --search gumbel --sims 16 --fast-sims 4 --games 400 --out runs/gumbel-pcr-1
# 시드 2도 같게
python bench/grade.py runs/c4-4x4.bin runs/*/model.pt
```

`--fast-sims`는 PR #32로 보드마당에 들어갔고, submodule이 그 커밋(`d88b79d`) 이후를 가리킵니다.

| 설정 | 시드 | 반복 | 판 | 남긴 국면 | 최선의 수 | 가치 오차 | 승·무·패 맞힘 |
|---|---|---|---|---|---|---|---|
| PUCT 50번 | 1 | 56 | 5,600 | 87,644 | 92.8% | 0.239 | 77.1% |
| PUCT 50번 | 2 | 55 | 5,500 | 86,061 | 92.1% | 0.190 | 80.2% |
| PUCT 50번, 섞기 10번 | 1 | 44 | 17,600 | 68,735 | 90.9% | 0.205 | 80.2% |
| PUCT 50번, 섞기 10번 | 2 | 42 | 16,800 | 65,310 | 91.4% | 0.185 | 80.6% |
| Gumbel 16번 | 1 | 80 | 8,000 | 126,733 | 84.7% | 0.265 | 72.1% |
| Gumbel 16번 | 2 | 82 | 8,200 | 129,814 | 85.6% | 0.257 | 74.3% |
| Gumbel 16번, 섞기 4번 | 1 | 62 | 24,800 | 97,148 | 85.4% | 0.275 | 73.6% |
| Gumbel 16번, 섞기 4번 | 2 | 69 | 27,600 | 107,757 | 88.1% | 0.236 | 76.0% |

가치를 늘 0이라고 답하면 오차가 0.359입니다 (정답 국면의 64.1%가 무승부). 마지막 평가에서 아무 수나 두는 상대에게는 8개 모두 한 판도 지지 않았습니다.

**결론**

- 섞으면 판 수는 3배쯤, 남긴 국면은 0.8배쯤이 됩니다.
- PUCT: 가치가 조금 나아지고(오차 평균 0.215 → 0.195), 정책은 조금 나빠집니다(92.5% → 91.2%). KataGo가 말한 방향과 같습니다.
- Gumbel: 셋 다 평균이 조금 나아지지만, 시드 1의 가치 오차는 오히려 나빠졌습니다.
- 같은 설정의 시드끼리 차이(PUCT 가치 오차 0.190~0.239)가 섞어서 생긴 차이보다 커서, **이 실험으로는 효과가 있다고 말할 수 없습니다.** 보드마당의 기본값은 꺼 둡니다.
- 따로 보인 것: 같은 5분이면 PUCT 50번이 Gumbel 16번보다 정책도 가치도 낫습니다. Gumbel은 판과 국면을 더 많이 만들고도 뒤집니다.

**다음에 잴 것**

- 한 판이 16수쯤이라 한 판의 국면끼리 닮아서 생기는 문제가 작은 게임이었습니다. KataGo는 한 판이 수백 수인 바둑에서 효과를 봤습니다.
  한 판이 길고 가치 학습이 약했던 Splat Chess(8×8, 한 판 20~30수)에서 GPU로 60분씩 다시 잽니다.
- 시드 간 차이가 크므로, 시드를 늘리거나 학습 시간을 늘려 차이가 굳는지 봅니다.
