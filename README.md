# madang-ai

[보드마당](https://boardmadang.com)에 올라온 여러 보드게임을 함께 배우는 AI를 연구하는 저장소입니다.

지금 보드마당의 AI는 게임마다 처음부터 따로 학습합니다 ([boardmadang/trainer](https://github.com/shin1244/boardmadang/tree/main/trainer)).
여기서는 사람이 장기를 두다 체스를 금방 배우듯, 여러 게임에서 배운 감각을 새 게임에 옮기는 방법을 찾습니다.

## 지키는 원칙

보드마당의 세 원칙을 그대로 따릅니다.

1. **제작자는 게임에 꼭 필요한 함수만 씁니다.** AI 때문에 새 함수를 요구하지 않습니다. AI에 필요한 것은 이미 있는 함수에서 뽑습니다.
2. **유저와 서버, 학습하는 컴퓨터 모두 보안이 뚫리지 않아야 합니다.** 올라온 게임 코드는 어디서 돌든 격리 안에서만 돌립니다.
3. **범용 모델 몇 개로 어떤 게임이든 기초 수준까지 학습합니다.**

## 어떻게 나누나

| 보드마당 ([boardmadang](https://github.com/shin1244/boardmadang), 이 저장소에 submodule로 고정) | 이 저장소 |
|---|---|
| 게임 API, 사이트, 서버 | 여러 게임을 함께 배우는 학습 |
| 관측과 행동을 텐서로 바꾸는 규칙 (`trainer/env.js`), 탐색 (`trainer/mcts.js`), 브라우저 AI | 게임별 학습 데이터 저장소 |
| 게임 하나를 학습하는 기본 학습기 (`trainer/train.py`) | 게임별 작은 모델로 줄이기 (증류) |
| 모델 파일 형식 (`model.onnx`, `model.json`) | 벤치마크와 실험 기록 |

두 저장소의 경계는 **모델 파일 형식**입니다. 여기서 어떻게 학습하든 결과물이 지금과 같은 `model.onnx`와 `model.json`이면 사이트는 바뀌지 않습니다.
관측 인코딩이 학습할 때와 브라우저에서 같아야 하므로, 보드마당은 복사하지 않고 submodule로 특정 커밋에 고정해서 씁니다.

## 계획

0. **재는 도구와 기준선**: 벤치마크 게임을 늘리고, 게임마다 처음부터 학습한 곡선을 시드 여러 개로 잽니다.
1. **이어 학습 (전이)**: 같은 게임의 새 버전이나 비슷한 게임의 모델에서 시작합니다. 처음부터 학습할 때와 같은 시간으로 비교합니다.
2. **여러 게임을 함께 학습**: 몸통은 공유하고, 게임마다 입력층(기물 채널이 게임마다 다름)과 작은 "게임 벡터"를 따로 둡니다.
   새 게임은 이 모델에서 짧게 이어 학습합니다. 비슷한 게임끼리 묶어 서로 방해하지 않게 합니다.
3. **올리자마자 쓰는 AI와 증류**: 게임을 돌려 얻은 규칙 지문으로 학습 없이 바로 두는 AI, 그리고 큰 모델을 게임별 작은 모델로 줄여 브라우저에서 빠르게 돌리기.
4. **실제 게임**: 본인이 만든 게임부터, 다른 제작자의 게임은 약관을 바꾼 뒤에.

단계마다의 구조, 재는 방법, "됐다"의 기준은 [docs/plan.md](docs/plan.md)에 있습니다. 참고한 연구는 [docs/references.md](docs/references.md)에 있습니다.

## 시작하기 전에 정할 것

- **약관**: 보드마당 약관은 올린 게임을 "서비스(… 앞으로 붙일 AI 상대)에 필요한 범위에서" 쓰도록 허락받습니다.
  한 게임으로 배운 것이 다른 게임의 AI에 쓰이는 공통 모델은 이 범위를 넘을 수 있으니, 사이트의 게임으로 학습하기 전에 약관과
  "AI 만들기" 동의 창에 이 용도를 밝힙니다. 그전까지는 벤치마크 게임과 보드마당 레포의 예제 게임만 씁니다.
- **게임을 내렸을 때**: 약관대로 내린 게임은 새 학습에 쓰지 않습니다. 그래서 데이터는 게임 id와 코드 버전별로 저장하고, 학습할 때 내린 게임을 뺍니다.
  이미 학습한 가중치에서 그 게임의 영향을 지우기는 어렵다는 점도 약관에 적어야 합니다.
- **격리**: 올라온 게임 코드는 네트워크가 없고 읽기 전용인 컨테이너 안에서만 돌립니다. 학습 결과 폴더(`model.pt`는 pickle)는 컨테이너에 연결하지 않습니다.
- **데이터 오염**: 한 게임의 데이터가 모두의 모델로 들어가므로 게임마다 데이터 상한을 두고, 새 게임을 넣기 전에 다른 게임의 벤치마크 점수가 떨어지지 않는지 봅니다.

## 벤치마크

`bench/games`의 게임은 모든 국면의 정답을 완전 탐색으로 구할 수 있을 만큼 작습니다. 학습한 신경망을 그 정답으로 채점합니다.

| 게임 | 판 | 정답 국면 | 완벽하게 두면 |
|---|---|---|---|
| `c3-4x4` | 4×4, 중력, 3목 | 65,757 | 선공 승 |
| `c4-4x4` | 4×4, 중력, 4목 | 304,575 | 무승부 |
| `c3-4x5` | 4행 × 5열, 중력, 3목 | 1,062,739 | 선공 승 |
| `m3-4x4` | 4×4, 중력, 3목을 만들면 짐 (`c3-4x4`의 미제르. 판과 관측이 같고 뜻만 반대) | 65,757 | 후공 승 |
| `othello-4x4` | 4×4 오델로 (둘 곳이 없으면 차례가 저절로 넘어감) | 79,791 | 후공 승 |
| `go-5x5` | 5×5 바둑 (Tromp-Taylor 규칙, 단순 패, 덤 7.5) | 너무 많음 | 흑 승 (알려진 결과). 모델끼리 대국으로 비교 |
| `shogi-3x4` | 3×4 판 위에서 기물을 옮기고, 잡은 기물을 다시 놓는 작은 장기 (십이장기와 같은 규칙. 위아래 한 줄씩은 손). 체스 기반 묶음용 | 너무 많음 | 모름. 모델끼리 대국으로 비교 |

```bash
git clone --recurse-submodules https://github.com/shin1244/madang-ai
cd madang-ai
pip install -r boardmadang/trainer/requirements.txt

# 학습 (보드마당의 기본 학습기)
python boardmadang/trainer/train.py bench/games/c4-4x4 --minutes 5 --solve-limit 0 --rollout-sims 0 --snapshot-every 0 --out runs/c4-base

# 채점 데이터 만들기 (정답 국면 6만 개) → 채점
node bench/dump.mjs bench/games/c4-4x4 runs/c4-4x4.bin 60000
python bench/grade.py runs/c4-4x4.bin runs/c4-base/model.pt
```

| 점수 | 뜻 |
|---|---|
| policy | 둔 수에 따라 결과가 갈리는 국면에서, 신경망이 가장 높게 본 수가 가장 좋은 수 중 하나인 비율 (탐색 없이) |
| value mse | 가치 예측과 정답(+1, 0, −1)의 평균 제곱 오차. 늘 0이라고 답할 때(`always 0`)보다 낮아야 승패를 맞히는 것입니다 |
| class | 가치 예측을 승, 무, 패로 나눴을 때 정답과 같은 비율 |

`bench/dump.mjs`는 게임 코드를 격리 없이 돌리므로 이 저장소의 벤치마크 게임에만 씁니다.

## 공통 모델 (여러 게임 함께 학습)

신경망 하나로 여러 게임을 함께 학습합니다 ([docs/plan.md](docs/plan.md)의 2단계). 게임마다 보드마당의 자기대국 워커를 띄우고,
몸통은 같이 쓰며 게임별 입력층·게임 벡터·특수 행동 머리만 따로 둡니다 ([madang/net.py](madang/net.py)).

```bash
# 3게임 함께 15분 (게임마다 5분꼴). GPU가 보이면 GPU로 계산합니다
python -m madang.train bench/games/c3-4x4 bench/games/c4-4x4 bench/games/c3-4x5 --minutes 15 --out runs/joint3

# 게임마다 채점. 공통 모델은 .bin에 적힌 게임의 경로로 채점합니다 (예전에 만든 .bin은 bench/dump.mjs로 다시 만듦)
python bench/grade.py runs/c4-4x4.bin runs/joint3/model.pt

# 정답이 없는 게임(바둑 5×5)은 두 모델을 서로 두게 해서 비교합니다 (공통 모델이든 게임 하나짜리든)
python -m madang.versus bench/games/go-5x5 runs/joint/model.pt runs/alone/model.pt --games 200

# 테스트 (신경망과 학습, 벤치마크 게임 규칙)
python -m unittest discover -s madang -t .
npm test
```

`madang/train.py`는 모듈로 실행합니다(`python -m`). 보드마당의 `train.py`, `net.py`와 이름이 같아서 스크립트로 실행하면 이 폴더의 것을 잘못 불러옵니다.
게임 하나만 주면 보드마당 학습기와 같은 설정·시드로 돌아서, 구현이 결과를 바꾸지 않았는지 비교할 수 있습니다.

지금까지 잰 것은 [docs/experiments.md](docs/experiments.md)에 있습니다.

## 라이선스

- **이 저장소의 코드**: [MIT](LICENSE). 상업적으로도 자유롭게 쓸 수 있고, 저작권 표시(원본이 이 저장소라는 것)만 남기면 됩니다.
- **공개하는 모델 파일**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ko). 상업적으로 쓸 수 있고, 출처만 밝히면 됩니다.
- **submodule `boardmadang/`**: 그 저장소의 라이선스를 따릅니다. 이 저장소가 쓰는 `sdk/`, `games/`, `trainer/`는 MIT이고,
  쓰지 않는 `server/`와 `web/`만 AGPL-3.0입니다 ([LICENSING.md](https://github.com/shin1244/boardmadang/blob/main/LICENSING.md)).
  그래서 이 저장소는 통째로 상업적으로 쓸 수 있습니다.
