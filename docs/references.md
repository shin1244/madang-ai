# 참고한 연구

## 여러 게임을 함께 배우기

- [Transfer of Fully Convolutional Policy-Value Networks Between Games and Game Variants](https://arxiv.org/abs/2102.12375) (Soemers 외, 2021),
  후속 [Towards a General Transfer Approach for Policy-Value Networks](https://research.dial.uclouvain.be/handle/2078.5/254050) (TMLR 2023).
  Ludii의 게임과 변형 사이에서 완전 합성곱 신경망을 옮깁니다. 채널의 공통 의미로 입력을 맞추고, 바로 옮겨도 잘 되는 경우가 많았습니다.
  보드마당의 표시(`marks`), 칠한 칸(`tint`), 두는 쪽 채널이 이런 공통 의미가 됩니다.
- [Deep Learning for General Game Playing with Ludii and Polygames](https://arxiv.org/abs/2101.09562) (2021). 게임 설명에서 입력과 출력 텐서를 자동으로 만들어, 게임마다 신경망을 따로 설계하지 않습니다.
- [General Board Game Concepts](https://arxiv.org/abs/2107.01078) (2021). 게임의 특징(개념)으로 게임 사이의 전이를 설명합니다. "게임 벡터"를 만들 때 참고합니다.
- [Multi-Game Decision Transformers](https://arxiv.org/abs/2205.15241) (Google, NeurIPS 2022). 가중치 하나로 Atari 41~46개 게임을 사람 가까이 두고, 새 게임에 미세조정으로 빨리 적응합니다.
- [Deep Reinforcement Learning for General Game Playing](https://ojs.aaai.org/index.php/AAAI/article/view/5533) (Goldwaser & Thielscher, AAAI 2020). 형식 언어로 쓴 규칙을 받아 AlphaZero로 학습합니다 (게임마다 처음부터).

## 학습 효율

- [Accelerating Self-Play Learning in Go](https://arxiv.org/abs/1902.10565) (KataGo, Wu 2019). 탐색 횟수 섞기(playout cap randomization), 칸별 보조 정답(ownership), 몸통의 전역 풀링.
  논문 이후의 기법은 KataGo 레포의 [KataGoMethods.md](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md).
- [Policy improvement by planning with Gumbel](https://iclr.cc/virtual/2022/poster/6418) (Danihelka 외, ICLR 2022). 탐색이 적어도 학습되는 Gumbel AlphaZero. 공식 구현 [mctx](https://github.com/deepmind/mctx).
- [Polygames: Improved Zero Learning](https://arxiv.org/abs/2001.09832) (2020). 완전 합성곱 정책과 전역 풀링 가치로 여러 보드게임을 학습하는 프레임워크. 19×19 Hex에서 강한 사람을 이겼습니다.
- [Targeted Search Control in AlphaZero](https://arxiv.org/abs/2302.12359) (Go-Exploit, AAMAS 2023). 자기대국을 시작 판이 아니라 지난 판들의 국면에서 시작합니다.
- [Improving AlphaZero Using Monte-Carlo Graph Search](https://ojs.aaai.org/index.php/ICAPS/article/view/15952) (ICAPS 2021). 다른 수순으로 같은 판에 오면 탐색 통계를 공유합니다.
- [Scaling Scaling Laws with Board Games](https://arxiv.org/abs/2104.03113) (Jones 2021). 판이 커지면 필요한 계산이 지수적으로 늘고, 학습 계산과 대국 때의 탐색을 서로 바꿀 수 있습니다.
- [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961) (AlphaGo, 2016). 가치 신경망이 판을 외우는 문제:
  한 판의 국면을 모두 쓰면 학습 0.19 / 시험 0.37로 벌어지고, 한 판에서 국면 하나만 뽑으니 0.226 / 0.234가 됐습니다.
- [MiniZero](https://arxiv.org/abs/2310.11305) (2023). 작은 예산의 AlphaZero, MuZero 비교. 8×8 오델로에서 같은 시간이면 Gumbel 2~16번이 AlphaZero 200번과 비슷하거나 낫습니다.
- [MuZero](https://arxiv.org/abs/1911.08265) (2019). Reanalyze에서 가치가 과적합하지 않도록 가치 손실 가중치를 0.25로 둡니다.
- [Lessons from AlphaZero, part 4: Improving the Training Target](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628) (Oracle, 2018). Connect 4에서 가치 정답을 승패(z)와 탐색 평가(q)의 평균으로 둡니다 (`--value-mix`).

## 주사위와 숨긴 정보

- [Planning in Stochastic Environments with a Learned Model](https://iclr.cc/virtual/2022/poster/6832) (Stochastic MuZero, ICLR 2022). 탐색에 확률 노드를 둡니다 (afterstate).
- [Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/75048/) (Cowling 외, 2012). 숨긴 정보가 있는 게임의 탐색.
- [Student of Games](https://arxiv.org/abs/2112.03178) (Schmid 외, 2023). 완전정보와 숨긴 정보를 한 알고리즘으로.

## 오픈소스

| 이름 | 쓸모 |
|---|---|
| [KataGo](https://github.com/lightvector/KataGo) | 학습 기법 문서와 구현 |
| [mctx](https://github.com/deepmind/mctx) | Gumbel 탐색의 공식 구현 (JAX) |
| [LightZero](https://github.com/opendilab/LightZero) | MCTS 계열 알고리즘 여러 개를 한곳에서 비교 |
| [OpenSpiel](https://github.com/google-deepmind/open_spiel) | 숨긴 정보, 동시 행동을 포함한 여러 게임과 알고리즘 |
