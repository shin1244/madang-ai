"""여러 게임이 함께 쓰는 정책·가치 신경망 (docs/plan.md의 "구조").

게임마다 보드마당 신경망(boardmadang/trainer/net.py)과 같은 입력과 출력을 쓰고, 몸통은 모든 게임이 같이 씁니다.

- 입력: 게임마다 뜻이 다른 채널(기물, observe())은 게임별 입력층을, 모든 게임에서 뜻이 같은 13채널
  (표시 7, 칠한 칸 3, 숫자, 두는 쪽, 판 안)은 공통 입력층을 지나 더해집니다.
- 몸통: 잔차 블록. 정규화의 평균·분산은 게임마다 따로 잽니다 (판 크기와 기물이 달라 분포가 다르고,
  한 묶음에는 한 게임만 들어가므로 같이 재면 평가 때의 평균이 마지막에 본 게임 쪽으로 쏠림).
  크기와 치우침은 게임 벡터에서 냅니다 (FiLM). 가운데 블록에서 판 전체를 모은 값을 칸마다 더합니다 (KataGo).
- 머리: 칸 클릭 점수와 가치는 공통, 특수 행동 점수만 게임별.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 모든 게임에서 뜻이 같은 관측 채널 (boardmadang/trainer/env.js의 channelNames). 신경망이 "판 안"(전부 1)을 하나 더 붙입니다
COMMON = [
    "mark:dot",
    "mark:ring",
    "mark:last-move",
    "mark:danger",
    "mark:ghost",
    "mark:blocked",
    "mark:frame",
    "tint:0",
    "tint:1",
    "tint:other",
    "text",
    "toPlay",
]
# 게임마다 저장하는 정보 (selfplay.js의 H에서)
GAME_KEYS = ("id", "rows", "cols", "channelNames", "specials", "nActions")


def split_channels(names):
    """관측 채널 이름 → (공통 채널의 자리를 COMMON 순서로, 게임별 채널의 자리)"""
    missing = [n for n in COMMON if n not in names]
    if missing:
        raise ValueError(f"observation has no {', '.join(missing)} channel (boardmadang env.js changed?)")
    return [names.index(n) for n in COMMON], [i for i, n in enumerate(names) if n not in COMMON]


def pool(x):
    """판 전체를 평균과 최댓값으로 모읍니다 (보드 크기와 상관없는 특징)"""
    return torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)


class GameNorm(nn.Module):
    """배치 정규화. 평균·분산은 게임마다 따로 잽니다 (BatchNorm2d와 같은 momentum 0.1, eps 1e-5).
    크기·치우침은 SharedNet이 게임 벡터에서 모든 층 것을 한 번에 내서 film[0][k], film[1][k]로 넘겨 줍니다 (FiLM).
    층마다 작은 계산을 따로 하면 작은 판에서는 신경망 한 번이 두 배 넘게 느려집니다."""

    def __init__(self, filters, games, k):
        super().__init__()
        self.filters = filters
        self.k = k  # SharedNet의 FiLM에서 몇 번째 층인지
        self.register_buffer("mean", torch.zeros(games, filters))
        self.register_buffer("var", torch.ones(games, filters))

    def forward(self, x, g, film):
        # 학습 중이면 mean[g], var[g]가 제자리에서 갱신됩니다 (다른 게임의 줄은 그대로)
        return F.batch_norm(x, self.mean[g], self.var[g], film[0][self.k], film[1][self.k], self.training, 0.1, 1e-5)


class Block(nn.Module):
    """잔차 블록. pooled면 첫 합성곱 뒤에 판 전체를 모은 값을 칸마다 더합니다"""

    def __init__(self, filters, games, norms, pooled=False):
        super().__init__()
        self.c1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.n1 = norms(filters)
        self.c2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.n2 = norms(filters)
        self.pooled = nn.Linear(2 * filters, filters) if pooled else None

    def forward(self, x, g, film):
        y = F.relu(self.n1(self.c1(x), g, film))
        if self.pooled is not None:
            y = y + self.pooled(pool(y))[:, :, None, None]
        y = self.n2(self.c2(y), g, film)
        return F.relu(x + y)


class SharedNet(nn.Module):
    """games는 게임마다 selfplay.js의 H (GAME_KEYS만 씀). forward(x, g)의 g는 games에서 몇 번째 게임인지입니다.
    입력 x [n, 그 게임의 채널, rows, cols] → (logits [n, 그 게임의 행동 수], value [n]). 보드마당 신경망과 같은 모양입니다."""

    def __init__(self, games, blocks=4, filters=64, dim=16):
        super().__init__()
        games = [{k: g[k] for k in GAME_KEYS} for g in games]
        ids = [g["id"] for g in games]
        if len(set(ids)) != len(ids):
            raise ValueError(f"the same game twice: {ids}")
        self.config = {"games": games, "blocks": blocks, "filters": filters, "dim": dim}
        self.ids = ids
        n = len(games)
        self.embed = nn.Embedding(n, dim)
        # 첫 층: 공통 채널(COMMON 순서) 12개, 게임별 채널, "판 안"의 순서로 이어 붙인 입력에 한 번에 합성곱합니다.
        # 가중치는 공통 몫(common_in, 마지막 칸이 판 안)과 게임별 몫(own_in)으로 나눠 둡니다
        self.common_in = nn.Parameter(torch.empty(filters, len(COMMON) + 1, 3, 3))
        self.own_in = nn.ParameterDict()
        for i, g in enumerate(games):
            common, own = split_channels(g["channelNames"])
            # 채널 자리는 config에서 다시 만들 수 있으므로 저장하지 않습니다
            self.register_buffer(f"order_{i}", torch.tensor(common + own), persistent=False)
            if own:
                self.own_in[str(i)] = nn.Parameter(torch.empty(filters, len(own), 3, 3))
        for w in [self.common_in, *self.own_in.values()]:
            nn.init.kaiming_uniform_(w, a=5**0.5)  # nn.Conv2d와 같은 초기화

        norms = []

        def norm(width):
            norms.append(GameNorm(width, n, len(norms)))
            return norms[-1]

        self.stem = norm(filters)
        self.trunk = nn.ModuleList(Block(filters, n, norm, pooled=k == blocks // 2) for k in range(blocks))
        self.policy = nn.Conv2d(filters, 32, 1, bias=False)
        self.policy_norm = norm(32)
        self.click = nn.Conv2d(32, 1, 1)
        self.special = nn.ModuleDict(
            {str(i): nn.Linear(64, len(g["specials"])) for i, g in enumerate(games) if g["specials"]}
        )
        self.value = nn.Conv2d(filters, 32, 1, bias=False)
        self.value_norm = norm(32)
        self.value_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        # FiLM: 게임 벡터 → 모든 정규화 층의 크기와 치우침. bias가 모든 게임이 같이 쓰는 몫, weight가 게임마다의 차이이고
        # 처음에는 차이 없이(크기 1, 치우침 0) 시작합니다
        self.widths = [m.filters for m in norms]
        self.film = nn.Linear(dim, 2 * sum(self.widths))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def index(self, game_id):
        """게임 id → forward의 g"""
        if game_id not in self.ids:
            raise KeyError(f"{game_id} is not one of this model's games ({', '.join(self.ids)})")
        return self.ids.index(game_id)

    def forward(self, x, g):
        scale, shift = self.film(self.embed.weight[g]).chunk(2)
        film = ((1 + scale).split(self.widths), shift.split(self.widths))
        x = torch.cat([x.index_select(1, getattr(self, f"order_{g}")), torch.ones_like(x[:, :1])], 1)
        w = self.common_in
        if str(g) in self.own_in:
            w = torch.cat([w[:, :-1], self.own_in[str(g)], w[:, -1:]], 1)
        h = F.relu(self.stem(F.conv2d(x, w, padding=1), g, film))
        for block in self.trunk:
            h = block(h, g, film)
        p = F.relu(self.policy_norm(self.policy(h), g, film))
        logits = self.click(p).flatten(1)  # row * cols + col 순서 (selfplay.js의 행동 번호와 같음)
        if str(g) in self.special:
            logits = torch.cat([logits, self.special[str(g)](pool(p))], 1)
        v = torch.tanh(self.value_fc(pool(F.relu(self.value_norm(self.value(h), g, film))))).squeeze(1)
        return logits, v


class GamePath(nn.Module):
    """공통 신경망에서 게임 하나의 경로. 보드마당 신경망처럼 forward(x)만 받습니다 (채점, 내보내기)"""

    def __init__(self, net, g):
        super().__init__()
        self.net = net
        self.g = g

    def forward(self, x):
        return self.net(x, self.g)


def load(saved, device="cpu"):
    """torch.load한 공통 모델 → 평가 모드의 SharedNet"""
    net = SharedNet(**saved["config"]).to(device)
    net.load_state_dict(saved["state"])
    return net.eval()


@torch.no_grad()
def init_from(net, saved):
    """다른 공통 모델(saved, torch.load한 model.pt)의 가중치로 net을 시작합니다. 게임 목록이 달라도 됩니다.

    모든 게임이 같이 쓰는 층은 그대로 옮깁니다 (몸통, 머리, 공통 입력층, FiLM. 크기가 같아야 함).
    게임마다의 부품(게임 벡터, 입력층, 정규화의 평균·분산, 특수 행동 머리)은 게임 id로 맞춰 옮깁니다.
    관측 채널이 바뀐 게임의 입력층과 특수 행동이 바뀐 게임의 머리는 새로 시작합니다.
    새 게임의 게임 벡터와 평균·분산은 옮긴 게임들의 평균에서 시작합니다 ("보통 게임"에서 출발).
    {"kept": 옮긴 게임, "fresh": 새 게임, "dropped": 빠진 게임, "layers": 그대로 옮긴 층 수}를 돌려줍니다."""
    old_games = {g["id"]: (i, g) for i, g in enumerate(saved["config"]["games"])}
    src = saved["state"]
    pairs = [(new, old_games[g["id"]][0]) for new, g in enumerate(net.config["games"]) if g["id"] in old_games]
    if not pairs:
        raise ValueError("the models share no game")
    kept = {new for new, _ in pairs}
    fresh = [i for i in range(len(net.ids)) if i not in kept]
    rows = [n for n, m in net.named_modules() if isinstance(m, GameNorm)]
    per_game_rows = {"embed.weight", *(f"{n}.{k}" for n in rows for k in ("mean", "var"))}
    state = net.state_dict()
    layers = 0
    for key, value in state.items():
        old = src.get(key)
        if key in per_game_rows:
            if old is None or old.shape[1:] != value.shape[1:]:
                raise ValueError(f"{key}: the models differ in size ({tuple(old.shape) if old is not None else None} vs {tuple(value.shape)})")
            for new, i in pairs:
                value[new] = old[i]
            if fresh:
                value[fresh] = old[[i for _, i in pairs]].mean(0)
        elif key.startswith(("own_in.", "special.")):
            part, g, *rest = key.split(".")
            g = int(g)
            match = [i for new, i in pairs if new == g]
            if not match:
                continue  # 새 게임: 새로 시작
            old_key = ".".join([part, str(match[0]), *rest])
            if old_key in src and src[old_key].shape == value.shape:
                value.copy_(src[old_key])
        else:
            if old is None or old.shape != value.shape:
                raise ValueError(f"{key}: the models differ in size (use the same --blocks, --filters, --dim)")
            value.copy_(old)
            layers += 1
    ids = net.ids
    return {
        "kept": [ids[new] for new, _ in pairs],
        "fresh": [ids[i] for i in fresh],
        "dropped": [g for g in old_games if g not in ids],
        "layers": layers,
    }
