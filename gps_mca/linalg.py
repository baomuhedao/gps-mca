"""
纯 Python 线性代数工具库
替代 numpy，实现 GPS-MCA 所需的全部向量/矩阵运算。
"""

from __future__ import annotations

import math
import random as _random


# ─────────────────────────────────────────────
# 可复现随机数生成器
# ─────────────────────────────────────────────
class RNG:
    """带种子的随机数生成器"""

    def __init__(self, seed: int | None = None):
        self._r = _random.Random(seed)

    def random(self) -> float:
        return self._r.random()

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._r.gauss(mu, sigma)

    def normal_vec(self, n: int, mu: float = 0.0, sigma: float = 1.0) -> Vec:
        return [self.gauss(mu, sigma) for _ in range(n)]

    def uniform_vec(self, n: int, lo: float = -1.0, hi: float = 1.0) -> Vec:
        return [self._r.uniform(lo, hi) for _ in range(n)]


# ─────────────────────────────────────────────
# 类型别名
# ─────────────────────────────────────────────
Vec = list[float]
Mat = list[list[float]]


# ─────────────────────────────────────────────
# 向量运算
# ─────────────────────────────────────────────
def vec_zeros(n: int) -> Vec:
    return [0.0] * n


def vec_add(a: Vec, b: Vec) -> Vec:
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: Vec, b: Vec) -> Vec:
    return [x - y for x, y in zip(a, b)]


def vec_scale(v: Vec, s: float) -> Vec:
    return [x * s for x in v]


def vec_dot(a: Vec, b: Vec) -> float:
    return sum(x * y for x, y in zip(a, b))


def vec_norm(v: Vec) -> float:
    return math.sqrt(sum(x * x for x in v))


def vec_mean_scalar(v: Vec) -> float:
    return sum(v) / len(v) if v else 0.0


def vec_std(v: Vec) -> float:
    if not v:
        return 0.0
    m = vec_mean_scalar(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def vec_rmse(a: Vec, b: Vec) -> float:
    if not a:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def vec_cosine_sim(a: Vec, b: Vec) -> float:
    na, nb = vec_norm(a), vec_norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return vec_dot(a, b) / (na * nb)


def vec_elementwise_mean(vectors: list[Vec]) -> Vec:
    if not vectors:
        return []
    n = len(vectors[0])
    k = len(vectors)
    return [sum(v[i] for v in vectors) / k for i in range(n)]


def vec_concat(*vecs: Vec) -> Vec:
    out: Vec = []
    for v in vecs:
        out.extend(v)
    return out


def vec_resize(v: Vec, n: int) -> Vec:
    """将向量调整到长度 n（截断或循环填充）"""
    if len(v) == n:
        return v[:]
    if len(v) > n:
        return v[:n]
    result = []
    while len(result) < n:
        result.extend(v)
    return result[:n]


# ─────────────────────────────────────────────
# 激活函数
# ─────────────────────────────────────────────
def relu(v: Vec) -> Vec:
    return [max(0.0, x) for x in v]


def tanh_vec(v: Vec) -> Vec:
    return [math.tanh(x) for x in v]


def sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def layer_norm(v: Vec) -> Vec:
    m = vec_mean_scalar(v)
    s = vec_std(v) + 1e-8
    return [(x - m) / s for x in v]


# ─────────────────────────────────────────────
# 矩阵运算
# ─────────────────────────────────────────────
def mat_zeros(rows: int, cols: int) -> Mat:
    return [[0.0] * cols for _ in range(rows)]


def mat_identity(n: int) -> Mat:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_random(rows: int, cols: int, scale: float = 0.1,
               rng: RNG | None = None) -> Mat:
    rng = rng or RNG()
    return [[rng.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def mat_vec_mul(M: Mat, v: Vec) -> Vec:
    return [vec_dot(row, v) for row in M]


def vec_mat_mul(v: Vec, M: Mat) -> Vec:
    """v (1×n) × M (n×m) → result (1×m)"""
    cols = len(M[0])
    return [sum(v[i] * M[i][j] for i in range(len(v))) for j in range(cols)]


def mat_add(A: Mat, B: Mat) -> Mat:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def mat_scale(A: Mat, s: float) -> Mat:
    return [[a * s for a in row] for row in A]


def outer_product(a: Vec, b: Vec) -> Mat:
    return [[ai * bj for bj in b] for ai in a]
