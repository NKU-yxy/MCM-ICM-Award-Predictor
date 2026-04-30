"""
序数阈值标定器

将连续质量分数映射到序数奖项概率 (O/F/M/H/S)。

核心思路:
- 用序数 probit 模型: P(score >= threshold_k) = Φ((score - threshold_k) / temperature)
- 类概率通过累积概率差分得到
- 阈值从 O 奖训练分数分布 + COMAP 先验约束联合估计

相比手设高斯峰值 (O=97, F=93, M=86, H=75, S=58)，
本模块从数据中学习阈值，同时受 COMAP 先验约束防止不合理估计。
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# 奖项从高到低的顺序
AWARD_ORDER = ["O", "F", "M", "H", "S"]

# 默认的 COMAP 先验 (全局平均)
DEFAULT_COMAP_PRIORS = {"O": 0.002, "F": 0.01, "M": 0.08, "H": 0.17, "S": 0.738}


class OrdinalCalibrator:
    """
    序数 probit 标定器

    将 0-100 质量分数映射到 5 个序数奖项的概率。

    模型:
      P(award >= tier_k | score) = Φ((score - threshold_k) / temperature)
      P(award == tier_k) = P(award >= tier_k) - P(award >= tier_{k+1})
    """

    def __init__(self, temperature: float = 2.0):
        """
        参数:
            temperature: probit 边界的软硬程度。越小越接近硬阈值，越大越平滑。
        """
        self.temperature = temperature

        # 4 个阈值，划分 5 个奖项等级
        # thresholds[0]: O/F 边界, thresholds[1]: F/M 边界,
        # thresholds[2]: M/H 边界, thresholds[3]: H/S 边界
        self.thresholds: Optional[np.ndarray] = None

        self.fitted = False

    def fit(
        self,
        scores: np.ndarray,
        comap_priors: Optional[Dict[str, float]] = None,
    ):
        """
        从 O 奖训练分数分布和 COMAP 先验估计序数阈值。

        策略:
          1. 训练数据全为 O 奖论文 — 它们的分数分布代表 "O 奖水平"
          2. O 奖训练分数的低分尾部 ≈ O 奖的下边界
          3. 更低奖项的阈值通过 COMAP 先验约束推断:
             - 训练分数在 O 奖中排 p-th 百分位 ≈ 在全体参赛者中排位可由先验反推
             - 例如，如果 O 奖占 0.17%，则训练集最低分大致对应 top 0.17%
          4. 用 COMAP 累积比例外推各阈值

        参数:
            scores: (N,) O 奖训练集的 0-100 质量分数
            comap_priors: dict 如 {'O': 0.002, 'F': 0.01, ...}，None 则用默认值
        """
        if comap_priors is None:
            comap_priors = DEFAULT_COMAP_PRIORS

        n = len(scores)
        if n < 5:
            logger.warning("训练样本太少，使用默认阈值")
            self._set_default_thresholds()
            self.fitted = True
            return

        scores_sorted = np.sort(scores)
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores))
        score_median = float(np.median(scores))
        score_min = float(np.min(scores))

        logger.info(
            f"OrdinalCalibrator: fitting on {n} O-award scores "
            f"(mean={score_mean:.1f}, std={score_std:.1f}, "
            f"median={score_median:.1f}, min={score_min:.1f})"
        )

        # 核心逻辑:
        # 训练集全是 O 奖论文。训练集最低分 ≈ O 奖在大样本中的下边界。
        # 假设 O 奖分数近似正态，用训练分布参数推断更低奖项的阈值。

        # O/F 边界: 训练集最低分的 5th 百分位以下 — 即便在 O 奖中也算弱的
        # 这里用训练分数的下尾来定 O/F 边界
        o_f_boundary = float(np.percentile(scores, 3))

        # 用 COMAP 累积比例计算各奖级的分位点
        # COMAP: O=0.2%, F=1.0%, M=8%, H=17%, S=72%
        # 累积: O以上=0.2%, F以上=1.2%, M以上=9.2%, H以上=26.2%
        p_O = comap_priors.get("O", 0.002)
        p_F = comap_priors.get("F", 0.01)
        p_M = comap_priors.get("M", 0.08)
        p_H = comap_priors.get("H", 0.17)

        cum_O = p_O
        cum_F = p_O + p_F
        cum_M = p_O + p_F + p_M
        cum_H = p_O + p_F + p_M + p_H

        # 假设分数服从正态分布 N(score_mean, score_std)
        # 但训练数据截断在 O 奖水平
        # 对全体参赛者，O 奖是最高的 cum_O 比例
        # z_O = Φ^{-1}(1 - cum_O) ≈ 对应 O 奖在全体中的 z-score
        z_O = norm.ppf(max(1 - cum_O, 1e-10))
        z_F = norm.ppf(max(1 - cum_F, 1e-10))
        z_M = norm.ppf(max(1 - cum_M, 1e-10))
        z_H = norm.ppf(max(1 - cum_H, 1e-10))

        # 假设 O 奖训练均值对应 z_O，则全体分数均值 = score_mean - z_O * score_std
        pop_mean = score_mean - z_O * score_std

        # 各阈值: threshold_k = pop_mean + z_k * score_std
        thresholds = np.array([
            pop_mean + z_F * score_std,  # O/F 边界
            pop_mean + z_M * score_std,  # F/M 边界
            pop_mean + z_H * score_std,  # M/H 边界
            pop_mean + 0.0 * score_std,  # H/S 边界 (z≈0 即全体均值)
        ])

        # 约束: 阈值必须单调递减且在合理范围
        # O/F 不应低于训练最低分太多
        thresholds[0] = max(thresholds[0], o_f_boundary - 0.5 * score_std)
        thresholds[0] = min(thresholds[0], score_mean + 1.0 * score_std)

        # 强制单调递减
        for i in range(1, 4):
            thresholds[i] = min(thresholds[i], thresholds[i - 1] - 2.0)
            thresholds[i] = max(thresholds[i], 0.0)

        self.thresholds = thresholds
        self.fitted = True

        logger.info(
            f"OrdinalCalibrator: thresholds = "
            f"O/F={thresholds[0]:.1f}, F/M={thresholds[1]:.1f}, "
            f"M/H={thresholds[2]:.1f}, H/S={thresholds[3]:.1f}"
        )

    def _set_default_thresholds(self):
        """设置默认阈值（当训练数据不足时）"""
        # 对应当前高斯峰值 (O=97, F=93, M=86, H=75, S=58) 的 probit 等价
        self.thresholds = np.array([94.0, 88.0, 79.0, 65.0])

    def predict_proba(self, score: float) -> Dict[str, float]:
        """
        将质量分数映射到 5 个奖项的概率分布。

        参数:
            score: 0-100 质量分数

        返回:
            {'O': 0.03, 'F': 0.07, 'M': 0.25, 'H': 0.35, 'S': 0.30}
        """
        if not self.fitted or self.thresholds is None:
            raise RuntimeError("OrdinalCalibrator 尚未训练")

        # P(score >= threshold_k)
        cum_probs = np.array([
            norm.cdf((score - t) / self.temperature)
            for t in self.thresholds
        ])

        # 类概率 = 累积概率差分
        p_O = cum_probs[0]
        p_F = max(cum_probs[1] - cum_probs[0], 0.0)
        p_M = max(cum_probs[2] - cum_probs[1], 0.0)
        p_H = max(cum_probs[3] - cum_probs[2], 0.0)
        p_S = max(1.0 - cum_probs[3], 0.0)

        probs = {"O": p_O, "F": p_F, "M": p_M, "H": p_H, "S": p_S}

        # 归一化
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def get_params(self) -> dict:
        """获取模型参数（用于序列化）"""
        return {
            "temperature": self.temperature,
            "thresholds": self.thresholds,
            "fitted": self.fitted,
        }

    def load_params(self, params: dict):
        """加载模型参数"""
        self.temperature = params.get("temperature", 2.0)
        self.thresholds = params.get("thresholds")
        self.fitted = params.get("fitted", False)
