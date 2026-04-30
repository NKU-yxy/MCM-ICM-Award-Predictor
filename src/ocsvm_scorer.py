"""
OC-SVM 质量打分器

用 One-Class SVM 学习 O 奖论文的特征边界，
以 decision_function 的 signed distance 作为论文质量分数。

相比单质心余弦相似度，OC-SVM 可以捕获 O 奖论文的多模态分布
（不同题目类型、不同建模风格可能形成多个簇），
提供更细粒度的质量区分能力。
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OCSVMScorer:
    """
    OC-SVM 论文质量打分器

    流程:
    1. PCA 降维 (942 → ~50, 保留 >95% 方差)
    2. OneClassSVM 学习 O 奖论文的紧凑边界
    3. decision_function 输出 signed distance 作为原始分数
    4. 通过训练集 O 奖论文的分数分布将原始分数映射到 0-100
    """

    def __init__(
        self,
        pca_components: int = 50,
        nu: float = 0.1,
        gamma: str = "scale",
        kernel: str = "rbf",
    ):
        self.pca_components = pca_components
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel

        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.ocsvm: Optional[OneClassSVM] = None

        # 分数映射参数 (从训练集 O 奖论文学习)
        self.score_median: float = 0.0
        self.score_mad: float = 1.0  # median absolute deviation
        self.score_p5: float = 0.0

        self.fitted = False

    def fit(self, features: np.ndarray) -> np.ndarray:
        """
        在 O 奖论文特征上训练 OC-SVM 并标定分数映射。

        参数:
            features: (N, D) O 奖论文融合特征矩阵

        返回:
            scores: (N,) 0-100 标定后的质量分数
        """
        n_samples, n_features = features.shape

        # 决定 PCA 组件数
        pca_n = min(self.pca_components, n_samples // 3, n_features)
        pca_n = max(pca_n, 10)
        logger.info(f"OC-SVM: PCA {n_features} → {pca_n} (n_samples={n_samples})")

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        # PCA 降维
        self.pca = PCA(n_components=pca_n)
        X_pca = self.pca.fit_transform(X_scaled)
        logger.info(
            f"OC-SVM: PCA explained variance = {self.pca.explained_variance_ratio_.sum():.3f}"
        )

        # 训练 OC-SVM
        self.ocsvm = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
        )
        self.ocsvm.fit(X_pca)

        # 训练集 decision scores
        raw_scores = self.ocsvm.decision_function(X_pca)

        # 标定分数映射 (基于训练 O 奖论文的分布)
        self.score_median = float(np.median(raw_scores))
        self.score_mad = float(np.median(np.abs(raw_scores - self.score_median)))
        self.score_mad = max(self.score_mad, 1e-6)
        self.score_p5 = float(np.percentile(raw_scores, 5))

        # 将训练集原始分数映射到 0-100
        scores = self._normalize_scores(raw_scores)

        n_inliers = int(np.sum(raw_scores >= 0))
        logger.info(
            f"OC-SVM: trained (nu={self.nu}), "
            f"{n_inliers}/{n_samples} inliers ({n_inliers/n_samples*100:.0f}%), "
            f"raw median={self.score_median:.3f}, MAD={self.score_mad:.3f}"
        )
        logger.info(
            f"OC-SVM: score mapping — median→85, p5={self.score_p5:.3f}→50"
        )

        self.fitted = True
        return scores

    def score(self, features: np.ndarray) -> float:
        """
        对单篇或一批论文打分。

        参数:
            features: (D,) 单篇 或 (N, D) 批量

        返回:
            float 或 (N,) 0-100 分数
        """
        if not self.fitted:
            raise RuntimeError("OCSVMScorer 尚未训练，请先调用 fit()")

        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        X_scaled = self.scaler.transform(features)
        X_pca = self.pca.transform(X_scaled)
        raw = self.ocsvm.decision_function(X_pca)
        scores = self._normalize_scores(raw)

        return float(scores[0]) if single else scores

    def _normalize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        将 OC-SVM decision_function 原始分数映射到 0-100。

        映射策略:
          - 训练集 O 奖中位数 → 85 分
          - 训练集 O 奖 5th 百分位 → 50 分
          - 高于 95th 百分位 → 接近 100 分
          - 远低于 O 奖分布 → 趋向 0 分

        使用稳健的 sigmoid 型映射，避免线性外推的极端值问题。
        """
        # 以中位数为参考中心
        centered = (raw_scores - self.score_median) / self.score_mad

        # sigmoid 映射: 中心=85, 范围从 p5→50 到 p95→98
        # 在中心附近近似线性，远处平滑饱和
        scaled = 85.0 + 35.0 * np.tanh(centered * 0.7)

        return np.clip(scaled, 0.0, 100.0)

    def get_params(self) -> dict:
        """获取模型参数（用于序列化到 scoring_model.pkl）"""
        return {
            "pca_components": self.pca_components,
            "nu": self.nu,
            "gamma": self.gamma,
            "kernel": self.kernel,
            "pca": self.pca,
            "scaler": self.scaler,
            "ocsvm": self.ocsvm,
            "score_median": self.score_median,
            "score_mad": self.score_mad,
            "score_p5": self.score_p5,
            "fitted": self.fitted,
        }

    def load_params(self, params: dict):
        """加载模型参数"""
        self.pca_components = params.get("pca_components", 50)
        self.nu = params.get("nu", 0.1)
        self.gamma = params.get("gamma", "scale")
        self.kernel = params.get("kernel", "rbf")
        self.pca = params.get("pca")
        self.scaler = params.get("scaler")
        self.ocsvm = params.get("ocsvm")
        self.score_median = params.get("score_median", 0.0)
        self.score_mad = params.get("score_mad", 1.0)
        self.score_p5 = params.get("score_p5", 0.0)
        self.fitted = params.get("fitted", False)
