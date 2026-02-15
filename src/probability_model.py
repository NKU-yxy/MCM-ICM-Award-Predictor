"""
概率模型模块
基于单类(O奖)数据，用贝叶斯方法估计各奖项概率

核心思路：
1. 只有 O/F 奖论文样本 → 用这些建立"O奖特征分布"
2. 用 KDE (核密度估计) 对 O 奖论文的余弦相似度建模
3. 利用 MCM/ICM 的已知获奖比例作为先验 (prior)
4. 假设: 论文质量越高 → 余弦相似度越高 → 获得高奖概率越大
5. 用 Bayesian 方法将 similarity score 映射为各奖项的后验概率

关键假设（基于合理推断）：
- O/F 奖论文的相似度最高，集中在分布右端
- M 奖论文相似度略低，但仍较高
- H 奖论文相似度中等
- S 奖论文相似度最分散，整体偏低
- 各奖项的相似度分布有重叠，但中心逐级递减
"""

import numpy as np
from typing import Dict, Tuple
from scipy.stats import norm, gaussian_kde
from scipy.special import softmax
import logging

logger = logging.getLogger(__name__)


# MCM/ICM 历年获奖比例先验 (基于 COMAP 官方数据)
# 这些比例相对稳定，是公开信息
AWARD_PRIORS = {
    'O': 0.01,    # ~1% Outstanding Winner
    'F': 0.02,    # ~2% Finalist
    'M': 0.08,    # ~8% Meritorious Winner
    'H': 0.17,    # ~17% Honorable Mention
    'S': 0.72,    # ~72% Successful Participant
}

# 合并后的先验
MERGED_PRIORS = {
    'O': 0.01,
    'F': 0.02,
    'M': 0.08,
    'H': 0.17,
    'S': 0.72,
}


class AwardProbabilityEstimator:
    """
    基于单类数据的奖项概率估计器
    
    使用 O 奖论文的相似度分布来推断各奖项概率。
    
    核心方法：
    1. 用 O 奖数据拟合 KDE 得到 p(similarity | O)
    2. 假设各奖项的相似度分布是 O 奖分布的位移/缩放版本
    3. 结合先验概率，用贝叶斯公式计算后验概率
    """
    
    def __init__(self, o_similarities: np.ndarray = None, stats: dict = None):
        """
        初始化估计器
        
        参数:
            o_similarities: O奖论文的余弦相似度数组
            stats: 训练时的统计信息
        """
        self.stats = stats or {}
        self.o_similarities = o_similarities
        
        # 各奖项的相似度分布参数（高斯分布近似）
        # 基于 O 奖分布推断其他奖项的分布
        self.award_distributions = {}
        
        if o_similarities is not None and len(o_similarities) > 3:
            self._fit_distributions(o_similarities)
    
    def _fit_distributions(self, o_similarities: np.ndarray):
        """
        基于 O 奖数据拟合各奖项的相似度分布
        
        策略：
        - O 奖分布直接用 KDE 拟合
        - 其他奖项分布通过对 O 奖分布降档推断:
          * F: 均值稍低，方差略大
          * M: 均值更低，方差更大
          * H: 均值明显更低，方差大
          * S: 均值最低，方差最大
        
        这种「降档推断」是在缺乏负样本时的合理假设：
        高奖论文的特征更集中（方差小），低奖论文的特征更分散（方差大）
        """
        o_mean = np.mean(o_similarities)
        o_std = max(np.std(o_similarities), 0.01)  # 防止零标准差
        
        # O 奖: 直接使用数据统计
        # 注意: 余弦相似度范围在 -1 到 1 之间，O 奖通常在 0.8-1.0
        
        # 降档系数：基于获奖比例和质量梯度的合理估计
        # 越低的奖项，均值向左偏移越多，方差越大
        self.award_distributions = {
            'O': {'mean': o_mean, 'std': o_std},
            'F': {'mean': o_mean - 0.3 * o_std, 'std': o_std * 1.2},
            'M': {'mean': o_mean - 1.0 * o_std, 'std': o_std * 1.5},
            'H': {'mean': o_mean - 2.0 * o_std, 'std': o_std * 2.0},
            'S': {'mean': o_mean - 3.5 * o_std, 'std': o_std * 3.0},
        }
        
        # 尝试用 KDE 拟合 O 奖分布（更精确）
        try:
            if len(o_similarities) >= 5:
                self.o_kde = gaussian_kde(o_similarities, bw_method='silverman')
            else:
                self.o_kde = None
        except Exception:
            self.o_kde = None
        
        logger.info(f"概率模型已拟合:")
        for award, params in self.award_distributions.items():
            logger.info(f"  {award}: mean={params['mean']:.4f}, std={params['std']:.4f}")
    
    def estimate_probabilities(self, similarity: float, score: float = None, 
                                structure_score: float = None) -> Dict[str, float]:
        """
        估计各奖项的概率
        
        参数:
            similarity: 余弦相似度（与 O 奖质心的距离）
            score: 0-100 分数（可选，用于校正）
            structure_score: 论文结构评分（可选，用于校正）
        
        返回:
            {'O': 0.03, 'F': 0.05, 'M': 0.45, 'H': 0.35, 'S': 0.12}
        """
        if not self.award_distributions:
            return self._default_probabilities()
        
        # 计算各奖项的似然 P(similarity | award)
        likelihoods = {}
        for award, params in self.award_distributions.items():
            # 用高斯分布计算似然
            likelihood = norm.pdf(similarity, loc=params['mean'], scale=params['std'])
            likelihoods[award] = likelihood
        
        # 贝叶斯公式: P(award | sim) ∝ P(sim | award) × P(award)
        posteriors = {}
        for award in MERGED_PRIORS:
            prior = MERGED_PRIORS[award]
            likelihood = likelihoods.get(award, 1e-10)
            posteriors[award] = likelihood * prior
        
        # 归一化为概率
        total = sum(posteriors.values())
        if total > 0:
            for award in posteriors:
                posteriors[award] /= total
        else:
            return self._default_probabilities()
        
        # 如果有结构评分，用它来微调概率
        if structure_score is not None:
            posteriors = self._adjust_by_structure(posteriors, structure_score)
        
        return posteriors
    
    def _adjust_by_structure(self, posteriors: Dict[str, float], 
                              structure_score: float) -> Dict[str, float]:
        """
        根据论文结构评分微调概率
        
        高结构评分（接近1.0）→ 略微提升高奖概率
        低结构评分（接近0.0）→ 略微降低高奖概率
        
        使用温和的调整系数，避免过度影响
        """
        # structure_score 在 0-1 之间
        # 调整强度（不宜太大，结构评分只是辅助信号）
        adjustment_strength = 0.15
        
        # 计算调整因子: structure_score = 0.5 → 无调整
        # > 0.5 → 偏向高奖, < 0.5 → 偏向低奖
        bias = (structure_score - 0.5) * adjustment_strength
        
        # 奖项权重梯度
        award_order = {'O': 2, 'F': 1.5, 'M': 0.5, 'H': -0.5, 'S': -1.5}
        
        adjusted = {}
        for award, prob in posteriors.items():
            factor = 1.0 + bias * award_order.get(award, 0)
            adjusted[award] = max(prob * factor, 1e-6)
        
        # 重新归一化
        total = sum(adjusted.values())
        for award in adjusted:
            adjusted[award] /= total
        
        return adjusted
    
    def _default_probabilities(self) -> Dict[str, float]:
        """返回默认概率（使用先验）"""
        return dict(MERGED_PRIORS)
    
    def get_parameters(self) -> dict:
        """获取模型参数（用于保存）"""
        return {
            'award_distributions': self.award_distributions,
            'priors': MERGED_PRIORS,
        }
    
    def load_parameters(self, params: dict):
        """加载模型参数"""
        self.award_distributions = params.get('award_distributions', {})
    
    @staticmethod
    def get_award_description(probs: Dict[str, float]) -> str:
        """
        根据概率分布生成文字描述
        """
        # 找到最可能的奖项
        best_award = max(probs, key=probs.get)
        best_prob = probs[best_award]
        
        award_names = {
            'O': '特等奖 (Outstanding Winner)',
            'F': '特等奖提名 (Finalist)',
            'M': '一等奖 (Meritorious Winner)',
            'H': '二等奖 (Honorable Mention)',
            'S': '成功参赛 (Successful Participant)',
        }
        
        description = f"最可能获得: {award_names.get(best_award, best_award)}"
        
        if best_prob > 0.6:
            description += " (高置信度)"
        elif best_prob > 0.4:
            description += " (中等置信度)"
        else:
            description += " (低置信度，多个奖项概率接近)"
        
        return description
    
    @staticmethod
    def format_probabilities(probs: Dict[str, float]) -> str:
        """格式化输出概率分布"""
        award_order = ['O', 'F', 'M', 'H', 'S']
        award_names = {
            'O': '特等奖 (O)',
            'F': '特等提名(F)',
            'M': '一等奖 (M)',
            'H': '二等奖 (H)',
            'S': '成功参赛(S)',
        }
        
        lines = []
        for award in award_order:
            if award in probs:
                prob = probs[award]
                bar_len = int(prob * 40)
                bar = "█" * bar_len
                marker = " ★" if prob == max(probs.values()) else ""
                lines.append(f"  {award_names[award]:12s} ({prob*100:5.1f}%): {bar}{marker}")
        
        return "\n".join(lines)
