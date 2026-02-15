"""
增强版概率模型模块 (v2)

核心改进:
1. 使用每题(A-F)独立的获奖比例先验（来自 COMAP 官方数据）
2. 结合题目类型特征偏好进行似然调整
3. 多信号融合：相似度 + 结构评分 + 题目适配度 + 写作质量
4. 分层贝叶斯：先用相似度粗筛，再用题目特征精调

解决「缺乏 H/S/F/O 样本」的方法:
- 不依赖 H/S/F 样本训练分类器
- 用 O 奖质心的相似度作为「论文质量信号」
- 结合官方获奖比例先验（贝叶斯后验估计）
- 用结构/写作质量等规则信号补充区分度

数学原理:
  P(Award=k | features) ∝ P(features | Award=k) × P(Award=k)

  其中:
  - P(Award=k) = 题目特定的先验概率 (来自 award_prior.py)
  - P(features | Award=k) = 似然函数 (基于O奖分布的降档推断 + 题目适配修正)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.stats import norm, gaussian_kde
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.award_prior import (
    get_award_prior, 
    get_problem_profile, 
    get_competition_intensity,
    get_problem_difficulty_adjustment,
    PROBLEM_TYPE_PROFILES,
)

logger = logging.getLogger(__name__)


class EnhancedAwardEstimator:
    """
    增强版奖项概率估计器
    
    改进点：
    1. 题目感知的先验概率
    2. 多维质量信号融合
    3. 题目适配度评分
    4. 更精细的贝叶斯推断
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
        self.award_distributions = {}
        
        if o_similarities is not None and len(o_similarities) > 3:
            self._fit_distributions(o_similarities)
    
    def _fit_distributions(self, o_similarities: np.ndarray):
        """
        基于 O 奖数据拟合各奖项的相似度分布
        
        改进: 使用更合理的降档系数，基于实际获奖比例反推
        """
        o_mean = np.mean(o_similarities)
        o_std = max(np.std(o_similarities), 0.01)
        
        # 使用 probit 模型的思路:
        # 假设论文质量服从正态分布，各奖项是不同的分位点
        # O: top ~0.2% → z ≈ 2.88
        # F: top ~1%   → z ≈ 2.33
        # M: top ~9%   → z ≈ 1.34
        # H: top ~26%  → z ≈ 0.64
        # S: 其余 74%  → z < 0.64
        
        # 相似度分布参数（高斯近似）
        # 均值偏移量和标准差基于质量梯度
        self.award_distributions = {
            'O': {
                'mean': o_mean,
                'std': o_std,
                'z_score': 2.88,  # 对应正态分布的分位点
            },
            'F': {
                'mean': o_mean - 0.25 * o_std,
                'std': o_std * 1.15,
                'z_score': 2.33,
            },
            'M': {
                'mean': o_mean - 0.8 * o_std,
                'std': o_std * 1.4,
                'z_score': 1.34,
            },
            'H': {
                'mean': o_mean - 1.6 * o_std,
                'std': o_std * 1.8,
                'z_score': 0.64,
            },
            'S': {
                'mean': o_mean - 2.8 * o_std,
                'std': o_std * 2.5,
                'z_score': -0.5,
            },
        }
        
        # KDE拟合O奖分布
        try:
            if len(o_similarities) >= 5:
                self.o_kde = gaussian_kde(o_similarities, bw_method='silverman')
            else:
                self.o_kde = None
        except Exception:
            self.o_kde = None
        
        logger.info("增强概率模型已拟合:")
        for award, params in self.award_distributions.items():
            logger.info(f"  {award}: mean={params['mean']:.4f}, std={params['std']:.4f}")
    
    def estimate_probabilities(
        self, 
        similarity: float,
        problem: str = 'A',
        year: int = 2024,
        score: float = None,
        structure_info: dict = None,
        full_text: str = None,
        aspect_scores: dict = None,
    ) -> Dict[str, float]:
        """
        估计各奖项概率（分数主导 + 贝叶斯辅助）
        
        核心改进: 
        - 旧方法用 COMAP 先验(O=0.17%, S=74%) 做贝叶斯推断，
          导致即使最好的 O 奖论文也只能得到 O=0.5%，预测全部偏向 S
        - 新方法: 70% 基于分数的概率映射 + 30% 软化先验的贝叶斯，
          分数直接反映与 O 奖质心的接近程度，比原始相似度更可靠
        
        参数:
            similarity: 与O奖质心的余弦相似度
            problem: 题目 (A-F)
            year: 年份
            score: 0-100分数（核心输入）
            structure_info: 论文结构信息（可选）
            full_text: 论文全文（可选，用于题目适配度评估）
            aspect_scores: 三维度子分（可选）{'abstract':70, 'figures':90, 'modeling':92}
        
        返回:
            {'O': 0.03, 'F': 0.05, 'M': 0.45, 'H': 0.35, 'S': 0.12}
        """
        if not self.award_distributions and score is None:
            return self._default_probabilities(problem, year)
        
        # ========== 第1步: 基于分数的概率映射（主导） ==========
        score_probs = self._score_based_probabilities(score, aspect_scores)
        
        # ========== 第2步: 贝叶斯概率（软化先验，辅助） ==========
        bayesian_probs = self._bayesian_with_soft_priors(
            similarity, problem, year, structure_info, full_text
        )
        
        # ========== 第3步: 加权融合 ==========
        # 分数可靠时以分数为主(70%)，贝叶斯提供题目特异性调整(30%)
        w_score = 0.70
        w_bayes = 0.30
        
        posteriors = {}
        for award in ['O', 'F', 'M', 'H', 'S']:
            posteriors[award] = (
                w_score * score_probs.get(award, 0) +
                w_bayes * bayesian_probs.get(award, 0)
            )
        
        # 最终归一化
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        else:
            return self._default_probabilities(problem, year)
        
        return posteriors
    
    def _score_based_probabilities(
        self, score: float, aspect_scores: dict = None,
    ) -> Dict[str, float]:
        """
        基于论文质量分数的概率映射
        
        原理:
        - 分数是与 O 奖质心的相似度经标定后的 0-100 值
        - 训练/验证集全部是 O 奖论文:
          * 训练集均分 ≈ 85, 验证/测试均分 ≈ 79-80
          * 中位数 ≈ 84-85
          * 范围 ≈ 40-100
        - 因此: 分数 ≈ 85 → 典型 O 奖水平, 分数 ≈ 79 → O 奖下游
        
        用高斯混合模型: 每个奖项有一个 "峰值分数" 和 "扩展度"
        - O: 峰值 93, 扩展 5  (分数 88-100 区域主导)
        - F: 峰值 87, 扩展 5  (分数 82-92 区域主导)
        - M: 峰值 80, 扩展 6  (分数 74-86 区域主导)
        - H: 峰值 70, 扩展 7  (分数 63-77 区域主导)
        - S: 峰值 55, 扩展 9  (分数 <64 区域主导)
        """
        if score is None:
            return {'O': 0.03, 'F': 0.07, 'M': 0.25, 'H': 0.30, 'S': 0.35}
        
        # 综合质量分（融合总分与三维度子分）
        if aspect_scores:
            vals = [aspect_scores.get(k, score) 
                    for k in ['abstract', 'figures', 'modeling']]
            avg_aspect = float(np.mean(vals))
            quality = 0.55 * score + 0.45 * avg_aspect
        else:
            quality = score
        
        # 高斯混合模型参数（根据训练/验证集 O 奖论文的分数分布标定）
        # 峰值: 该分数最可能对应的奖项
        # 扩展: 分布的宽度（越低越严格）
        award_params = {
            'O': {'peak': 93, 'spread': 5.0},
            'F': {'peak': 87, 'spread': 5.0},
            'M': {'peak': 80, 'spread': 6.0},
            'H': {'peak': 70, 'spread': 7.0},
            'S': {'peak': 55, 'spread': 9.0},
        }
        
        logits = {}
        for award, params in award_params.items():
            logits[award] = -(quality - params['peak']) ** 2 / (
                2 * params['spread'] ** 2
            )
        
        # Softmax 归一化
        max_logit = max(logits.values())
        exp_logits = {k: np.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_logits.values())
        
        return {k: v / total for k, v in exp_logits.items()}
    
    def _bayesian_with_soft_priors(
        self, similarity: float, problem: str, year: int,
        structure_info: dict = None, full_text: str = None,
    ) -> Dict[str, float]:
        """
        贝叶斯概率估计（使用软化先验）
        
        改进:
        - 旧方法: 直接用 COMAP 先验 (O=0.17%, S=74%) → S 永远占主导
        - 新方法: 将 COMAP 先验与均匀先验混合 (30%/70%)，
          保留题目特异性信息但避免 S 先验压倒一切
        """
        if not self.award_distributions:
            return self._default_probabilities(problem, year)
        
        # 相似度似然
        likelihoods = self._compute_similarity_likelihoods(similarity)
        
        # 软化先验: 30% COMAP + 70% 平坦
        comap_priors = get_award_prior(problem, year)
        flat_priors = {'O': 0.10, 'F': 0.15, 'M': 0.25, 'H': 0.25, 'S': 0.25}
        
        priors = {}
        blend = 0.30  # COMAP 先验的保留比例
        for award in comap_priors:
            priors[award] = (
                blend * comap_priors[award] + 
                (1 - blend) * flat_priors[award]
            )
        
        # 贝叶斯后验
        posteriors = {}
        for award in priors:
            posteriors[award] = likelihoods.get(award, 1e-10) * priors[award]
        
        # 归一化
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {k: v / total for k, v in posteriors.items()}
        else:
            return self._default_probabilities(problem, year)
        
        # 结构评分调整
        if structure_info:
            posteriors = self._adjust_by_structure(posteriors, structure_info, problem)
        
        # 题目适配度调整
        if full_text:
            posteriors = self._adjust_by_problem_fit(posteriors, full_text, problem)
        
        # 竞争强度调整
        posteriors = self._adjust_by_competition(posteriors, problem)
        
        # 归一化
        total = sum(posteriors.values())
        return {k: v / total for k, v in posteriors.items()}
    
    def _compute_similarity_likelihoods(self, similarity: float) -> Dict[str, float]:
        """计算基于相似度的似然"""
        likelihoods = {}
        for award, params in self.award_distributions.items():
            likelihood = norm.pdf(similarity, loc=params['mean'], scale=params['std'])
            likelihoods[award] = max(likelihood, 1e-30)
        return likelihoods
    
    def _adjust_by_structure(self, posteriors: Dict[str, float],
                              structure_info: dict, problem: str) -> Dict[str, float]:
        """
        根据论文结构质量调整概率
        
        结构信号包括:
        - 结构完整度 (有摘要/引言/方法/结果/结论/参考文献)
        - 公式密度
        - 图表数量
        - 引用密度
        - 是否有灵敏度分析/模型验证等高级内容
        
        这些信号对不同题目有不同的权重
        """
        profile = get_problem_profile(problem)
        indicators = profile.get('key_indicators', {})
        
        # 计算结构质量综合评分 (0-1)
        quality_signals = []
        
        # 1. 结构完整度
        completeness = structure_info.get('structure_completeness', 0.5)
        quality_signals.append(completeness * 1.5)  # 权重最高
        
        # 2. 公式密度（根据题目类型加权）
        formula_density = structure_info.get('formula_count', 0) / max(structure_info.get('total_word_count', 1), 1) * 1000
        formula_score = min(formula_density / 15.0, 1.0) * indicators.get('formula_weight', 1.0)
        quality_signals.append(formula_score)
        
        # 3. 图表丰富度
        figure_count = structure_info.get('figure_caption_count', 0)
        table_count = structure_info.get('table_count', 0)
        visual_score = min((figure_count + table_count) / 15.0, 1.0) * indicators.get('figure_weight', 1.0)
        quality_signals.append(visual_score)
        
        # 4. 灵敏度分析（非常重要的O奖指标）
        has_sensitivity = structure_info.get('has_sensitivity_analysis', False)
        has_validation = structure_info.get('has_model_validation', False)
        has_sw = structure_info.get('has_strengths_weaknesses', False)
        advanced_score = (int(has_sensitivity) + int(has_validation) + int(has_sw)) / 3.0
        advanced_score *= indicators.get('sensitivity_weight', 1.0)
        quality_signals.append(advanced_score * 1.2)  # 高级内容加权
        
        # 5. 引用密度
        citation_count = structure_info.get('citation_count', 0)
        citation_score = min(citation_count / 50.0, 1.0)
        quality_signals.append(citation_score * 0.8)
        
        # 综合结构质量分 (0-1)
        weights = [1.5, 1.0, 1.0, 1.2, 0.8]
        structure_quality = sum(s * w for s, w in zip(quality_signals, weights)) / sum(weights)
        structure_quality = np.clip(structure_quality, 0, 1)
        
        # 用结构质量调整概率
        # 高质量 → 提升 O/F/M, 降低 H/S
        # 低质量 → 降低 O/F/M, 提升 H/S
        adjustment_strength = 0.25  # 调整强度
        bias = (structure_quality - 0.5) * adjustment_strength
        
        award_order = {'O': 2.0, 'F': 1.5, 'M': 0.5, 'H': -0.5, 'S': -1.5}
        
        adjusted = {}
        for award, prob in posteriors.items():
            factor = 1.0 + bias * award_order.get(award, 0)
            adjusted[award] = max(prob * factor, 1e-8)
        
        # 归一化
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}
    
    def _adjust_by_problem_fit(self, posteriors: Dict[str, float],
                                full_text: str, problem: str) -> Dict[str, float]:
        """
        根据论文内容与题目的适配度调整概率
        
        核心思路:
        - 好的论文应该高度契合所选题目的领域
        - 如果论文大量使用了该题目的特征关键词，说明切题且专业
        - 不切题的论文（如选了A题但内容像C题），获奖概率降低
        """
        profile = get_problem_profile(problem)
        preferred_keywords = profile.get('preferred_keywords', [])
        
        if not preferred_keywords or not full_text:
            return posteriors
        
        text_lower = full_text.lower()
        total_words = max(len(text_lower.split()), 1)
        
        # 计算关键词出现密度
        keyword_hits = 0
        for kw in preferred_keywords:
            keyword_hits += text_lower.count(kw.lower())
        
        # 密度归一化（每千字命中数）
        keyword_density = keyword_hits / total_words * 1000
        
        # 适配度评分 (0-1)
        # 密度 > 5 算好, > 10 算优秀
        fit_score = min(keyword_density / 10.0, 1.0)
        
        # 温和调整
        adjustment = 0.12  # 最大调整12%
        bias = (fit_score - 0.4) * adjustment
        
        award_order = {'O': 1.5, 'F': 1.0, 'M': 0.3, 'H': -0.3, 'S': -1.0}
        
        adjusted = {}
        for award, prob in posteriors.items():
            factor = 1.0 + bias * award_order.get(award, 0)
            adjusted[award] = max(prob * factor, 1e-8)
        
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}
    
    def _adjust_by_competition(self, posteriors: Dict[str, float],
                                problem: str) -> Dict[str, float]:
        """
        根据竞争强度微调
        
        热门题目（如C题）竞争更激烈，获高奖更难
        冷门题目（如D题）竞争相对温和
        """
        intensity = get_competition_intensity(problem)
        
        if abs(intensity - 1.0) < 0.05:
            return posteriors  # 接近平均，不调整
        
        # 竞争强度 > 1: 降低高奖概率
        # 竞争强度 < 1: 提升高奖概率
        adjustment = 0.08 * (1.0 - intensity)  # 竞争越强，adjustment越负
        
        adjusted = {}
        award_weights = {'O': 2.0, 'F': 1.5, 'M': 0.5, 'H': -0.3, 'S': -0.8}
        
        for award, prob in posteriors.items():
            factor = 1.0 + adjustment * award_weights.get(award, 0)
            adjusted[award] = max(prob * factor, 1e-8)
        
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}
    
    def _default_probabilities(self, problem: str = 'A', year: int = 2024) -> Dict[str, float]:
        """返回默认概率（使用题目特定先验）"""
        return get_award_prior(problem, year)
    
    def get_parameters(self) -> dict:
        """获取模型参数（用于保存）"""
        return {
            'award_distributions': self.award_distributions,
            'version': 'v2_enhanced',
        }
    
    def load_parameters(self, params: dict):
        """加载模型参数"""
        self.award_distributions = params.get('award_distributions', {})
    
    @staticmethod
    def get_award_description(probs: Dict[str, float]) -> str:
        """根据概率分布生成文字描述"""
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
        
        # 添加辅助解读
        o_f_prob = probs.get('O', 0) + probs.get('F', 0)
        m_prob = probs.get('M', 0)
        h_prob = probs.get('H', 0)
        
        if o_f_prob > 0.15:
            description += "\n  💡 有冲击O/F奖的潜力！"
        elif m_prob > 0.5:
            description += "\n  💡 很有可能获得M奖（一等奖）"
        elif h_prob > 0.4:
            description += "\n  💡 有望获得H奖（二等奖），建议加强模型深度"
        
        return description
    
    @staticmethod
    def format_probabilities(probs: Dict[str, float], problem: str = None) -> str:
        """格式化输出概率分布"""
        award_order = ['O', 'F', 'M', 'H', 'S']
        award_names = {
            'O': '特等奖  (O)',
            'F': '特等提名(F)',
            'M': '一等奖  (M)',
            'H': '二等奖  (H)',
            'S': '成功参赛(S)',
        }
        
        lines = []
        if problem:
            lines.append(f"  [基于 Problem {problem} 的获奖比例]")
        
        for award in award_order:
            if award in probs:
                prob = probs[award]
                bar_len = int(prob * 40)
                bar = "█" * bar_len
                marker = " ★" if prob == max(probs.values()) else ""
                lines.append(f"  {award_names[award]:12s} ({prob*100:5.1f}%): {bar}{marker}")
        
        return "\n".join(lines)
    
    @staticmethod
    def compute_quality_tier(probs: Dict[str, float]) -> Tuple[str, str]:
        """
        根据概率分布确定论文的质量层级
        
        返回: (层级名, emoji)
        """
        expected_rank = (
            probs.get('O', 0) * 5 + 
            probs.get('F', 0) * 4 + 
            probs.get('M', 0) * 3 + 
            probs.get('H', 0) * 2 + 
            probs.get('S', 0) * 1
        )
        
        if expected_rank >= 4.0:
            return "顶尖水平", "🌟"
        elif expected_rank >= 3.5:
            return "优秀水平", "⭐"
        elif expected_rank >= 3.0:
            return "良好水平", "✨"
        elif expected_rank >= 2.5:
            return "中等偏上", "💡"
        elif expected_rank >= 2.0:
            return "中等水平", "📝"
        else:
            return "有待提升", "📌"


# 保持向后兼容：包装旧接口
class AwardProbabilityEstimator(EnhancedAwardEstimator):
    """向后兼容的概率估计器（继承增强版）"""
    pass
