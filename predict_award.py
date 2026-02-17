"""
统一奖项概率预测脚本 (v2)

输入: PDF 文件路径
输出: 各奖项 (O/F/M/H/S) 的概率分布

核心改进:
1. 自动识别题目 (A-F) 和年份
2. 使用题目特定的获奖比例先验
3. 多维质量信号融合 (相似度 + 结构 + 题目适配 + 竞争强度)
4. 无需 H/S/F 样本，通过贝叶斯推断实现全奖项概率估计

使用方法:
    python predict_award.py <pdf文件路径> [--problem A] [--year 2024]
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import argparse
import logging
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pdf_parser import PDFParser
from src.text_features import TextFeatureExtractor
from src.image_features import ImageFeatureExtractor
from src.feature_fusion import fuse_features
from src.probability_model_v2 import EnhancedAwardEstimator
from src.problem_detector import ProblemDetector, detect_problem
from src.award_prior import get_award_prior, get_problem_profile, list_available_data

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class AwardPredictor:
    """
    MCM/ICM 论文获奖概率预测器 (v2)
    
    输入 PDF → 输出 O/F/M/H/S 各奖项概率
    
    核心流程:
    1. 解析 PDF → 提取摘要、图片、全文、结构信息
    2. 自动识别题目类型 (A-F) 和年份
    3. 提取特征向量 (文本语义 + 图像 + 元数据)
    4. 计算与 O 奖质心的相似度
    5. 基于贝叶斯推断估计各奖项概率
    6. 用结构质量、题目适配度等信号修正概率
    """
    
    def __init__(self, model_path: str = "models/scoring_model.pkl"):
        """
        初始化预测器
        
        参数:
            model_path: 训练好的模型路径
        """
        logger.info("初始化 MCM/ICM 获奖概率预测器 v2...")
        
        # 加载模型
        self.model_data = self._load_model(model_path)
        if self.model_data is None:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 初始化各模块
        self.pdf_parser = PDFParser()
        self.text_extractor = TextFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        self.problem_detector = ProblemDetector()
        
        # 初始化概率估计器
        prob_params = self.model_data.get('prob_model_params')
        if prob_params and 'award_distributions' in prob_params:
            self.prob_estimator = EnhancedAwardEstimator()
            self.prob_estimator.load_parameters(prob_params)
        else:
            # 使用训练集的相似度重新拟合
            stats = self.model_data['stats']
            # 从stats中模拟O奖相似度分布
            sim_mean = stats['similarity_mean']
            sim_std = stats['similarity_std']
            fake_sims = np.random.normal(sim_mean, sim_std, 50)
            self.prob_estimator = EnhancedAwardEstimator(fake_sims, stats)
        
        n_papers = self.model_data.get('n_papers', '?')
        logger.info(f"预测器初始化完成（基于 {n_papers} 篇 O 奖论文训练）\n")
    
    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            logger.error("请先运行: python scripts/train_scoring_model.py")
            return None
        
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def predict(self, pdf_path: str, problem: str = None, 
                year: int = None, verbose: bool = True) -> dict:
        """
        预测单篇论文的各奖项概率
        
        参数:
            pdf_path: PDF 文件路径
            problem: 题目 (A-F)，None 则自动检测
            year: 年份，None 则自动检测
            verbose: 是否打印详细信息
        
        返回:
            {
                'probabilities': {'O': 0.01, 'F': 0.03, 'M': 0.45, ...},
                'score': 78.5,
                'aspect_scores': {'abstract': 72, 'figures': 65, 'modeling': 80},
                'aspect_details': {...},
                'similarity': 0.823,
                'problem': 'A',
                'contest': 'MCM',
                'year': 2024,
                'quality_tier': '良好水平',
                'description': '...',
                'metadata': {...},
                'success': True,
            }
        """
        if not os.path.exists(pdf_path):
            return {'error': f'文件不存在: {pdf_path}', 'success': False}
        
        try:
            # ==================== 步骤1: 解析 PDF ====================
            if verbose:
                logger.info(f"📄 论文: {os.path.basename(pdf_path)}")
                logger.info("=" * 60)
                logger.info("  [1/7] 解析 PDF...")
            
            parsed = self.pdf_parser.parse(pdf_path)
            
            if not parsed.get('success', False):
                return {'error': 'PDF 解析失败', 'success': False}
            
            abstract = parsed['abstract']
            full_text = parsed.get('full_text', '')
            images = parsed['images']
            structure = parsed.get('structure', {})
            metadata = parsed.get('metadata', {})
            
            if verbose:
                logger.info(f"    ✓ 摘要: {len(abstract)} 字符")
                logger.info(f"    ✓ 全文: {len(full_text)} 字符")
                logger.info(f"    ✓ 图片: {len(images)} 张")
                logger.info(f"    ✓ 页数: {metadata.get('page_count', '?')} 页")
            
            # ==================== 步骤2: 自动识别题目 ====================
            if verbose:
                logger.info("  [2/7] 自动识别题目...")
            
            if not problem or not year:
                detection = self.problem_detector.detect(full_text, pdf_path, year)
                
                if not problem:
                    problem = detection.get('problem', 'A')
                if not year:
                    year = detection.get('year') or self.problem_detector.detect_year(full_text, pdf_path) or 2024
                
                contest = detection.get('contest', 'MCM' if problem in 'ABC' else 'ICM')
                confidence = detection.get('confidence', 0.5)
                
                if verbose:
                    logger.info(f"    ✓ 题目: Problem {problem} ({contest})")
                    logger.info(f"    ✓ 年份: {year}")
                    logger.info(f"    ✓ 检测置信度: {confidence:.0%}")
            else:
                contest = 'MCM' if problem.upper() in 'ABC' else 'ICM'
            
            problem = problem.upper()
            
            # ==================== 步骤3: 提取特征 ====================
            if verbose:
                logger.info("  [3/7] 提取文本特征...")
            
            text_result = self.text_extractor.extract(
                abstract, full_text=full_text, structure=structure
            )
            text_feat = text_result['feature_vector']
            
            if verbose:
                logger.info("  [4/7] 提取图像特征...")
            
            image_result = self.image_extractor.extract(images)
            image_feat = image_result['feature_vector']
            
            # ==================== 步骤4: 融合特征 & 计算相似度 ====================
            if verbose:
                logger.info("  [5/7] 特征融合...")
            
            page_count = metadata.get('page_count', 20)
            ref_count = metadata.get('ref_count', 15)
            
            fused = fuse_features(
                text_features=text_feat,
                image_features=image_feat,
                year=year,
                contest=contest,
                problem=problem,
                page_count=page_count,
                ref_count=ref_count,
            )
            
            # 计算与 O 奖质心的余弦相似度
            centroid = self.model_data['centroid']
            stats = self.model_data['stats']
            
            fused, centroid = self._align_dimensions(fused, centroid)
            
            similarity = cosine_similarity(
                fused.reshape(1, -1), centroid.reshape(1, -1)
            )[0, 0]
            
            score = self._compute_score(similarity, stats)
            
            # ==================== 步骤5: 三维度子分 ====================
            if verbose:
                logger.info("  [6/7] 三维度对比评分...")
            
            aspect_scores, aspect_details = self._compute_aspect_scores(
                text_result, image_result, structure, images,
                abstract, full_text, page_count, ref_count,
            )
            
            # ==================== 步骤6: 贝叶斯概率估计 ====================
            if verbose:
                logger.info("  [7/7] 贝叶斯概率估计...")
            
            probabilities = self.prob_estimator.estimate_probabilities(
                similarity=similarity,
                problem=problem,
                year=year,
                score=score,
                structure_info=structure,
                full_text=full_text,
                aspect_scores=aspect_scores,
            )
            
            quality_tier, emoji = EnhancedAwardEstimator.compute_quality_tier(probabilities)
            description = EnhancedAwardEstimator.get_award_description(probabilities)
            
            result = {
                'probabilities': probabilities,
                'score': score,
                'aspect_scores': aspect_scores,
                'aspect_details': aspect_details,
                'similarity': float(similarity),
                'problem': problem,
                'contest': contest,
                'year': year,
                'quality_tier': quality_tier,
                'emoji': emoji,
                'description': description,
                'metadata': {
                    'abstract_length': len(abstract),
                    'full_text_length': len(full_text),
                    'image_count': len(images),
                    'page_count': page_count,
                    'ref_count': ref_count,
                    'structure': structure,
                },
                'success': True,
            }
            
            if verbose:
                self.print_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'success': False}
    
    def _align_dimensions(self, features: np.ndarray, centroid: np.ndarray):
        """自动对齐特征向量与质心的维度"""
        f_dim = features.shape[0]
        c_dim = centroid.shape[0]
        if f_dim == c_dim:
            return features, centroid
        diff = abs(f_dim - c_dim)
        if diff > 50:
            logger.warning(f"⚠️ 特征维度差异过大 ({f_dim} vs {c_dim})，建议重新训练模型")
        min_dim = min(f_dim, c_dim)
        return features[:min_dim], centroid[:min_dim]
    
    # ------------------------------------------------------------------
    #  三维度对比评分 (摘要 / 图表 / 建模)
    # ------------------------------------------------------------------
    def _compute_aspect_scores(
        self, text_result, image_result, structure, images,
        abstract, full_text, page_count, ref_count,
    ):
        """
        计算三个维度的子分 (0-100)，并生成与 O 奖论文的文字差异说明。

        维度:
          1. 摘要写作 (abstract)  — 语义嵌入相似度 + 写作统计
          2. 图表水平 (figures)   — 图像特征相似度 + 数量/质量
          3. 建模深度 (modeling)  — 结构特征相似度 + 公式/引用/灵敏度

        返回:
          aspect_scores: {'abstract': 72, 'figures': 65, 'modeling': 80}
          aspect_details: {'abstract': {...detail dict...}, ...}
        """
        aspect_stats = self.model_data.get('aspect_stats', {})
        scores = {}
        details = {}

        # ---------- 1. 摘要维度 ----------
        abs_stat = aspect_stats.get('abstract', {})
        abs_centroid = abs_stat.get('centroid')
        sem_feat = text_result.get('semantic_features')

        if abs_centroid is not None and sem_feat is not None:
            abs_centroid_arr = np.array(abs_centroid)
            min_d = min(len(sem_feat), len(abs_centroid_arr))
            sim = float(cosine_similarity(
                sem_feat[:min_d].reshape(1, -1),
                abs_centroid_arr[:min_d].reshape(1, -1)
            )[0, 0])
            abs_score = self._sim_to_score(
                sim, abs_stat.get('sim_mean', 0.7), abs_stat.get('sim_std', 0.1),
                abs_stat.get('sim_min', 0.3), abs_stat.get('sim_max', 1.0),
            )
        else:
            abs_score = 60.0
            sim = 0.0

        # 统计对比
        o_avg_len = abs_stat.get('avg_length', 1200)
        abs_len = len(abstract)
        len_ratio = abs_len / max(o_avg_len, 1)

        abs_diff = []
        if len_ratio < 0.5:
            abs_diff.append(f"摘要长度偏短 ({abs_len}字符 vs O奖平均{o_avg_len:.0f}字符)")
            abs_score *= 0.9
        elif len_ratio > 2.0:
            abs_diff.append(f"摘要偏长 ({abs_len}字符 vs O奖平均{o_avg_len:.0f}字符)")
        else:
            abs_diff.append(f"摘要长度合理 ({abs_len}字符, O奖平均{o_avg_len:.0f}字符)")

        stat_feat = text_result.get('statistical_features', np.zeros(6))
        if stat_feat[3] < 0.5:
            abs_diff.append("摘要缺少定量结果(数字/百分比)，O奖论文摘要通常包含关键数据")
        if stat_feat[4] < 0.15:
            abs_diff.append("技术术语密度偏低，建议使用更专业的学术表达")

        abs_score = float(np.clip(abs_score, 0, 100))
        scores['abstract'] = round(abs_score, 1)
        details['abstract'] = {
            'score': round(abs_score, 1),
            'similarity': round(sim, 4),
            'differences': abs_diff,
        }

        # ---------- 2. 图表维度 ----------
        fig_stat = aspect_stats.get('figures', {})
        fig_centroid = fig_stat.get('centroid')
        img_feat = image_result.get('feature_vector')

        if fig_centroid is not None and img_feat is not None:
            fig_centroid_arr = np.array(fig_centroid)
            min_d = min(len(img_feat), len(fig_centroid_arr))
            fig_sim = float(cosine_similarity(
                img_feat[:min_d].reshape(1, -1),
                fig_centroid_arr[:min_d].reshape(1, -1)
            )[0, 0])
            fig_score = self._sim_to_score(
                fig_sim, fig_stat.get('sim_mean', 0.5), fig_stat.get('sim_std', 0.15),
                fig_stat.get('sim_min', 0.1), fig_stat.get('sim_max', 0.9),
            )
        else:
            fig_score = 60.0
            fig_sim = 0.0

        o_avg_imgs = fig_stat.get('avg_image_count', 20)
        n_imgs = len(images)
        fig_captions = structure.get('figure_caption_count', 0)
        table_count = structure.get('table_count', 0)

        fig_diff = []
        if n_imgs < o_avg_imgs * 0.4:
            fig_diff.append(f"图表数量偏少 ({n_imgs}张 vs O奖平均{o_avg_imgs:.0f}张)，应增加可视化")
            fig_score *= 0.85
        elif n_imgs > o_avg_imgs * 1.5:
            fig_diff.append(f"图表数量充足 ({n_imgs}张 vs O奖平均{o_avg_imgs:.0f}张)")
        else:
            fig_diff.append(f"图表数量合理 ({n_imgs}张, O奖平均{o_avg_imgs:.0f}张)")

        if fig_captions < 5:
            fig_diff.append(f"图表标题数仅{fig_captions}个，O奖通常有10+个清晰的图表标题")
        if table_count == 0:
            fig_diff.append("未检测到表格，O奖论文通常有数据对比表格")
        
        # 新增：图片画风/质量分析
        img_stats = image_result.get('statistical_features', np.zeros(18))
        if len(img_stats) >= 18 and n_imgs > 0:
            chart_ratio = img_stats[6]
            professional = img_stats[13]
            high_res_ratio = img_stats[17]
            color_ratio = img_stats[9]
            
            if chart_ratio < 0.3:
                fig_diff.append(f"图表类图片比例偏低({chart_ratio:.0%})，照片/装饰图较多，建议增加专业图表")
                fig_score *= 0.92
            elif chart_ratio > 0.6:
                fig_diff.append(f"图表类图片占比{chart_ratio:.0%}，图表画风专业")
            
            if professional < 0.3:
                fig_diff.append("图片整体专业度偏低，建议使用矢量图/高对比度配色")
                fig_score *= 0.90
            elif professional > 0.6:
                fig_diff.append("图片专业度较高（对比度、排版一致性好）")
            
            if high_res_ratio < 0.5:
                fig_diff.append(f"高分辨率图片仅占{high_res_ratio:.0%}，建议提升图表清晰度")
            
            if color_ratio > 0.7:
                fig_diff.append("大部分为彩色图表，视觉效果好")

        fig_score = float(np.clip(fig_score, 0, 100))
        scores['figures'] = round(fig_score, 1)
        details['figures'] = {
            'score': round(fig_score, 1),
            'similarity': round(fig_sim, 4),
            'differences': fig_diff,
        }

        # ---------- 3. 建模深度维度 ----------
        mod_stat = aspect_stats.get('modeling', {})
        mod_centroid = mod_stat.get('centroid')
        struct_feat = text_result.get('structural_features')

        if mod_centroid is not None and struct_feat is not None:
            mod_centroid_arr = np.array(mod_centroid)
            min_d = min(len(struct_feat), len(mod_centroid_arr))
            mod_sim = float(cosine_similarity(
                struct_feat[:min_d].reshape(1, -1),
                mod_centroid_arr[:min_d].reshape(1, -1)
            )[0, 0])
            mod_score = self._sim_to_score(
                mod_sim, mod_stat.get('sim_mean', 0.5), mod_stat.get('sim_std', 0.15),
                mod_stat.get('sim_min', 0.1), mod_stat.get('sim_max', 0.9),
            )
        else:
            mod_score = 60.0
            mod_sim = 0.0

        o_avg_formula = mod_stat.get('avg_formula_count', 20)
        o_avg_cite = mod_stat.get('avg_citation_count', 30)
        o_avg_complete = mod_stat.get('avg_completeness', 0.8)
        o_avg_adv = mod_stat.get('avg_advanced_sections', 2)
        o_avg_pages = mod_stat.get('avg_page_count', 22)

        formula_count = structure.get('formula_count', 0)
        citation_count = structure.get('citation_count', 0)
        completeness = structure.get('structure_completeness', 0)
        adv_count = structure.get('advanced_section_count', 0)

        mod_diff = []
        if formula_count < o_avg_formula * 0.5:
            mod_diff.append(f"公式数量偏少 ({formula_count}个 vs O奖平均{o_avg_formula:.0f}个)，建模深度不足")
            mod_score *= 0.9
        else:
            mod_diff.append(f"公式数量{'充足' if formula_count>=o_avg_formula else '合理'} ({formula_count}个, O奖平均{o_avg_formula:.0f}个)")

        if citation_count < o_avg_cite * 0.4:
            mod_diff.append(f"引用偏少 ({citation_count}处 vs O奖平均{o_avg_cite:.0f}处)，文献综述可加强")
        if completeness < o_avg_complete - 0.15:
            mod_diff.append(f"结构完整度偏低 ({completeness:.0%} vs O奖平均{o_avg_complete:.0%})")

        has_sens = structure.get('has_sensitivity_analysis', False)
        has_val = structure.get('has_model_validation', False)
        if not has_sens:
            mod_diff.append("缺少灵敏度分析，O奖论文通常包含 Sensitivity Analysis")
        if not has_val:
            mod_diff.append("缺少模型验证，建议加入 Model Validation/Testing 部分")
        if adv_count < o_avg_adv * 0.5:
            mod_diff.append(f"高级建模内容偏少 ({adv_count}节 vs O奖平均{o_avg_adv:.0f}节)")

        if page_count < o_avg_pages * 0.6:
            mod_diff.append(f"论文页数偏少 ({page_count}页 vs O奖平均{o_avg_pages:.0f}页)")

        mod_score = float(np.clip(mod_score, 0, 100))
        scores['modeling'] = round(mod_score, 1)
        details['modeling'] = {
            'score': round(mod_score, 1),
            'similarity': round(mod_sim, 4),
            'differences': mod_diff,
        }

        return scores, details
    
    @staticmethod
    def _sim_to_score(sim, mean, std, smin, smax):
        """相似度 → 0-100 分"""
        std = max(std, 0.01)
        if sim >= smax:
            return 100.0
        elif sim >= mean:
            return 85 + 15 * (sim - mean) / max(smax - mean, 1e-6)
        elif sim >= mean - 2 * std:
            return 50 + 35 * (sim - (mean - 2 * std)) / max(2 * std, 1e-6)
        else:
            thr = mean - 2 * std
            return max(0, 50 * (sim - smin) / max(thr - smin, 1e-6))
    
    def _compute_score(self, similarity: float, stats: dict) -> float:
        """将余弦相似度映射为 0-100 分数"""
        sim_mean = stats['similarity_mean']
        sim_std = stats['similarity_std']
        sim_min = stats['similarity_min']
        sim_max = stats['similarity_max']
        
        if similarity >= sim_max:
            score = 100.0
        elif similarity >= sim_mean:
            score = 85 + 15 * (similarity - sim_mean) / max(sim_max - sim_mean, 1e-6)
        elif similarity >= sim_mean - 2 * sim_std:
            score = 50 + 35 * (similarity - (sim_mean - 2 * sim_std)) / max(2 * sim_std, 1e-6)
        else:
            threshold = sim_mean - 2 * sim_std
            if similarity > sim_min:
                score = 50 * (similarity - sim_min) / max(threshold - sim_min, 1e-6)
            else:
                score = max(0.0, 50 * (similarity - sim_min) / max(threshold - sim_min, 1e-6))
        
        return float(np.clip(score, 0, 100))
    
    def print_result(self, result: dict):
        """格式化打印预测结果（含三维度子分）"""
        if not result.get('success', False):
            logger.info(f"\n❌ 预测失败: {result.get('error', 'Unknown')}")
            return
        
        probs = result['probabilities']
        problem = result['problem']
        contest = result['contest']
        year = result['year']
        score = result['score']
        similarity = result['similarity']
        quality_tier = result['quality_tier']
        emoji_tier = result['emoji']
        aspect_scores = result.get('aspect_scores', {})
        aspect_details = result.get('aspect_details', {})
        
        logger.info(f"\n{'='*64}")
        logger.info(f"  MCM/ICM 获奖概率预测结果 (v2)")
        logger.info(f"{'='*64}")
        
        logger.info(f"\n  题目: {contest} Problem {problem} ({year})")
        profile = get_problem_profile(problem)
        logger.info(f"  类型: {profile['name']}")
        
        # 先验信息
        prior = get_award_prior(problem, year)
        logger.info(f"\n  📊 该题历史获奖比例:")
        logger.info(f"     O: {prior['O']*100:.2f}% | F: {prior['F']*100:.2f}% | "
                     f"M: {prior['M']*100:.1f}% | H: {prior['H']*100:.1f}% | S: {prior['S']*100:.1f}%")
        
        logger.info(f"\n  {emoji_tier} 综合评分: {score:.1f} / 100 ({quality_tier})")
        logger.info(f"  余弦相似度: {similarity:.4f}")
        
        stats = self.model_data['stats']
        logger.info(f"  O奖参考: 均值={stats['similarity_mean']:.4f}, "
                     f"范围=[{stats['similarity_min']:.4f}, {stats['similarity_max']:.4f}]")
        
        # ==================== 三维度子分 ====================
        logger.info(f"\n{'─'*64}")
        logger.info(f"  📐 三维度对比评分 (与 O 奖论文对比)")
        logger.info(f"{'─'*64}")
        
        aspect_names = {
            'abstract': ('📝 摘要写作', '摘要语义与O奖的接近程度、写作质量'),
            'figures':  ('📊 图表水平', '图表特征与O奖的接近程度、数量质量'),
            'modeling': ('🔬 建模深度', '结构特征与O奖的接近程度、公式引用等'),
        }
        
        for key in ['abstract', 'figures', 'modeling']:
            name, desc = aspect_names[key]
            s = aspect_scores.get(key, 0)
            detail = aspect_details.get(key, {})
            diffs = detail.get('differences', [])
            sim_val = detail.get('similarity', 0)
            
            # 分数条
            bar_full = 20
            bar_filled = int(s / 100 * bar_full)
            bar = "█" * bar_filled + "░" * (bar_full - bar_filled)
            
            logger.info(f"\n  {name}:  {s:.0f} / 100")
            logger.info(f"    [{bar}] (相似度={sim_val:.4f})")
            
            for d in diffs:
                logger.info(f"    • {d}")
        
        # ==================== 概率分布 ====================
        logger.info(f"\n{'─'*64}")
        logger.info(f"  🎯 预测获奖概率:")
        prob_str = EnhancedAwardEstimator.format_probabilities(probs, problem)
        logger.info(prob_str)
        
        # 描述
        logger.info(f"\n  {result['description']}")
        
        # 元数据
        meta = result['metadata']
        structure = meta.get('structure', {})
        logger.info(f"\n{'─'*64}")
        logger.info(f"  📋 论文基本信息:")
        logger.info(f"     摘要: {meta['abstract_length']} 字符")
        logger.info(f"     全文: ~{meta['full_text_length']//1000}k 字符")
        logger.info(f"     图表: {meta['image_count']} 张图片 + "
                     f"{structure.get('figure_caption_count', 0)} 个图表标题 + "
                     f"{structure.get('table_count', 0)} 个表格")
        logger.info(f"     公式: {structure.get('formula_count', 0)} 个")
        logger.info(f"     引用: {structure.get('citation_count', 0)} 处引用, "
                     f"{meta['ref_count']} 条参考文献")
        logger.info(f"     页数: {meta['page_count']} 页")
        
        # 结构完整度
        completeness = structure.get('structure_completeness', 0)
        logger.info(f"\n  📐 结构完整度: {completeness:.0%}")
        section_flags = [
            ('摘要', structure.get('has_abstract', False)),
            ('引言', structure.get('has_introduction', False)),
            ('方法', structure.get('has_methodology', False)),
            ('结果', structure.get('has_results', False)),
            ('结论', structure.get('has_conclusion', False)),
            ('参考文献', structure.get('has_references', False)),
        ]
        present = [name for name, flag in section_flags if flag]
        missing = [name for name, flag in section_flags if not flag]
        logger.info(f"     ✓ 包含: {', '.join(present) if present else '无'}")
        if missing:
            logger.info(f"     ✗ 缺少: {', '.join(missing)}")
        
        advanced = []
        if structure.get('has_sensitivity_analysis'):
            advanced.append('灵敏度分析')
        if structure.get('has_model_validation'):
            advanced.append('模型验证')
        if structure.get('has_strengths_weaknesses'):
            advanced.append('优缺点分析')
        if structure.get('has_future_work'):
            advanced.append('未来工作')
        
        if advanced:
            logger.info(f"     ⭐ 高级内容: {', '.join(advanced)}")
        else:
            logger.info(f"     ⚠️ 建议添加: 灵敏度分析, 模型验证, 优缺点分析")
        
        logger.info(f"\n{'='*64}")
    
    def batch_predict(self, pdf_dir: str, problem: str = None, 
                      year: int = None) -> list:
        """批量预测目录下的所有PDF"""
        pdf_dir = Path(pdf_dir)
        
        if not pdf_dir.exists():
            logger.error(f"目录不存在: {pdf_dir}")
            return []
        
        pdf_files = list(pdf_dir.rglob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"目录中没有PDF文件")
            return []
        
        logger.info(f"\n批量预测: {len(pdf_files)} 个文件")
        logger.info("=" * 60)
        
        results = []
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"\n[{i}/{len(pdf_files)}] {pdf_file.name}")
            result = self.predict(str(pdf_file), problem, year, verbose=False)
            
            if result.get('success'):
                results.append(result)
                probs = result['probabilities']
                best = max(probs, key=probs.get)
                logger.info(f"  {result['emoji']} Score={result['score']:.1f} | "
                             f"Problem {result['problem']} | "
                             f"Best={best}({probs[best]*100:.1f}%)")
            else:
                logger.info(f"  ❌ {result.get('error', 'Failed')}")
        
        # 汇总统计
        if results:
            self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: list):
        """打印批量预测汇总"""
        logger.info(f"\n{'='*60}")
        logger.info(f"  批量预测汇总 ({len(results)} 篇)")
        logger.info(f"{'='*60}")
        
        scores = [r['score'] for r in results]
        logger.info(f"\n  评分统计:")
        logger.info(f"    平均分: {np.mean(scores):.1f}")
        logger.info(f"    中位数: {np.median(scores):.1f}")
        logger.info(f"    最高分: {np.max(scores):.1f}")
        logger.info(f"    最低分: {np.min(scores):.1f}")
        
        # 各奖项平均概率
        avg_probs = {'O': 0, 'F': 0, 'M': 0, 'H': 0, 'S': 0}
        for r in results:
            for award, prob in r['probabilities'].items():
                avg_probs[award] += prob
        avg_probs = {k: v / len(results) for k, v in avg_probs.items()}
        
        logger.info(f"\n  平均获奖概率:")
        for award in ['O', 'F', 'M', 'H', 'S']:
            prob = avg_probs[award]
            bar = "█" * int(prob * 40)
            logger.info(f"    {award}: {prob*100:5.1f}% {bar}")
        
        # 最可能获奖分布
        best_awards = [max(r['probabilities'], key=r['probabilities'].get) for r in results]
        from collections import Counter
        award_counts = Counter(best_awards)
        
        logger.info(f"\n  最可能获奖分布:")
        for award in ['O', 'F', 'M', 'H', 'S']:
            count = award_counts.get(award, 0)
            pct = count / len(results) * 100
            logger.info(f"    {award}: {count} 篇 ({pct:.0f}%)")
        
        logger.info(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="MCM/ICM 论文获奖概率预测 v2",
        epilog="示例:\n"
               "  python predict_award.py paper.pdf\n"
               "  python predict_award.py paper.pdf --problem A --year 2024\n"
               "  python predict_award.py ./papers/ --batch\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("pdf_path", help="PDF文件路径或目录路径")
    parser.add_argument("--problem", "-p", type=str, choices=list('ABCDEF'),
                        help="题目 (A-F)，不指定则自动检测")
    parser.add_argument("--year", "-y", type=int, 
                        help="年份，不指定则自动检测")
    parser.add_argument("--model", "-m", default="models/scoring_model.pkl",
                        help="模型路径 (默认: models/scoring_model.pkl)")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="批量预测目录下所有PDF")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="安静模式（减少输出）")
    parser.add_argument("--show-priors", action="store_true",
                        help="显示各题目的历史获奖比例数据")
    
    args = parser.parse_args()
    
    # 显示先验数据
    if args.show_priors:
        print(list_available_data())
        return
    
    # 初始化预测器
    try:
        predictor = AwardPredictor(model_path=args.model)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # 预测
    if args.batch or os.path.isdir(args.pdf_path):
        predictor.batch_predict(args.pdf_path, args.problem, args.year)
    else:
        predictor.predict(
            args.pdf_path, 
            problem=args.problem, 
            year=args.year,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
