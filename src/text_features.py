"""
文本特征提取模块
从摘要文本中提取语义特征和统计特征
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config, 
    count_sentences, 
    calculate_avg_sentence_length,
    has_numerical_results,
    calculate_technical_term_density,
    check_abstract_structure,
    calculate_readability_score,
    calculate_vocabulary_diversity,
    count_academic_phrases
)


class TextFeatureExtractor:
    """文本特征提取器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化提取器"""
        self.config = load_config(config_path)
        self.text_config = self.config['text_features']
        
        # 加载预训练模型
        print("正在加载文本嵌入模型...")
        self.model = SentenceTransformer(self.text_config['model_name'])
        self.embedding_dim = self.text_config['embedding_dim']
        print(f"模型加载完成: {self.text_config['model_name']}")
    
    def extract(self, abstract: str, full_text: str = None, structure: dict = None) -> Dict:
        """
        提取文本特征
        
        参数：
            abstract: 摘要文本
            full_text: 全文文本（可选，用于提取更丰富的特征）
            structure: 论文结构分析字典（可选）
        
        返回：
        {
            'semantic_features': np.ndarray,  # 语义向量 (384维)
            'statistical_features': np.ndarray,  # 统计特征 (6维)
            'structural_features': np.ndarray,  # 结构+质量特征 (18维)
            'feature_vector': np.ndarray  # 完整特征向量 (408维)
        }
        """
        # 截断过长文本
        max_length = self.text_config['max_abstract_length']
        if len(abstract) > max_length:
            abstract = abstract[:max_length]
        
        # 1. 语义特征（深度学习）
        semantic_features = self._extract_semantic_features(abstract)
        
        # 2. 统计特征（手工特征，基于摘要）
        statistical_features = self._extract_statistical_features(abstract)
        
        # 3. 结构特征（基于全文和结构分析）
        structural_features = self._extract_structural_features(
            abstract, full_text or abstract, structure or {}
        )
        
        # 合并特征
        feature_vector = np.concatenate([semantic_features, statistical_features, structural_features])
        
        return {
            'semantic_features': semantic_features,
            'statistical_features': statistical_features,
            'structural_features': structural_features,
            'feature_vector': feature_vector,
            'feature_dim': len(feature_vector)
        }
    
    def _extract_semantic_features(self, text: str) -> np.ndarray:
        """
        提取语义特征
        使用 sentence-transformers 的预训练模型
        """
        if not text or len(text.strip()) < 10:
            # 空文本返回零向量
            return np.zeros(self.embedding_dim)
        
        # 生成嵌入向量
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        return embedding
    
    def _extract_statistical_features(self, text: str) -> np.ndarray:
        """
        提取统计特征（归一化版本）
        
        特征列表：
        1. word_count_norm - 单词数（归一化到0-1，假设范围0-500）
        2. sentence_count_norm - 句子数（归一化到0-1，假设范围0-50）
        3. avg_sentence_length_norm - 平均句长（归一化到0-1，假设范围0-50）
        4. has_results - 是否包含数字结果 (0/1)
        5. technical_term_density - 技术术语密度 (0-1)
        6. structure_score - 结构完整性评分 (0-1)
        """
        features = []
        
        # 1. 单词数（归一化）
        word_count = len(text.split())
        word_count_norm = min(word_count / 500.0, 1.0)
        features.append(word_count_norm)
        
        # 2. 句子数（归一化）
        sentence_count = count_sentences(text)
        sentence_count_norm = min(sentence_count / 50.0, 1.0)
        features.append(sentence_count_norm)
        
        # 3. 平均句长（归一化）
        avg_sent_len = calculate_avg_sentence_length(text)
        avg_sent_len_norm = min(avg_sent_len / 50.0, 1.0)
        features.append(avg_sent_len_norm)
        
        # 4. 是否包含数字结果
        has_results = 1.0 if has_numerical_results(text) else 0.0
        features.append(has_results)
        
        # 5. 技术术语密度
        term_density = calculate_technical_term_density(text)
        features.append(term_density)
        
        # 6. 结构完整性
        structure = check_abstract_structure(text)
        features.append(structure)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_structural_features(self, abstract: str, full_text: str, structure: dict) -> np.ndarray:
        """
        提取结构特征（基于全文和论文结构分析，18维）

        特征列表（18维）：
        1. readability_score - 可读性分数（学术性指标）
        2. vocabulary_diversity - 词汇多样性
        3. academic_phrase_density - 学术短语密度
        4. structure_completeness - 结构完整度 (0-1)
        5. formula_quality - 公式密度质量分（最优区间评分，非单调递增）
        6. table_quality - 表格数量质量分（最优区间评分）
        7. figure_quality - 图表数量质量分（最优区间评分）
        8. citation_quality - 引用密度质量分（最优区间评分）
        9. word_count_quality - 总字数质量分（最优区间评分）
        10. advanced_quality_score - 高级+质量内容综合评分
        11. paragraph_quality - 段落长度质量分（最优区间评分）
        12. section_count_norm - 标准节数（归一化）
        --- 新增质量信号 ---
        13. has_assumption_justification - 是否有假设合理性论证
        14. has_model_comparison - 是否有模型对比
        15. has_error_analysis - 是否有误差/不确定性分析
        16. has_dimensional_analysis - 是否有量纲/归一化讨论
        17. logical_coherence_score - 段落间语义连贯性
        18. paragraph_variety_score - 段落长度多样性（结构丰富度）
        """
        features = []

        # 1. 可读性分数（基于摘要，衡量学术写作水平）
        readability = calculate_readability_score(abstract)
        features.append(readability)

        # 2. 词汇多样性（基于全文前5000词）
        text_sample = ' '.join(full_text.split()[:5000])
        vocab_diversity = calculate_vocabulary_diversity(text_sample)
        features.append(vocab_diversity)

        # 3. 学术短语密度（基于全文）
        academic_density = count_academic_phrases(full_text)
        features.append(academic_density)

        # 以下特征基于结构分析结果
        total_words = max(structure.get('total_word_count', len(full_text.split())), 1)

        # 4. 结构完整度
        features.append(structure.get('structure_completeness', 0.5))

        # 5. 公式密度质量分（最优区间：每千字 10-25 个公式）
        formula_count = structure.get('formula_count', 0)
        formula_per_k = formula_count / total_words * 1000
        formula_quality = self._gaussian_optimal(formula_per_k, optimal=18, spread=12)
        features.append(formula_quality)

        # 6. 表格数量质量分（最优区间：4-10 个表格）
        table_count = structure.get('table_count', 0)
        table_quality = self._gaussian_optimal(table_count, optimal=6, spread=5)
        features.append(table_quality)

        # 7. 图表数量质量分（最优区间：10-25 个图表）
        figure_count = structure.get('figure_caption_count', 0)
        figure_quality = self._gaussian_optimal(figure_count, optimal=15, spread=10)
        features.append(figure_quality)

        # 8. 引用密度质量分（最优区间：每千字 8-20 处引用）
        citation_count = structure.get('citation_count', 0)
        citation_per_k = citation_count / total_words * 1000
        citation_quality = self._gaussian_optimal(citation_per_k, optimal=12, spread=8)
        features.append(citation_quality)

        # 9. 总字数量分（最优区间：8000-16000 词）
        word_quality = self._gaussian_optimal(total_words, optimal=12000, spread=5000)
        features.append(word_quality)

        # 10. 高级+质量内容综合评分
        advanced_count = structure.get('advanced_section_count', 0)
        quality_count = structure.get('quality_section_count', 0)
        advanced_quality_score = min((advanced_count * 0.7 + quality_count * 1.0) / 6.0, 1.0)
        features.append(advanced_quality_score)

        # 11. 段落长度质量分（最优区间：80-160 词/段）
        avg_para_len = structure.get('avg_paragraph_length', 0)
        para_quality = self._gaussian_optimal(avg_para_len, optimal=120, spread=60)
        features.append(para_quality)

        # 12. 检测到的标准节数（归一化，最多 6 个核心节）
        section_count = structure.get('section_count', 0)
        features.append(min(section_count / 6.0, 1.0))

        # --- 新增质量信号（13-18） ---
        # 13. 假设合理性论证
        features.append(1.0 if structure.get('has_assumption_justification', False) else 0.0)

        # 14. 模型对比
        features.append(1.0 if structure.get('has_model_comparison', False) else 0.0)

        # 15. 误差/不确定性分析
        features.append(1.0 if structure.get('has_error_analysis', False) else 0.0)

        # 16. 量纲/归一化讨论
        features.append(1.0 if structure.get('has_dimensional_analysis', False) else 0.0)

        # 17. 逻辑连贯性（段落间语义相似度的均值和方差）
        coherence = self._compute_coherence(full_text)
        features.append(coherence)

        # 18. 段落长度多样性（标准差/均值，高值=结构丰富）
        paragraphs_text = [p.strip() for p in full_text.split('\n\n') if p.strip() and len(p.strip()) > 30]
        if len(paragraphs_text) > 3:
            para_lens = [len(p.split()) for p in paragraphs_text]
            para_mean = np.mean(para_lens)
            para_std = np.std(para_lens)
            variety = min(para_std / max(para_mean, 1), 1.0)
        else:
            variety = 0.0
        features.append(variety)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _gaussian_optimal(value: float, optimal: float, spread: float) -> float:
        """
        高斯最优区间评分：值越接近 optimal 得分越高，偏离任一侧都降分。
        解决了原先「越多越好」的单调归一化问题。
        """
        z = (value - optimal) / max(spread, 1.0)
        return float(np.exp(-0.5 * z * z))

    def _compute_coherence(self, full_text: str) -> float:
        """
        计算段落间语义连贯性。

        对相邻段落用 sentence-transformer 提取嵌入，计算 cosine similarity。
        连贯的论文相邻段落语义相似度高且稳定。
        """
        try:
            paragraphs = [p.strip() for p in full_text.split('\n\n')
                          if p.strip() and len(p.strip()) > 100]
            if len(paragraphs) < 3:
                return 0.5

            # 取前 8 段，控制计算量
            sample_paras = paragraphs[:8]
            embeddings = self.model.encode(sample_paras, convert_to_numpy=True)

            similarities = []
            for i in range(len(embeddings) - 1):
                sim = float(np.dot(embeddings[i], embeddings[i + 1]) / (
                    max(np.linalg.norm(embeddings[i]), 1e-8) *
                    max(np.linalg.norm(embeddings[i + 1]), 1e-8)
                ))
                similarities.append(sim)

            if not similarities:
                return 0.5

            avg_sim = np.mean(similarities)
            # 连贯性 = 高平均相似度 + 低方差（稳定过渡）
            std_sim = np.std(similarities) if len(similarities) > 1 else 0.0
            coherence = avg_sim * (1.0 - std_sim)
            return float(np.clip(coherence, 0.0, 1.0))
        except Exception:
            return 0.5
    
    def batch_extract(self, abstracts: List[str]) -> np.ndarray:
        """
        批量提取特征
        
        参数：
            abstracts: 摘要文本列表
        
        返回：
            特征矩阵 (n_samples, feature_dim)
        """
        print(f"正在提取 {len(abstracts)} 篇论文的文本特征...")
        
        features = []
        for i, abstract in enumerate(abstracts):
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(abstracts)}")
            
            result = self.extract(abstract)
            features.append(result['feature_vector'])
        
        feature_matrix = np.vstack(features)
        print(f"文本特征提取完成: {feature_matrix.shape}")
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        semantic_names = [f"semantic_{i}" for i in range(self.embedding_dim)]
        statistical_names = [
            "word_count_norm",
            "sentence_count_norm",
            "avg_sentence_length_norm",
            "has_numerical_results",
            "technical_term_density",
            "structure_score"
        ]
        structural_names = [
            "readability_score",
            "vocabulary_diversity",
            "academic_phrase_density",
            "structure_completeness",
            "formula_quality",
            "table_quality",
            "figure_quality",
            "citation_quality",
            "word_count_quality",
            "advanced_quality_score",
            "paragraph_quality",
            "section_count_norm",
            # 新增质量信号
            "has_assumption_justification",
            "has_model_comparison",
            "has_error_analysis",
            "has_dimensional_analysis",
            "logical_coherence_score",
            "paragraph_variety_score",
        ]
        
        return semantic_names + statistical_names + structural_names


def extract_text_features(text: str, config_path: str = "config.yaml") -> np.ndarray:
    """
    便捷函数：提取文本特征
    
    参数:
        text: 文本内容（通常是摘要）
        config_path: 配置文件路径
    
    返回:
        numpy.ndarray: 特征向量
    """
    extractor = TextFeatureExtractor(config_path)
    result = extractor.extract(text)
    return result['feature_vector']


def main():
    """测试文本特征提取器"""
    # 示例摘要
    sample_abstract = """
    Climate change is one of the most pressing issues facing humanity today. 
    In this paper, we develop a comprehensive mathematical model to predict 
    temperature changes over the next 50 years. Our model incorporates 
    multiple factors including CO2 emissions, deforestation rates, and 
    ocean temperature variations. Using historical data from 1970-2020, 
    we train a neural network that achieves 94.3% accuracy on validation data. 
    The results show that global temperatures will rise by 2.1°C by 2070 
    under current emission scenarios. We conclude that immediate action 
    is necessary to mitigate climate change impacts.
    """
    
    extractor = TextFeatureExtractor()
    
    print("\n" + "="*60)
    print("文本特征提取测试")
    print("="*60)
    
    result = extractor.extract(sample_abstract)
    
    print(f"\n语义特征维度: {result['semantic_features'].shape}")
    print(f"语义特征前5维: {result['semantic_features'][:5]}")
    
    print(f"\n统计特征维度: {result['statistical_features'].shape}")
    print(f"统计特征: {result['statistical_features']}")
    
    feature_names = extractor.get_feature_names()
    stat_features = result['statistical_features']
    
    print("\n统计特征详情:")
    for name, value in zip(feature_names[-6:], stat_features):
        print(f"  {name}: {value:.3f}")
    
    print(f"\n完整特征向量维度: {result['feature_vector'].shape}")


if __name__ == "__main__":
    main()
