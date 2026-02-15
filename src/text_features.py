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
            'structural_features': np.ndarray,  # 结构特征 (12维) - 新增
            'feature_vector': np.ndarray  # 完整特征向量 (402维)
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
        提取结构特征（基于全文和论文结构分析）
        
        特征列表（12维）：
        1. readability_score - 可读性分数（学术性指标）
        2. vocabulary_diversity - 词汇多样性
        3. academic_phrase_density - 学术短语密度
        4. structure_completeness - 结构完整度 (0-1)
        5. formula_density - 公式密度（公式数/总词数*1000）
        6. table_count_norm - 表格数量（归一化）
        7. figure_count_norm - 图表标题数量（归一化）
        8. citation_density - 引用密度（引用数/总词数*1000）
        9. total_word_count_norm - 总字数（归一化）
        10. advanced_section_score - 高级节评分（归一化）
        11. avg_paragraph_length_norm - 平均段落长度（归一化）
        12. section_count_norm - 检测到的标准节数（归一化）
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
        
        # 5. 公式密度（每千字公式数，归一化到 0-1）
        formula_count = structure.get('formula_count', 0)
        formula_density = min(formula_count / total_words * 1000 / 20.0, 1.0)
        features.append(formula_density)
        
        # 6. 表格数量（归一化，假设最多 20 个）
        table_count = structure.get('table_count', 0)
        features.append(min(table_count / 20.0, 1.0))
        
        # 7. 图表标题数量（归一化，假设最多 30 个）
        figure_count = structure.get('figure_caption_count', 0)
        features.append(min(figure_count / 30.0, 1.0))
        
        # 8. 引用密度（每千字引用数，归一化到 0-1）
        citation_count = structure.get('citation_count', 0)
        citation_density = min(citation_count / total_words * 1000 / 15.0, 1.0)
        features.append(citation_density)
        
        # 9. 总字数（归一化，假设范围 1000-15000）
        word_count_norm = min(max((total_words - 1000) / 14000.0, 0.0), 1.0)
        features.append(word_count_norm)
        
        # 10. 高级节评分（最多4个高级节）
        advanced_count = structure.get('advanced_section_count', 0)
        features.append(min(advanced_count / 4.0, 1.0))
        
        # 11. 平均段落长度（归一化，假设范围 0-200 词）
        avg_para_len = structure.get('avg_paragraph_length', 0)
        features.append(min(avg_para_len / 200.0, 1.0))
        
        # 12. 检测到的标准节数（归一化，最多 6 个核心节）
        section_count = structure.get('section_count', 0)
        features.append(min(section_count / 6.0, 1.0))
        
        return np.array(features, dtype=np.float32)
    
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
            "formula_density",
            "table_count_norm",
            "figure_count_norm",
            "citation_density",
            "total_word_count_norm",
            "advanced_section_score",
            "avg_paragraph_length_norm",
            "section_count_norm"
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
