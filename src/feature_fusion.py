"""
特征融合模块
将文本特征、图像特征和元数据融合为一个特征向量
"""

import numpy as np
from typing import Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config


class FeatureFusion:
    """特征融合器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化融合器"""
        self.config = load_config(config_path)
        
        # 题目编码映射
        self.problem_encoder = {
            'A': 0, 'B': 1, 'C': 2,
            'D': 3, 'E': 4, 'F': 5
        }
        
        # 赛道编码
        self.contest_encoder = {
            'MCM': 0,
            'ICM': 1
        }
    
    def _l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """L2 归一化向量，避免零向量除零"""
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            return vec / norm
        return vec
    
    def fuse(self, text_features: np.ndarray, image_features: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        融合所有特征
        
        对每个特征子组分别做 L2 归一化后加权拼接，
        确保不同维度/尺度的特征组有均等贡献。
        
        特征子组：
          1. text_semantic (384维) - 文本语义嵌入，权重 1.0 (最具区分力)
          2. text_stats (6维) - 文本统计特征，权重 0.3
          3. text_structural (12维) - 文本结构特征，权重 0.5 (新增，重要)
          4. image_deep (512维) - 图像深度特征，权重 0.8
          5. image_stats (18维) - 图像统计特征（含画风分析），权重 0.5
          6. meta (10维) - 元数据特征，权重 0.3
        
        返回：
            融合后的特征向量 (~942维)
        """
        # 获取各组维度
        text_embed_dim = self.config['text_features']['embedding_dim']  # 384
        text_stats_dim = 6   # 统计特征维度
        image_feat_dim = self.config['image_features']['feature_dim']   # 512
        
        # 分拆特征子组
        text_semantic = text_features[:text_embed_dim]      # 384
        text_stats = text_features[text_embed_dim:text_embed_dim + text_stats_dim]  # 6
        # 新增：结构特征（如果存在）
        text_structural = text_features[text_embed_dim + text_stats_dim:]  # 12 (if available)
        
        image_deep = image_features[:image_feat_dim]         # 512
        image_stats = image_features[image_feat_dim:]        # 6
        meta_features = self._extract_meta_features(metadata)  # 10
        
        # 各组权重（语义嵌入最重要，结构特征也很重要）
        groups = [
            (text_semantic, 1.0),   # 文本语义 - 最重要
            (text_stats,   0.3),   # 文本统计
        ]
        
        # 如果有结构特征（新版本），加入
        if len(text_structural) > 0:
            groups.append((text_structural, 0.5))  # 结构特征 - 较重要
        
        groups.extend([
            (image_deep,   0.8),   # 图像深度特征
            (image_stats,  0.5),   # 图像统计 - 含画风分析，权重提升
            (meta_features, 0.3),  # 元数据
        ])
        
        # L2 归一化每组 + 加权
        normalized_groups = []
        for group, weight in groups:
            normalized_groups.append(weight * self._l2_normalize(group))
        
        fused_features = np.concatenate(normalized_groups)
        return fused_features
    
    def _extract_meta_features(self, metadata: Dict) -> np.ndarray:
        """
        从元数据中提取特征
        
        特征列表：
        1. page_count - 页数（标准化）
        2. ref_count - 参考文献数（标准化）
        3. year_normalized - 年份（标准化到 0-1）
        4-9. problem_one_hot - 题目 one-hot 编码 (6维)
        10. contest_type - 赛道类型 (MCM=0, ICM=1)
        
        总计: 10 维
        """
        features = []
        
        # 1. 页数（标准化，假设正常范围 10-30 页）
        page_count = metadata.get('page_count', 20)
        page_count_norm = (page_count - 10) / 20  # 标准化
        features.append(page_count_norm)
        
        # 2. 参考文献数（标准化，假设正常范围 0-50）
        ref_count = metadata.get('ref_count', 10)
        ref_count_norm = ref_count / 50  # 标准化
        features.append(ref_count_norm)
        
        # 3. 年份标准化（范围 2010-2025）
        year = metadata.get('year', 2023)
        year_norm = (year - 2010) / 15  # 标准化到 0~1
        features.append(year_norm)
        
        # 4-9. 题目 one-hot 编码
        problem = metadata.get('problem', 'A')
        problem_one_hot = self._one_hot_encode_problem(problem)
        features.extend(problem_one_hot)
        
        # 10. 赛道类型
        contest = metadata.get('contest', 'MCM')
        contest_code = self.contest_encoder.get(contest, 0)
        features.append(contest_code)
        
        return np.array(features, dtype=np.float32)
    
    def _one_hot_encode_problem(self, problem: str) -> List[float]:
        """题目 one-hot 编码"""
        one_hot = [0.0] * 6  # A, B, C, D, E, F
        
        if problem in self.problem_encoder:
            idx = self.problem_encoder[problem]
            one_hot[idx] = 1.0
        
        return one_hot
    
    def batch_fuse(self, 
                   text_features_matrix: np.ndarray,
                   image_features_matrix: np.ndarray,
                   metadata_list: List[Dict]) -> np.ndarray:
        """
        批量融合特征
        
        参数：
            text_features_matrix: 文本特征矩阵 (n, 390)
            image_features_matrix: 图像特征矩阵 (n, 518)
            metadata_list: 元数据列表
        
        返回：
            融合特征矩阵 (n, ~918)
        """
        n_samples = len(metadata_list)
        print(f"正在融合 {n_samples} 个样本的特征...")
        
        fused_features_list = []
        
        for i in range(n_samples):
            fused = self.fuse(
                text_features_matrix[i],
                image_features_matrix[i],
                metadata_list[i]
            )
            fused_features_list.append(fused)
        
        fused_matrix = np.vstack(fused_features_list)
        print(f"特征融合完成: {fused_matrix.shape}")
        
        return fused_matrix
    
    def get_feature_names(self, text_feature_names: List[str], image_feature_names: List[str]) -> List[str]:
        """获取所有特征名称"""
        meta_names = [
            "page_count_norm",
            "ref_count_norm",
            "year_norm",
            "problem_A",
            "problem_B",
            "problem_C",
            "problem_D",
            "problem_E",
            "problem_F",
            "contest_type"
        ]
        
        return text_feature_names + image_feature_names + meta_names


def fuse_features(text_features: np.ndarray, 
                  image_features: np.ndarray,
                  year: int = 2024,
                  contest: str = 'MCM',
                  problem: str = 'A',
                  page_count: int = 20,
                  ref_count: int = 15,
                  config_path: str = "config.yaml") -> np.ndarray:
    """
    便捷函数：融合文本、图像和元数据特征
    
    参数:
        text_features: 文本特征向量
        image_features: 图像特征向量
        year: 年份
        contest: 赛道 (MCM/ICM)
        problem: 题目 (A-F)
        page_count: 页数
        ref_count: 参考文献数
        config_path: 配置文件路径
    
    返回:
        numpy.ndarray: 融合后的特征向量
    """
    fusion = FeatureFusion(config_path)
    metadata = {
        'page_count': page_count,
        'ref_count': ref_count,
        'year': year,
        'problem': problem,
        'contest': contest
    }
    return fusion.fuse(text_features, image_features, metadata)


def main():
    """测试特征融合"""
    # 模拟特征
    text_features = np.random.randn(390)
    image_features = np.random.randn(518)
    metadata = {
        'page_count': 25,
        'ref_count': 20,
        'year': 2023,
        'problem': 'A',
        'contest': 'MCM'
    }
    
    fusion = FeatureFusion()
    
    print("\n" + "="*60)
    print("特征融合测试")
    print("="*60)
    
    fused = fusion.fuse(text_features, image_features, metadata)
    
    print(f"\n文本特征维度: {text_features.shape}")
    print(f"图像特征维度: {image_features.shape}")
    print(f"元数据特征维度: 10")
    print(f"融合后特征维度: {fused.shape}")
    
    print(f"\n元数据特征（最后10维）:")
    print(fused[-10:])


if __name__ == "__main__":
    main()
