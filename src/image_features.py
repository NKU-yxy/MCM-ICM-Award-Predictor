"""
图像特征提取模块
从论文图片中提取视觉特征和统计特征
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config


class ImageFeatureExtractor:
    """图像特征提取器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化提取器"""
        self.config = load_config(config_path)
        self.image_config = self.config['image_features']
        
        # 加载预训练 ResNet18 模型
        print("正在加载图像特征提取模型...")
        self.model = models.resnet18(pretrained=True)
        
        # 移除最后的分类层，只保留特征提取部分
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()
        
        self.feature_dim = self.image_config['feature_dim']
        print(f"模型加载完成: {self.image_config['model_name']}")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.image_config['image_size'], self.image_config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.image_config['normalize_mean'],
                std=self.image_config['normalize_std']
            )
        ])
        
        # 设置设备（CPU）
        self.device = torch.device('cpu')
        self.feature_extractor.to(self.device)
    
    def extract(self, images: List[Image.Image]) -> Dict:
        """
        提取图像特征
        
        参数：
            images: PIL Image 列表
        
        返回：
        {
            'deep_features': np.ndarray,  # 深度特征 (512维)
            'statistical_features': np.ndarray,  # 统计特征 (6维)
            'feature_vector': np.ndarray  # 完整特征向量 (518维)
        }
        """
        if not images:
            # 无图片，返回零向量
            return {
                'deep_features': np.zeros(self.feature_dim),
                'statistical_features': np.zeros(6),
                'feature_vector': np.zeros(self.feature_dim + 6),
                'feature_dim': self.feature_dim + 6,
                'image_count': 0
            }
        
        # 1. 深度特征（ResNet）
        deep_features = self._extract_deep_features(images)
        
        # 2. 统计特征
        statistical_features = self._extract_statistical_features(images)
        
        # 合并特征
        feature_vector = np.concatenate([deep_features, statistical_features])
        
        return {
            'deep_features': deep_features,
            'statistical_features': statistical_features,
            'feature_vector': feature_vector,
            'feature_dim': len(feature_vector),
            'image_count': len(images)
        }
    
    def _extract_deep_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        使用 ResNet18 提取深度特征
        
        对所有图片提取特征后取平均（均值池化）
        """
        features_list = []
        
        with torch.no_grad():
            for img in images:
                try:
                    # 跳过超大图片（可能导致内存问题）
                    if img.width * img.height > 20_000_000:
                        continue
                    
                    # 预处理图像
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    
                    # 提取特征
                    feature = self.feature_extractor(img_tensor)
                    
                    # 展平 (1, 512, 1, 1) -> (512,)
                    feature = feature.squeeze().cpu().numpy()
                    features_list.append(feature)
                except Exception as e:
                    # 跳过无法处理的图片
                    continue
        
        # 均值池化：对所有图片的特征取平均
        if features_list:
            avg_features = np.mean(features_list, axis=0)
        else:
            avg_features = np.zeros(self.feature_dim)
        
        return avg_features
    
    def _extract_statistical_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        提取统计特征（归一化版本）
        
        特征列表：
        1. image_count_norm - 图片数量（归一化到0-1，假设范围0-100张）
        2. avg_width_norm - 平均宽度（归一化到0-1，假设范围0-2000像素）
        3. avg_height_norm - 平均高度（归一化到0-1，假设范围0-2000像素）
        4. avg_aspect_ratio - 平均宽高比（已经在合理范围0-3）
        5. resolution_std_norm - 分辨率标准差（归一化，假设范围0-1000000）
        6. avg_color_richness_norm - 平均颜色丰富度（归一化到0-1，假设范围0-2500）
        """
        if not images:
            return np.zeros(6)
        
        features = []
        
        # 1. 图片数量（归一化到0-1）
        image_count = len(images)
        image_count_norm = min(image_count / 100.0, 1.0)  # 假设最多100张图
        features.append(image_count_norm)
        
        # 收集尺寸信息
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        aspect_ratios = [w / h if h > 0 else 1.0 for w, h in zip(widths, heights)]
        resolutions = [w * h for w, h in zip(widths, heights)]
        
        # 2. 平均宽度（归一化）
        avg_width = np.mean(widths)
        avg_width_norm = min(avg_width / 2000.0, 1.0)  # 假设最大2000像素
        features.append(avg_width_norm)
        
        # 3. 平均高度（归一化）
        avg_height = np.mean(heights)
        avg_height_norm = min(avg_height / 2000.0, 1.0)  # 假设最大2000像素
        features.append(avg_height_norm)
        
        # 4. 平均宽高比（已经在合理范围）
        avg_aspect_ratio = np.mean(aspect_ratios)
        features.append(avg_aspect_ratio)
        
        # 5. 分辨率标准差（归一化）
        resolution_std = np.std(resolutions)
        resolution_std_norm = min(resolution_std / 1000000.0, 1.0)  # 归一化
        features.append(resolution_std_norm)
        
        # 6. 平均颜色丰富度（归一化）
        color_richness_list = []
        for img in images[:10]:  # 只采样前10张图片，避免太慢
            try:
                # 降低分辨率加速计算
                img_small = img.resize((50, 50))
                colors = img_small.getcolors(maxcolors=10000)
                if colors:
                    color_richness = len(colors)
                    color_richness_list.append(color_richness)
            except:
                pass
        
        avg_color_richness = np.mean(color_richness_list) if color_richness_list else 0
        avg_color_richness_norm = min(avg_color_richness / 2500.0, 1.0)  # 归一化
        features.append(avg_color_richness_norm)
        
        return np.array(features, dtype=np.float32)
    
    def batch_extract(self, images_list: List[List[Image.Image]]) -> np.ndarray:
        """
        批量提取特征
        
        参数：
            images_list: 图片列表的列表（每篇论文对应一个图片列表）
        
        返回：
            特征矩阵 (n_samples, feature_dim)
        """
        print(f"正在提取 {len(images_list)} 篇论文的图像特征...")
        
        features = []
        for i, images in enumerate(images_list):
            if (i + 1) % 10 == 0:
                print(f"  进度: {i + 1}/{len(images_list)}")
            
            result = self.extract(images)
            features.append(result['feature_vector'])
        
        feature_matrix = np.vstack(features)
        print(f"图像特征提取完成: {feature_matrix.shape}")
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        deep_names = [f"resnet_{i}" for i in range(self.feature_dim)]
        statistical_names = [
            "image_count_norm",
            "avg_width_norm",
            "avg_height_norm",
            "avg_aspect_ratio",
            "resolution_std_norm",
            "avg_color_richness_norm"
        ]
        
        return deep_names + statistical_names


def extract_image_features(images: List, config_path: str = "config.yaml") -> np.ndarray:
    """
    便捷函数：提取图像特征
    
    参数:
        images: PIL.Image对象列表
        config_path: 配置文件路径
    
    返回:
        numpy.ndarray: 特征向量
    """
    extractor = ImageFeatureExtractor(config_path)
    result = extractor.extract(images)
    return result['feature_vector']


def main():
    """测试图像特征提取器"""
    # 创建示例图片
    sample_images = [
        Image.new('RGB', (800, 600), color='red'),
        Image.new('RGB', (600, 400), color='blue'),
        Image.new('RGB', (1000, 800), color='green'),
    ]
    
    extractor = ImageFeatureExtractor()
    
    print("\n" + "="*60)
    print("图像特征提取测试")
    print("="*60)
    
    result = extractor.extract(sample_images)
    
    print(f"\n深度特征维度: {result['deep_features'].shape}")
    print(f"深度特征前5维: {result['deep_features'][:5]}")
    
    print(f"\n统计特征维度: {result['statistical_features'].shape}")
    print(f"统计特征: {result['statistical_features']}")
    
    feature_names = extractor.get_feature_names()
    stat_features = result['statistical_features']
    
    print("\n统计特征详情:")
    for name, value in zip(feature_names[-6:], stat_features):
        print(f"  {name}: {value:.2f}")
    
    print(f"\n完整特征向量维度: {result['feature_vector'].shape}")
    print(f"图片数量: {result['image_count']}")


if __name__ == "__main__":
    main()
