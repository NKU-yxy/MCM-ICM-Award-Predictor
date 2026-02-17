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
                'statistical_features': np.zeros(18),
                'feature_vector': np.zeros(self.feature_dim + 18),
                'feature_dim': self.feature_dim + 18,
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
        提取统计特征（增强版，含画风/内容分析）
        
        特征列表（18维）：
        1. image_count_norm - 图片数量（归一化到0-1，假设范围0-100张）
        2. avg_width_norm - 平均宽度（归一化到0-1，假设范围0-2000像素）
        3. avg_height_norm - 平均高度（归一化到0-1，假设范围0-2000像素）
        4. avg_aspect_ratio - 平均宽高比（已经在合理范围0-3）
        5. resolution_std_norm - 分辨率标准差（归一化，假设范围0-1000000）
        6. avg_color_richness_norm - 平均颜色丰富度（归一化到0-1，假设范围0-2500）
        --- 新增：画风/内容/专业度分析 ---
        7. chart_ratio - 图表类图片比例 (vs 照片/装饰)
        8. avg_edge_density - 平均边缘密度（高=图表/技术图，低=照片）
        9. avg_saturation - 平均饱和度（学术图表通常饱和度适中）
        10. color_mode_ratio - 彩色图片比例（vs 灰度）
        11. avg_contrast - 平均对比度（专业图表通常对比度高）
        12. size_consistency - 图片尺寸一致性（高=排版规范）
        13. avg_whitespace_ratio - 平均留白比例（图表通常有白色背景）
        14. professional_score - 综合专业度评分
        15. has_colorbar - 是否有热力图/colorbar类图（科学可视化指标）
        16. avg_text_region_ratio - 图片中文字区域比例（说明标注丰富度）
        17. visual_diversity - 图片视觉多样性（好论文图表类型多样）
        18. high_res_ratio - 高分辨率图片比例（高质量图表指标）
        """
        if not images:
            return np.zeros(18)
        
        features = []
        
        # 1. 图片数量（归一化到0-1）
        image_count = len(images)
        image_count_norm = min(image_count / 100.0, 1.0)
        features.append(image_count_norm)
        
        # 收集尺寸信息
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        aspect_ratios = [w / h if h > 0 else 1.0 for w, h in zip(widths, heights)]
        resolutions = [w * h for w, h in zip(widths, heights)]
        
        # 2. 平均宽度（归一化）
        avg_width = np.mean(widths)
        avg_width_norm = min(avg_width / 2000.0, 1.0)
        features.append(avg_width_norm)
        
        # 3. 平均高度（归一化）
        avg_height = np.mean(heights)
        avg_height_norm = min(avg_height / 2000.0, 1.0)
        features.append(avg_height_norm)
        
        # 4. 平均宽高比
        avg_aspect_ratio = np.mean(aspect_ratios)
        features.append(avg_aspect_ratio)
        
        # 5. 分辨率标准差（归一化）
        resolution_std = np.std(resolutions)
        resolution_std_norm = min(resolution_std / 1000000.0, 1.0)
        features.append(resolution_std_norm)
        
        # 6. 平均颜色丰富度（归一化）
        color_richness_list = []
        for img in images[:10]:
            try:
                img_small = img.resize((50, 50))
                colors = img_small.getcolors(maxcolors=10000)
                if colors:
                    color_richness = len(colors)
                    color_richness_list.append(color_richness)
            except:
                pass
        
        avg_color_richness = np.mean(color_richness_list) if color_richness_list else 0
        avg_color_richness_norm = min(avg_color_richness / 2500.0, 1.0)
        features.append(avg_color_richness_norm)
        
        # ==================== 新增：画风/内容分析 ====================
        # 采样分析（最多20张，避免太慢）
        sample_images = images[:20]
        
        edge_densities = []
        saturations = []
        contrasts = []
        whitespace_ratios = []
        is_color_list = []
        has_colorbar_count = 0
        text_region_ratios = []
        
        for img in sample_images:
            try:
                analysis = self._analyze_single_image(img)
                edge_densities.append(analysis['edge_density'])
                saturations.append(analysis['avg_saturation'])
                contrasts.append(analysis['contrast'])
                whitespace_ratios.append(analysis['whitespace_ratio'])
                is_color_list.append(analysis['is_color'])
                if analysis['has_colorbar_hint']:
                    has_colorbar_count += 1
                text_region_ratios.append(analysis['text_region_ratio'])
            except:
                pass
        
        n_analyzed = max(len(edge_densities), 1)
        
        # 7. 图表类图片比例（边缘密度高 = 图表/技术图）
        chart_threshold = 0.15  # 边缘密度阈值
        chart_count = sum(1 for ed in edge_densities if ed > chart_threshold)
        chart_ratio = chart_count / n_analyzed
        features.append(chart_ratio)
        
        # 8. 平均边缘密度
        avg_edge_density = np.mean(edge_densities) if edge_densities else 0
        features.append(min(avg_edge_density, 1.0))
        
        # 9. 平均饱和度
        avg_saturation = np.mean(saturations) if saturations else 0
        features.append(avg_saturation)
        
        # 10. 彩色图片比例
        color_ratio = sum(is_color_list) / n_analyzed if is_color_list else 0
        features.append(color_ratio)
        
        # 11. 平均对比度
        avg_contrast = np.mean(contrasts) if contrasts else 0
        features.append(min(avg_contrast / 128.0, 1.0))  # 归一化到0-1
        
        # 12. 尺寸一致性（标准差越小越一致）
        if len(resolutions) > 1:
            res_cv = np.std(resolutions) / max(np.mean(resolutions), 1)
            size_consistency = max(0, 1.0 - res_cv)  # CV越小越一致
        else:
            size_consistency = 1.0
        features.append(size_consistency)
        
        # 13. 平均留白比例
        avg_whitespace = np.mean(whitespace_ratios) if whitespace_ratios else 0
        features.append(avg_whitespace)
        
        # 14. 综合专业度评分
        # 好的学术图表特征：高边缘密度、适中饱和度、高对比度、有留白、尺寸一致
        professional_score = (
            0.25 * chart_ratio +
            0.20 * min(avg_edge_density / 0.2, 1.0) +
            0.15 * avg_contrast / 128.0 +
            0.15 * size_consistency +
            0.15 * avg_whitespace +
            0.10 * (1.0 - abs(avg_saturation - 0.3))  # 饱和度适中得分高
        )
        features.append(min(professional_score, 1.0))
        
        # 15. 有colorbar类图（科学可视化）
        has_colorbar = min(has_colorbar_count / n_analyzed, 1.0)
        features.append(has_colorbar)
        
        # 16. 图片中文字区域比例
        avg_text_ratio = np.mean(text_region_ratios) if text_region_ratios else 0
        features.append(avg_text_ratio)
        
        # 17. 视觉多样性（图片间特征方差）
        if len(edge_densities) > 1:
            visual_diversity = np.std(edge_densities) + np.std(saturations)
            visual_diversity = min(visual_diversity / 0.3, 1.0)
        else:
            visual_diversity = 0
        features.append(visual_diversity)
        
        # 18. 高分辨率图片比例（宽>400或高>300）
        high_res_count = sum(1 for w, h in zip(widths, heights) if w > 400 or h > 300)
        high_res_ratio = high_res_count / max(len(images), 1)
        features.append(high_res_ratio)
        
        return np.array(features, dtype=np.float32)
    
    def _analyze_single_image(self, img: Image.Image) -> dict:
        """
        分析单张图片的画风/内容特征
        
        返回各项指标的字典
        """
        # 缩小到便于分析的尺寸
        analysis_size = (100, 100)
        img_small = img.resize(analysis_size)
        pixels = np.array(img_small, dtype=np.float32)
        
        # --- 边缘密度（Sobel approximation） ---
        gray = np.mean(pixels, axis=2)  # 灰度
        # 简单梯度近似
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        edge_map = (np.mean(dx) + np.mean(dy)) / 2.0
        edge_density = edge_map / 255.0  # 归一化到0-1
        
        # --- 饱和度 ---
        # 将RGB转HSV手动计算（避免依赖cv2）
        r, g, b = pixels[:,:,0]/255, pixels[:,:,1]/255, pixels[:,:,2]/255
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        sat = np.where(cmax > 0, delta / cmax, 0)
        avg_saturation = float(np.mean(sat))
        
        # --- 对比度（标准差） ---
        contrast = float(np.std(gray))
        
        # --- 留白比例（接近白色的像素比例） ---
        white_threshold = 240
        white_pixels = np.all(pixels > white_threshold, axis=2)
        whitespace_ratio = float(np.mean(white_pixels))
        
        # --- 是否彩色 ---
        is_color = float(avg_saturation > 0.08)
        
        # --- colorbar 检测（图片右侧有窄条状渐变色带） ---
        has_colorbar_hint = False
        try:
            right_strip = pixels[:, -10:, :]  # 最右10列
            right_v_std = np.std(np.mean(right_strip, axis=1), axis=0)
            # 垂直方向有渐变 + 水平方向较一致
            right_h_std = np.std(np.mean(right_strip, axis=0), axis=0)
            if np.mean(right_v_std) > 20 and np.mean(right_h_std) < 30:
                has_colorbar_hint = True
        except:
            pass
        
        # --- 文字区域比例（高对比度、小块密集区 → 可能是文字标注） ---
        # 简化: 高梯度且低饱和度的区域倾向于是文字
        high_gradient = edge_density > 0.1
        low_sat = avg_saturation < 0.15
        text_region_ratio = 0.0
        if high_gradient and low_sat:
            # 检查是否有大量小高对比点（类似文字）
            binary = (gray > 128).astype(float)
            transitions = np.abs(np.diff(binary, axis=1)).sum() + np.abs(np.diff(binary, axis=0)).sum()
            text_region_ratio = min(transitions / (analysis_size[0] * analysis_size[1]), 1.0)
        
        return {
            'edge_density': edge_density,
            'avg_saturation': avg_saturation,
            'contrast': contrast,
            'whitespace_ratio': whitespace_ratio,
            'is_color': is_color,
            'has_colorbar_hint': has_colorbar_hint,
            'text_region_ratio': text_region_ratio,
        }
    
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
            "avg_color_richness_norm",
            "chart_ratio",
            "avg_edge_density",
            "avg_saturation",
            "color_mode_ratio",
            "avg_contrast",
            "size_consistency",
            "avg_whitespace_ratio",
            "professional_score",
            "has_colorbar",
            "avg_text_region_ratio",
            "visual_diversity",
            "high_res_ratio",
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
