"""
预测脚本
对单篇论文进行获奖等级预测
"""

import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.pdf_parser import PDFParser
from src.text_features import TextFeatureExtractor
from src.image_features import ImageFeatureExtractor
from src.feature_fusion import FeatureFusion
from scripts.train import MCMClassifier


class MCMPredictor:
    """MCM/ICM 论文获奖等级预测器"""
    
    def __init__(self, model_dir: str = "data/models", config_path: str = "config.yaml"):
        """初始化预测器"""
        print("初始化预测器...")
        
        # 加载各模块
        self.pdf_parser = PDFParser(config_path)
        self.text_extractor = TextFeatureExtractor(config_path)
        self.image_extractor = ImageFeatureExtractor(config_path)
        self.fusion = FeatureFusion(config_path)
        
        # 加载训练好的模型
        self.classifier = MCMClassifier(config_path)
        self.classifier.load_model(model_dir)
        
        print("预测器初始化完成！\n")
    
    def predict(self, pdf_path: str, problem: str = None, year: int = None) -> dict:
        """
        预测单篇论文的获奖等级
        
        参数：
            pdf_path: PDF 文件路径
            problem: 题目 (A/B/C/D/E/F)，如果为 None 则从文件路径推断
            year: 年份，如果为 None 则从文件路径推断
        
        返回：
            {
                'probabilities': dict,  # 各等级的概率 {'O+F': 0.05, 'M': 0.70, ...}
                'predicted_label': str,  # 预测的等级
                'confidence': float,  # 置信度（最高概率）
            }
        """
        print(f"正在预测: {pdf_path}")
        
        # 1. 解析 PDF
        print("  [1/4] 解析 PDF...")
        parsed = self.pdf_parser.parse(pdf_path)
        
        if not parsed['success']:
            return {
                'error': f"PDF 解析失败: {parsed.get('error', 'Unknown error')}",
                'success': False
            }
        
        # 补充元数据
        metadata = parsed['metadata']
        if problem:
            metadata['problem'] = problem
        if year:
            metadata['year'] = year
        
        # 2. 提取文本特征
        print("  [2/4] 提取文本特征...")
        text_result = self.text_extractor.extract(parsed['abstract'])
        text_features = text_result['feature_vector']
        
        # 3. 提取图像特征
        print("  [3/4] 提取图像特征...")
        image_result = self.image_extractor.extract(parsed['images'])
        image_features = image_result['feature_vector']
        
        # 4. 融合特征
        print("  [4/4] 特征融合与预测...")
        fused_features = self.fusion.fuse(text_features, image_features, metadata)
        
        # 标准化
        fused_features_scaled = self.classifier.scaler.transform(fused_features.reshape(1, -1))
        
        # 预测
        pred_proba = self.classifier.model.predict_proba(fused_features_scaled)[0]
        pred_label_idx = np.argmax(pred_proba)
        pred_label = self.classifier.label_encoder.inverse_transform([pred_label_idx])[0]
        
        # 构建概率字典
        probabilities = {}
        for i, label in enumerate(self.classifier.label_encoder.classes_):
            probabilities[label] = float(pred_proba[i])
        
        return {
            'probabilities': probabilities,
            'predicted_label': pred_label,
            'confidence': float(pred_proba[pred_label_idx]),
            'success': True,
            'metadata': {
                'abstract_length': len(parsed['abstract']),
                'image_count': len(parsed['images']),
                'page_count': metadata.get('page_count', 0)
            }
        }
    
    def print_result(self, result: dict, pdf_path: str = None):
        """格式化打印预测结果"""
        print("\n" + "="*60)
        print("  MCM/ICM 获奖等级预测结果")
        print("="*60)
        
        if not result.get('success', False):
            print(f"预测失败: {result.get('error', 'Unknown error')}")
            return
        
        if pdf_path:
            print(f"论文: {Path(pdf_path).name}")
        
        print(f"\n预测概率分布:")
        probabilities = result['probabilities']
        
        # 按概率排序
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for label, prob in sorted_probs:
            bar_length = int(prob * 40)
            bar = "█" * bar_length
            marker = " ★ 最可能" if label == result['predicted_label'] else ""
            print(f"  {label:5s} ({prob*100:5.1f}%): {bar}{marker}")
        
        # 置信度评估
        confidence = result['confidence']
        if confidence > 0.7:
            confidence_level = "高"
        elif confidence > 0.5:
            confidence_level = "中等"
        else:
            confidence_level = "低"
        
        print(f"\n置信度: {confidence_level} (最高概率 {confidence*100:.1f}%)")
        
        # 元数据
        if 'metadata' in result:
            meta = result['metadata']
            print(f"\n论文信息:")
            print(f"  摘要长度: {meta.get('abstract_length', 0)} 字符")
            print(f"  图片数量: {meta.get('image_count', 0)} 张")
            print(f"  总页数: {meta.get('page_count', 0)} 页")
        
        print("="*60 + "\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCM/ICM 论文获奖等级预测")
    parser.add_argument("--pdf", type=str, required=True, help="PDF 文件路径")
    parser.add_argument("--problem", type=str, help="题目 (A/B/C/D/E/F)")
    parser.add_argument("--year", type=int, help="年份")
    parser.add_argument("--model_dir", type=str, default="data/models", help="模型目录")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.pdf).exists():
        print(f"错误: 文件不存在 {args.pdf}")
        return
    
    # 初始化预测器
    predictor = MCMPredictor(model_dir=args.model_dir)
    
    # 预测
    result = predictor.predict(args.pdf, args.problem, args.year)
    
    # 显示结果
    predictor.print_result(result, args.pdf)


if __name__ == "__main__":
    main()
