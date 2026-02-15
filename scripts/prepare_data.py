"""
数据预处理脚本
批量解析 PDF、提取特征、生成训练数据集
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pdf_parser import PDFParser
from src.text_features import TextFeatureExtractor
from src.image_features import ImageFeatureExtractor
from src.feature_fusion import FeatureFusion
from src.utils import load_config, ensure_dir, save_pickle


def prepare_dataset(config_path: str = "config.yaml"):
    """
    准备训练数据集
    
    流程：
    1. 扫描 raw 目录下的所有 PDF
    2. 解析 PDF（提取摘要、图片、元数据）
    3. 提取文本特征
    4. 提取图像特征
    5. 融合特征
    6. 保存到 processed 目录
    """
    print("="*60)
    print("MCM/ICM 数据预处理")
    print("="*60)
    
    # 加载配置
    config = load_config(config_path)
    raw_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    ensure_dir(processed_dir)
    
    # 初始化各模块
    print("\n1. 初始化模块...")
    pdf_parser = PDFParser(config_path)
    text_extractor = TextFeatureExtractor(config_path)
    image_extractor = ImageFeatureExtractor(config_path)
    fusion = FeatureFusion(config_path)
    
    # 扫描所有 PDF 文件
    print(f"\n2. 扫描 PDF 文件 ({raw_dir})...")
    pdf_files = list(Path(raw_dir).rglob("*.pdf"))
    
    if not pdf_files:
        print(f"错误: 在 {raw_dir} 中未找到 PDF 文件")
        print("请先使用 crawl_papers.py 下载论文或手动放置 PDF 文件")
        return
    
    print(f"找到 {len(pdf_files)} 个 PDF 文件")
    
    # 解析所有 PDF
    print("\n3. 解析 PDF 文件...")
    parsed_data = []
    
    for pdf_file in tqdm(pdf_files, desc="解析PDF"):
        result = pdf_parser.parse(str(pdf_file))
        result['file_path'] = str(pdf_file)
        parsed_data.append(result)
    
    # 过滤失败的解析
    successful_data = [d for d in parsed_data if d['success']]
    print(f"成功解析: {len(successful_data)}/{len(parsed_data)}")
    
    if not successful_data:
        print("错误: 没有成功解析的 PDF")
        return
    
    # 提取文本特征
    print("\n4. 提取文本特征...")
    abstracts = [d['abstract'] for d in successful_data]
    text_features = text_extractor.batch_extract(abstracts)
    
    # 提取图像特征
    print("\n5. 提取图像特征...")
    images_list = [d['images'] for d in successful_data]
    image_features = image_extractor.batch_extract(images_list)
    
    # 融合特征
    print("\n6. 融合特征...")
    metadata_list = [d['metadata'] for d in successful_data]
    fused_features = fusion.batch_fuse(text_features, image_features, metadata_list)
    
    # 提取标签
    print("\n7. 提取标签...")
    labels = []
    label_mapping = config['label_mapping']
    merge_labels = config['merge_labels']
    
    for d in successful_data:
        award = d['metadata'].get('award', 'S')
        
        # 应用标签合并（O 和 F 合并为 O+F）
        if award in merge_labels:
            award = merge_labels[award]
        
        labels.append(award)
    
    # 统计标签分布
    print("\n标签分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")
    
    # 保存数据
    print(f"\n8. 保存处理后的数据到 {processed_dir}...")
    
    # 保存特征矩阵
    np.save(Path(processed_dir) / "features.npy", fused_features)
    print(f"  特征矩阵: {fused_features.shape}")
    
    # 保存标签
    np.save(Path(processed_dir) / "labels.npy", np.array(labels))
    print(f"  标签数量: {len(labels)}")
    
    # 保存元数据
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df['label'] = labels
    metadata_df.to_csv(Path(processed_dir) / "metadata.csv", index=False)
    print(f"  元数据保存到 metadata.csv")
    
    # 保存特征名称
    feature_names = fusion.get_feature_names(
        text_extractor.get_feature_names(),
        image_extractor.get_feature_names()
    )
    save_pickle(feature_names, str(Path(processed_dir) / "feature_names.pkl"))
    print(f"  特征名称: {len(feature_names)} 个")
    
    print("\n" + "="*60)
    print("数据预处理完成！")
    print("="*60)
    print(f"特征维度: {fused_features.shape}")
    print(f"样本数量: {len(labels)}")
    print(f"输出目录: {processed_dir}")


def check_data_quality(processed_dir: str = "data/processed"):
    """
    检查处理后的数据质量
    """
    print("\n数据质量检查:")
    print("-" * 60)
    
    # 加载数据
    features = np.load(Path(processed_dir) / "features.npy")
    labels = np.load(Path(processed_dir) / "labels.npy")
    metadata_df = pd.read_csv(Path(processed_dir) / "metadata.csv")
    
    # 检查缺失值
    print(f"特征矩阵形状: {features.shape}")
    print(f"标签数量: {len(labels)}")
    print(f"是否有 NaN: {np.isnan(features).any()}")
    print(f"是否有 Inf: {np.isinf(features).any()}")
    
    # 检查特征范围
    print(f"\n特征统计:")
    print(f"  最小值: {features.min():.4f}")
    print(f"  最大值: {features.max():.4f}")
    print(f"  均值: {features.mean():.4f}")
    print(f"  标准差: {features.std():.4f}")
    
    # 检查标签分布
    print(f"\n标签分布:")
    for label in np.unique(labels):
        count = (labels == label).sum()
        ratio = count / len(labels) * 100
        print(f"  {label}: {count} ({ratio:.1f}%)")
    
    # 检查元数据
    print(f"\n元数据统计:")
    if 'year' in metadata_df.columns:
        print(f"  年份范围: {metadata_df['year'].min()} - {metadata_df['year'].max()}")
    if 'contest' in metadata_df.columns:
        print(f"  赛道分布:\n{metadata_df['contest'].value_counts()}")
    if 'problem' in metadata_df.columns:
        print(f"  题目分布:\n{metadata_df['problem'].value_counts()}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCM/ICM 数据预处理")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--check", action="store_true", help="检查数据质量")
    
    args = parser.parse_args()
    
    if args.check:
        check_data_quality()
    else:
        prepare_dataset(args.config)
        check_data_quality()


if __name__ == "__main__":
    main()
