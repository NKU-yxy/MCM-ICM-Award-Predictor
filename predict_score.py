"""
O奖相似度打分预测脚本
输出0-100的相似度分数
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_parser import extract_paper_content
from src.text_features import extract_text_features
from src.image_features import extract_image_features
from src.feature_fusion import fuse_features

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_scoring_model(model_path: str = "models/scoring_model.pkl"):
    """加载打分模型"""
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        logger.error("请先运行训练脚本: python scripts/train_scoring_model.py")
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data


def compute_score(features: np.ndarray, centroid: np.ndarray, stats: dict) -> float:
    """
    计算相似度分数 (0-100)
    
    基于余弦相似度，映射到0-100区间
    
    策略:
    1. 计算余弦相似度 (范围 -1 到 1)
    2. 使用O奖统计信息标定分数
    3. similarity_mean 对应 ~85分（平均O奖水平）
    4. similarity_max 对应 ~100分（最优O奖）
    5. similarity_mean - 2*std 对应 ~50分（及格线）
    """
    # 维度对齐（模型可能用旧版特征训练）
    f_dim = features.shape[-1] if features.ndim > 1 else features.shape[0]
    c_dim = centroid.shape[-1] if centroid.ndim > 1 else centroid.shape[0]
    if f_dim != c_dim:
        min_dim = min(f_dim, c_dim)
        logger.warning(f"特征维度不匹配: features={f_dim}, centroid={c_dim}，截断到{min_dim}维")
        features = features.flatten()[:min_dim]
        centroid = centroid.flatten()[:min_dim]
    
    # 计算余弦相似度
    similarity = cosine_similarity(
        features.reshape(1, -1),
        centroid.reshape(1, -1)
    )[0, 0]
    
    # 获取统计信息
    sim_mean = stats['similarity_mean']
    sim_std = stats['similarity_std']
    sim_min = stats['similarity_min']
    sim_max = stats['similarity_max']
    
    # 分数映射（添加除零保护）
    if similarity >= sim_max:
        score = 100.0
    elif similarity >= sim_mean:
        score = 85 + 15 * (similarity - sim_mean) / max(sim_max - sim_mean, 1e-6)
    elif similarity >= sim_mean - 2 * sim_std:
        score = 50 + 35 * (similarity - (sim_mean - 2*sim_std)) / max(2*sim_std, 1e-6)
    else:
        threshold = sim_mean - 2 * sim_std
        if similarity > sim_min:
            score = 50 * (similarity - sim_min) / max(threshold - sim_min, 1e-6)
        else:
            score = max(0.0, 50 * (similarity - sim_min) / max(threshold - sim_min, 1e-6))
    
    # 确保范围
    score = np.clip(score, 0, 100)
    
    return float(score), float(similarity)


def get_score_interpretation(score: float) -> tuple:
    """
    解释分数等级
    
    返回: (等级, 描述, 颜色标记)
    """
    if score >= 95:
        return "卓越", "达到顶尖O奖水平", "🌟"
    elif score >= 85:
        return "优秀", "达到典型O奖水平", "⭐"
    elif score >= 75:
        return "良好", "接近O奖水平", "✨"
    elif score >= 60:
        return "中等", "有一定潜力，需改进", "💡"
    elif score >= 40:
        return "及格", "基础可用，需大幅提升", "📝"
    else:
        return "较弱", "与O奖标准差距较大", "⚠️"


def predict_paper(pdf_path: str, model_data: dict, verbose: bool = True) -> dict:
    """
    预测单篇论文的分数
    
    返回:
        dict: 包含分数、相似度、等级等信息
    """
    if not os.path.exists(pdf_path):
        logger.error(f"文件不存在: {pdf_path}")
        return None
    
    try:
        # 1. 提取内容
        if verbose:
            logger.info(f"\n处理论文: {os.path.basename(pdf_path)}")
            logger.info("="*60)
        
        content = extract_paper_content(pdf_path)
        
        if not content['abstract']:
            logger.error("未找到摘要！")
            return None
        
        if verbose:
            logger.info(f"✓ 提取摘要: {len(content['abstract'])} 字符")
            logger.info(f"✓ 提取图片: {len(content['images'])} 张")
        
        # 2. 提取特征
        text_feat = extract_text_features(content['abstract'])
        image_feat = extract_image_features(content['images'])
        
        # 从路径提取元数据
        path_parts = Path(pdf_path).parts
        year, contest, problem = None, 'MCM', 'A'
        
        # 尝试从路径提取
        for part in path_parts:
            if part.isdigit() and 2010 <= int(part) <= 2030:
                year = int(part)
            if '_' in part and part.count('_') == 1:
                c, p = part.split('_')
                if c in ['MCM', 'ICM'] and p in 'ABCDEF':
                    contest, problem = c, p
        
        # 如果没找到，使用默认值
        if not year:
            year = 2025
        
        # 从PDF元数据中获取页数和参考文献数
        pdf_metadata = content.get('metadata', {})
        page_count = pdf_metadata.get('page_count', 20)
        ref_count = pdf_metadata.get('ref_count', 15)
        
        # 融合特征（已在fusion层做per-group L2归一化）
        features = fuse_features(text_feat, image_feat, year, contest, problem,
                                 page_count=page_count, ref_count=ref_count)
        
        # 3. 计算分数
        centroid = model_data['centroid']
        stats = model_data['stats']
        
        score, similarity = compute_score(features, centroid, stats)
        
        # 4. 解释
        level, description, emoji = get_score_interpretation(score)
        
        result = {
            'score': score,
            'similarity': similarity,
            'level': level,
            'description': description,
            'emoji': emoji,
            'filename': os.path.basename(pdf_path),
            'abstract_length': len(content['abstract']),
            'image_count': len(content['images'])
        }
        
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"评分结果")
            logger.info(f"{'='*60}")
            logger.info(f"\n  {emoji} 分数: {score:.1f} / 100")
            logger.info(f"  等级: {level}")
            logger.info(f"  评价: {description}")
            logger.info(f"\n  相似度: {similarity:.4f}")
            logger.info(f"  O奖平均: {stats['similarity_mean']:.4f}")
            logger.info(f"\n{'='*60}")
        
        return result
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def batch_predict(pdf_dir: str, model_data: dict):
    """批量预测目录下的所有PDF"""
    pdf_dir = Path(pdf_dir)
    
    if not pdf_dir.exists():
        logger.error(f"目录不存在: {pdf_dir}")
        return
    
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"目录中没有PDF文件: {pdf_dir}")
        return
    
    logger.info(f"\n批量预测: {len(pdf_files)} 个文件")
    logger.info("="*60)
    
    results = []
    
    for pdf_file in pdf_files:
        result = predict_paper(str(pdf_file), model_data, verbose=False)
        
        if result:
            results.append(result)
            logger.info(f"{result['emoji']} {result['score']:5.1f} | {result['filename']}")
    
    # 统计
    if results:
        scores = [r['score'] for r in results]
        logger.info(f"\n{'='*60}")
        logger.info("批量统计")
        logger.info(f"{'='*60}")
        logger.info(f"  总计: {len(scores)} 篇")
        logger.info(f"  平均分: {np.mean(scores):.1f}")
        logger.info(f"  标准差: {np.std(scores):.1f}")
        logger.info(f"  最高分: {np.max(scores):.1f}")
        logger.info(f"  最低分: {np.min(scores):.1f}")
        logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="O奖相似度打分预测")
    parser.add_argument("pdf_path", help="PDF文件路径或目录")
    parser.add_argument("--model", default="models/scoring_model.pkl", help="模型路径")
    parser.add_argument("--batch", action="store_true", help="批量预测模式")
    
    args = parser.parse_args()
    
    # 加载模型
    model_data = load_scoring_model(args.model)
    
    if model_data is None:
        return
    
    logger.info(f"\n已加载模型 (基于 {model_data['n_papers']} 篇 O 奖论文)")
    
    # 预测
    if args.batch or os.path.isdir(args.pdf_path):
        batch_predict(args.pdf_path, model_data)
    else:
        predict_paper(args.pdf_path, model_data)


if __name__ == "__main__":
    main()
