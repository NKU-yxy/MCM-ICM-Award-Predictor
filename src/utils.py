"""
工具函数模块
"""

import os
import yaml
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List


def load_config(config_path: str = "config.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_json(data: Any, filepath: str):
    """保存 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """加载 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str):
    """保存 Pickle 文件"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """加载 Pickle 文件"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def ensure_dir(directory: str):
    """确保目录存在"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_paper_info_from_filename(filename: str) -> Dict:
    """
    从文件名提取论文信息
    格式: {队伍编号}_{获奖等级}.pdf
    例: 2312345_M.pdf
    """
    name = Path(filename).stem
    parts = name.split('_')
    
    if len(parts) >= 2:
        return {
            'team_id': parts[0],
            'award': parts[1]
        }
    return {}


def parse_problem_path(filepath: str) -> Dict:
    """
    从文件路径提取题目信息
    例: data/raw/2023/MCM_A/2312345_M.pdf
    """
    parts = Path(filepath).parts
    info = {}
    
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) == 4:
            info['year'] = int(part)
        if '_' in part and any(x in part for x in ['MCM', 'ICM']):
            contest_problem = part.split('_')
            if len(contest_problem) == 2:
                info['contest'] = contest_problem[0]
                info['problem'] = contest_problem[1]
    
    return info


def count_sentences(text: str) -> int:
    """统计句子数量"""
    import re
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def calculate_avg_sentence_length(text: str) -> float:
    """计算平均句长"""
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    total_words = sum(len(s.split()) for s in sentences)
    return total_words / len(sentences)


def has_numerical_results(text: str) -> bool:
    """检测文本中是否包含数字结果"""
    import re
    # 查找百分比、小数、科学计数法等
    patterns = [
        r'\d+\.\d+',  # 小数
        r'\d+%',      # 百分比
        r'\d+e[+-]?\d+',  # 科学计数法
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def calculate_technical_term_density(text: str) -> float:
    """
    计算技术术语密度
    简单实现：统计长单词（>= 8 字符）占比
    """
    words = text.split()
    if not words:
        return 0.0
    
    long_words = [w for w in words if len(w) >= 8]
    return len(long_words) / len(words)


def check_abstract_structure(text: str) -> float:
    """
    检测摘要结构完整性
    好的摘要通常包含：背景、方法、结果、结论
    """
    keywords = {
        'background': ['background', 'problem', 'challenge', 'issue'],
        'method': ['method', 'approach', 'algorithm', 'model', 'framework'],
        'result': ['result', 'find', 'achieve', 'obtain', 'show'],
        'conclusion': ['conclusion', 'finally', 'summary', 'demonstrate']
    }
    
    text_lower = text.lower()
    score = 0
    
    for category, words in keywords.items():
        if any(word in text_lower for word in words):
            score += 1
    
    return score / len(keywords)  # 0-1 之间的分数


def calculate_readability_score(text: str) -> float:
    """
    计算 Flesch Reading Ease 近似分数
    
    公式: 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    高分 = 易读，低分 = 难读
    学术论文通常在 10-30 之间
    归一化到 0-1 范围（0=极难, 1=极易）
    """
    import re
    
    words = text.split()
    total_words = len(words)
    if total_words == 0:
        return 0.5
    
    sentences = re.split(r'[.!?]+', text)
    total_sentences = max(len([s for s in sentences if s.strip()]), 1)
    
    # 简单音节计数（每个元音组算一个音节）
    def count_syllables(word):
        word = word.lower().strip(".,!?;:'\"")
        if len(word) <= 2:
            return 1
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)
    
    total_syllables = sum(count_syllables(w) for w in words)
    
    flesch = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    
    # 归一化到 0-1 (学术论文一般 0-50, 越低越专业)
    # 映射: 0 (极难) -> 1.0 (高学术性), 100 (极易) -> 0.0
    normalized = max(0.0, min(1.0, (100 - flesch) / 100.0))
    
    return normalized


def calculate_vocabulary_diversity(text: str) -> float:
    """
    计算词汇多样性 (Type-Token Ratio)
    
    高质量论文通常有更丰富的词汇
    返回 0-1 之间的 TTR
    """
    import re
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return 0.0
    
    # 使用根号 TTR (Root TTR) 避免长文本偏低
    # Root TTR = unique_words / sqrt(total_words)
    unique_words = len(set(words))
    total_words = len(words)
    
    root_ttr = unique_words / (total_words ** 0.5)
    
    # 归一化：通常 root_ttr 在 5-20 之间
    normalized = min(root_ttr / 20.0, 1.0)
    
    return normalized


def count_academic_phrases(text: str) -> float:
    """
    统计学术短语密度
    
    高质量论文使用更多学术表达
    """
    academic_phrases = [
        'in this paper', 'we propose', 'we present', 'our approach',
        'we develop', 'we analyze', 'we investigate', 'we evaluate',
        'experimental results', 'our model', 'we demonstrate',
        'state of the art', 'state-of-the-art', 'novel approach',
        'we introduce', 'our contribution', 'we consider',
        'without loss of generality', 'it follows that',
        'we observe', 'furthermore', 'moreover', 'nevertheless',
        'in particular', 'specifically', 'in contrast',
        'as shown in', 'as illustrated', 'in figure',
        'optimal solution', 'convergence', 'objective function',
        'constraint', 'optimization', 'sensitivity analysis',
        'validation', 'cross-validation', 'robustness',
    ]
    
    text_lower = text.lower()
    total_words = max(len(text_lower.split()), 1)
    
    count = sum(1 for phrase in academic_phrases if phrase in text_lower)
    
    # 归一化：密度 = 匹配数 / 总词数 * 1000
    density = count / total_words * 1000
    normalized = min(density / 10.0, 1.0)
    
    return normalized


if __name__ == "__main__":
    # 测试
    config = load_config()
    print("配置加载成功！")
    print(f"数据目录: {config['data']['raw_dir']}")
