"""
题目自动识别模块

从 PDF 论文内容中自动识别其对应的 MCM/ICM 题目 (A-F)。

核心策略：
1. 首先检查文件路径/名称中是否有题目信息
2. 分析论文正文中对题目的引用（如 "Problem A", "问题C" 等）
3. 基于论文内容关键词匹配各题目的特征词
4. 结合多个信号综合判断，输出置信度
"""

import re
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# 各年份各题目的关键词（用于内容匹配）
# 这些关键词来自真实的 MCM/ICM 赛题
PROBLEM_KEYWORDS_BY_YEAR = {
    2024: {
        'A': ['lamprey', 'sea lamprey', 'great lakes', 'invasive species', 'population dynamics'],
        'B': ['momentum', 'tennis', 'match', 'game', 'scoring', 'point'],
        'C': ['momentum', 'tennis', 'wimbledon', 'match momentum', 'sport'],
        'D': ['great lakes', 'lake network', 'water', 'shipping', 'navigation'],
        'E': ['property insurance', 'climate', 'insurance', 'catastrophe', 'risk'],
        'F': ['equal access', 'education', 'resources', 'equity', 'access'],
    },
    2023: {
        'A': ['drought', 'plant', 'communities', 'ecosystem'],
        'B': ['reword', 'search and rescue', 'submarine', 'search'],
        'C': ['wordle', 'word game', 'guess', 'puzzle', 'letters'],
        'D': ['priority', 'united nations', 'sustainability', 'sdg'],
        'E': ['light pollution', 'biodiversity', 'night sky', 'illumination'],
        'F': ['green space', 'urban', 'park', 'city', 'livability'],
    },
    2022: {
        'A': ['cyclist', 'bicycle', 'power', 'pedal', 'riding'],
        'B': ['water level', 'bath', 'bathtub', 'flood', 'dam'],
        'C': ['trading', 'cryptocurrency', 'bitcoin', 'gold', 'portfolio'],
        'D': ['power grid', 'network', 'energy', 'transmission'],
        'E': ['forest', 'fire', 'wildfire', 'carbon', 'sequestration'],
        'F': ['migration', 'population movement', 'immigration', 'refugees'],
    },
    2021: {
        'A': ['mushroom', 'fungi', 'mycelium', 'decomposition'],
        'B': ['drone', 'fire', 'disaster', 'emergency', 'relief'],
        'C': ['confirm', 'COVID', 'covid-19', 'pandemic', 'infection'],
        'D': ['music', 'influence', 'network', 'artist', 'genre'],
        'E': ['water', 'scarcity', 'reservoir', 'drought', 'supply'],
        'F': ['gig economy', 'freelance', 'worker', 'platform', 'labor'],
    },
    2020: {
        'A': ['football', 'pass', 'strategy', 'soccer', 'team'],
        'B': ['sandcastle', 'sand', 'water', 'erosion', 'shape'],
        'C': ['wealth', 'income', 'inequality', 'gini', 'distribution'],
        'D': ['football', 'soccer', 'team', 'network', 'passing'],
        'E': ['plastic', 'ocean', 'waste', 'pollution', 'microplastic'],
        'F': ['e-scooter', 'mobility', 'transportation', 'city', 'sharing'],
    },
}

# 题目类型通用关键词（不按年份变化的固有特征）
PROBLEM_GENERIC_KEYWORDS = {
    'A': [
        'continuous', 'differential equation', 'ode', 'pde', 'optimization', 
        'numerical method', 'simulation', 'physics', 'dynamics', 'heat transfer',
        'fluid', 'mechanics', 'mathematical model', 'continuous model',
        'boundary condition', 'finite element', 'calculus',
    ],
    'B': [
        'discrete', 'graph theory', 'combinatorial', 'network flow', 
        'integer programming', 'scheduling', 'routing', 'allocation',
        'algorithm', 'heuristic', 'greedy', 'dynamic programming',
        'assignment', 'matching', 'tree', 'spanning',
    ],
    'C': [
        'data analysis', 'machine learning', 'regression', 'time series',
        'classification', 'clustering', 'prediction', 'visualization',
        'dataset', 'feature engineering', 'cross validation', 'random forest',
        'neural network', 'deep learning', 'big data', 'correlation',
        'statistical analysis', 'data-driven',
    ],
    'D': [
        'network science', 'operations research', 'supply chain',
        'complex network', 'centrality', 'resilience', 'robustness',
        'system dynamics', 'simulation', 'stochastic', 'queuing',
        'sustainability', 'development', 'cooperation',
    ],
    'E': [
        'environment', 'ecosystem', 'biodiversity', 'climate change',
        'sustainability', 'carbon', 'emission', 'pollution', 'ecology',
        'conservation', 'renewable', 'habitat', 'species',
        'global warming', 'deforestation', 'water quality',
    ],
    'F': [
        'policy', 'society', 'social', 'economic model', 'governance',
        'regulation', 'equity', 'stakeholder', 'public policy',
        'game theory', 'agent-based model', 'system dynamics',
        'public health', 'education', 'inequality', 'welfare',
    ],
}


class ProblemDetector:
    """
    自动识别论文对应的 MCM/ICM 题目
    """
    
    def __init__(self):
        pass
    
    def detect(self, full_text: str, pdf_path: str = None, 
               year: int = None) -> Dict:
        """
        综合检测论文对应的题目
        
        参数:
            full_text: 论文全文文本
            pdf_path: PDF文件路径（可选，用于从路径提取信息）
            year: 年份（可选，如已知）
        
        返回:
            {
                'problem': 'A',           # 最可能的题目
                'contest': 'MCM',          # MCM 或 ICM
                'confidence': 0.85,        # 置信度 (0-1)
                'scores': {'A': 0.85, 'B': 0.05, ...},  # 各题目得分
                'detection_method': 'content_keyword',   # 检测方法
                'year': 2024,              # 检测到的年份
            }
        """
        result = {
            'problem': None,
            'contest': None,
            'confidence': 0.0,
            'scores': {},
            'detection_method': 'unknown',
            'year': year,
        }
        
        # 方法1: 从文件路径检测（最可靠）
        if pdf_path:
            path_result = self._detect_from_path(pdf_path)
            if path_result['problem'] and path_result['confidence'] > 0.9:
                result.update(path_result)
                if not result['year']:
                    result['year'] = path_result.get('year')
                return result
        
        # 方法2: 从论文文本中的直接引用检测
        ref_result = self._detect_from_references(full_text)
        if ref_result['problem'] and ref_result['confidence'] > 0.7:
            result.update(ref_result)
            return result
        
        # 方法3: 基于内容关键词匹配
        keyword_result = self._detect_from_keywords(full_text, year)
        result.update(keyword_result)
        
        # 确定赛道
        if result['problem']:
            result['contest'] = 'MCM' if result['problem'] in 'ABC' else 'ICM'
        
        return result
    
    def _detect_from_path(self, pdf_path: str) -> Dict:
        """从文件路径中检测题目信息"""
        result = {'problem': None, 'confidence': 0.0, 'detection_method': 'path', 'year': None}
        
        path_parts = Path(pdf_path).parts
        
        for part in path_parts:
            # 匹配 "MCM_A", "ICM_D" 等
            match = re.match(r'(MCM|ICM)[_\s]([A-F])', part, re.IGNORECASE)
            if match:
                result['contest'] = match.group(1).upper()
                result['problem'] = match.group(2).upper()
                result['confidence'] = 0.95
                break
            
            # 匹配单独的题目字母 "A", "B" 等（在年份目录下）
            if len(part) == 1 and part.upper() in 'ABCDEF':
                result['problem'] = part.upper()
                result['contest'] = 'MCM' if part.upper() in 'ABC' else 'ICM'
                result['confidence'] = 0.8
        
        # 提取年份
        for part in path_parts:
            if part.isdigit() and 2010 <= int(part) <= 2030:
                result['year'] = int(part)
                break
        
        return result
    
    def _detect_from_references(self, text: str) -> Dict:
        """从论文中对题目的直接引用检测"""
        result = {'problem': None, 'confidence': 0.0, 'detection_method': 'text_reference'}
        
        # 取论文前3000字符（题目引用通常在开头）
        header = text[:3000].lower()
        
        # 模式匹配
        patterns = [
            r'problem\s+([a-f])\b',
            r'mcm\s+problem\s+([a-f])\b',
            r'icm\s+problem\s+([a-f])\b',
            r'question\s+([a-f])\b',
            r'mcm[/-]?([a-f])\b',
            r'icm[/-]?([a-f])\b',
            r'题目\s*([a-f])\b',
        ]
        
        detected = Counter()
        for pattern in patterns:
            for match in re.finditer(pattern, header):
                letter = match.group(1).upper()
                if letter in 'ABCDEF':
                    detected[letter] += 1
        
        if detected:
            best = detected.most_common(1)[0]
            result['problem'] = best[0]
            result['confidence'] = min(0.9, 0.5 + best[1] * 0.15)
            result['contest'] = 'MCM' if best[0] in 'ABC' else 'ICM'
        
        return result
    
    def _detect_from_keywords(self, text: str, year: int = None) -> Dict:
        """基于关键词匹配检测题目"""
        result = {
            'problem': None, 
            'confidence': 0.0, 
            'scores': {},
            'detection_method': 'content_keyword',
        }
        
        text_lower = text.lower()
        scores = {}
        
        # 1. 年份特定关键词匹配（权重更高）
        if year and year in PROBLEM_KEYWORDS_BY_YEAR:
            year_keywords = PROBLEM_KEYWORDS_BY_YEAR[year]
            for problem, keywords in year_keywords.items():
                score = 0
                for kw in keywords:
                    count = text_lower.count(kw.lower())
                    if count > 0:
                        score += min(count, 5) * 2.0  # 年份特定关键词权重 ×2
                scores[problem] = scores.get(problem, 0) + score
        
        # 2. 通用关键词匹配
        for problem, keywords in PROBLEM_GENERIC_KEYWORDS.items():
            score = 0
            for kw in keywords:
                count = text_lower.count(kw.lower())
                if count > 0:
                    score += min(count, 10) * 1.0  # 通用关键词权重 ×1
            scores[problem] = scores.get(problem, 0) + score
        
        # 3. 归一化分数
        total = sum(scores.values())
        if total > 0:
            for p in scores:
                scores[p] /= total
        
        result['scores'] = scores
        
        # 找最高分的题目
        if scores:
            best = max(scores, key=scores.get)
            result['problem'] = best
            result['confidence'] = scores[best]
            result['contest'] = 'MCM' if best in 'ABC' else 'ICM'
        
        return result
    
    def detect_year(self, text: str, pdf_path: str = None) -> Optional[int]:
        """
        从文本或路径中检测年份
        """
        # 先从路径检测
        if pdf_path:
            path_parts = Path(pdf_path).parts
            for part in path_parts:
                if part.isdigit() and 2010 <= int(part) <= 2030:
                    return int(part)
        
        # 从文本检测（通常在标题或摘要中提到年份）
        header = text[:2000]
        # 寻找 MCM/ICM + 年份 的模式
        patterns = [
            r'(?:MCM|ICM|COMAP)\s*(\d{4})',
            r'(\d{4})\s*(?:MCM|ICM|COMAP)',
            r'(?:contest|competition)\s*(\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, header, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 2010 <= year <= 2030:
                    return year
        
        return None


def detect_problem(full_text: str, pdf_path: str = None, 
                   year: int = None) -> Dict:
    """
    便捷函数：自动检测论文对应的题目
    
    参数:
        full_text: 论文全文
        pdf_path: PDF路径
        year: 年份
    
    返回:
        检测结果字典
    """
    detector = ProblemDetector()
    return detector.detect(full_text, pdf_path, year)


if __name__ == "__main__":
    # 测试
    sample_text = """
    MCM Problem A: Continuous model for sea lamprey population dynamics...
    We develop a differential equation model to predict the population of sea lamprey 
    in the Great Lakes region. Our continuous optimization approach uses numerical 
    simulation to find optimal control strategies.
    """
    
    result = detect_problem(sample_text, year=2024)
    print(f"检测结果: 题目={result['problem']}, 赛道={result['contest']}, "
          f"置信度={result['confidence']:.2f}")
    print(f"各题目得分: {result['scores']}")
