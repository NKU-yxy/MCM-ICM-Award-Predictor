"""
PDF 解析器模块
从论文 PDF 中提取摘要文本、图片和元数据
"""

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config


class PDFParser:
    """PDF 解析器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """初始化解析器"""
        self.config = load_config(config_path)
        self.pdf_config = self.config['pdf_parser']
        self.min_image_size = self.pdf_config['min_image_size']
        self.max_images = self.pdf_config['max_images_per_paper']
        self.abstract_keywords = self.pdf_config['abstract_keywords']
        self.intro_keywords = self.pdf_config['intro_keywords']
    
    def parse(self, pdf_path: str) -> Dict:
        """
        解析 PDF 文件
        
        返回：
        {
            'abstract': str,  # 摘要文本
            'full_text': str,  # 全文文本
            'images': List[PIL.Image],  # 提取的图片列表
            'metadata': Dict,  # 元数据
            'structure': Dict  # 论文结构分析
        }
        """
        try:
            doc = fitz.open(pdf_path)
            
            # 提取全文
            full_text = self._extract_full_text(doc)
            
            # 提取摘要
            abstract = self.extract_abstract(doc)
            
            # 提取图片
            images = self.extract_images(doc)
            
            # 提取元数据
            metadata = self.extract_metadata(doc, pdf_path)
            
            # 提取论文结构特征
            structure = self._analyze_paper_structure(full_text, doc)
            
            doc.close()
            
            return {
                'abstract': abstract,
                'full_text': full_text,
                'images': images,
                'metadata': metadata,
                'structure': structure,
                'success': True
            }
            
        except Exception as e:
            print(f"解析 PDF 失败 {pdf_path}: {e}")
            return {
                'abstract': '',
                'full_text': '',
                'images': [],
                'metadata': {},
                'structure': {},
                'success': False,
                'error': str(e)
            }
    
    def extract_abstract(self, doc: fitz.Document) -> str:
        """
        提取摘要文本
        
        策略：
        1. 查找 "Abstract" 关键词
        2. 提取到 "Introduction" 之前的文本
        3. 如果失败，返回第一页全部文本
        """
        # 获取前两页文本（摘要通常在前两页）
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text()
        
        # 方法1：通过关键词定位
        abstract = self._extract_by_keywords(text)
        
        if abstract:
            return abstract
        
        # 方法2：fallback - 返回第一页文本
        print("  未找到 Abstract 关键词，使用第一页全文")
        return doc[0].get_text()[:2000]  # 限制长度
    
    def _extract_by_keywords(self, text: str) -> Optional[str]:
        """通过关键词定位摘要"""
        text_lower = text.lower()
        
        # 查找 Abstract 开始位置
        abstract_start = -1
        for keyword in self.abstract_keywords:
            pos = text_lower.find(keyword.lower())
            if pos != -1:
                abstract_start = pos + len(keyword)
                break
        
        if abstract_start == -1:
            return None
        
        # 查找 Introduction 结束位置
        abstract_end = len(text)
        for keyword in self.intro_keywords:
            pos = text_lower.find(keyword.lower(), abstract_start)
            if pos != -1:
                abstract_end = min(abstract_end, pos)
        
        # 提取摘要
        abstract = text[abstract_start:abstract_end].strip()
        
        # 清理文本
        abstract = self._clean_text(abstract)
        
        return abstract if len(abstract) > 50 else None
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除页眉页脚（通常包含 Page, Team 等）
        lines = text.split('\n')
        lines = [line for line in lines if not re.match(r'^\s*(Page|Team|Control Number)', line, re.I)]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def extract_images(self, doc: fitz.Document) -> List[Image.Image]:
        """
        提取 PDF 中的所有图片
        
        策略：
        1. 遍历每一页
        2. 提取嵌入的图片
        3. 过滤太小的图片（可能是 logo 或装饰）
        4. 过滤纯白/纯黑/纯灰色图片（可能是背景遮罩）
        5. 过滤极端宽高比图片（可能是分隔线或页眉装饰）
        """
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                if len(images) >= self.max_images:
                    break
                
                try:
                    # 获取图片数据
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # 转换为 PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # 过滤太小的图片
                    if pil_image.width < self.min_image_size or pil_image.height < self.min_image_size:
                        continue
                    
                    # 过滤极端宽高比（分隔线、页眉装饰等）
                    aspect = pil_image.width / max(pil_image.height, 1)
                    if aspect > 15 or aspect < 1/15:
                        continue
                    
                    # 转换为 RGB（如果是 RGBA 或其他格式）
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')
                    
                    # 过滤纯色/近纯色图片（背景遮罩等）
                    if self._is_trivial_image(pil_image):
                        continue
                    
                    images.append(pil_image)
                    
                except Exception as e:
                    print(f"  提取图片失败 (页 {page_num}, 图 {img_index}): {e}")
                    continue
        
        print(f"  提取了 {len(images)} 张有效图片（已过滤装饰/纯色/极小图片）")
        return images
    
    def _is_trivial_image(self, img: Image.Image) -> bool:
        """
        判断图片是否为无意义的纯色/近纯色图片
        
        - 采样缩小到 20x20 检查像素方差
        - 方差极小说明是纯色/近纯色背景
        """
        try:
            small = img.resize((20, 20))
            pixels = np.array(small, dtype=np.float32)
            # 所有通道的标准差之和
            total_std = pixels.std()
            return total_std < 5.0  # 方差很小 → 纯色
        except Exception:
            return False
    
    def extract_metadata(self, doc: fitz.Document, pdf_path: str) -> Dict:
        """
        提取元数据
        
        包括：
        - 总页数
        - 文件名信息（队伍编号、获奖等级、题目等）
        - 参考文献数量
        """
        metadata = {
            'page_count': len(doc),
            'file_name': Path(pdf_path).name,
        }
        
        # 从路径提取年份、赛道、题目
        from src.utils import parse_problem_path, get_paper_info_from_filename
        
        path_info = parse_problem_path(pdf_path)
        metadata.update(path_info)
        
        file_info = get_paper_info_from_filename(Path(pdf_path).name)
        metadata.update(file_info)
        
        # 统计参考文献数量（简单方法：搜索 "References" 后的内容）
        last_page_text = doc[-1].get_text() if len(doc) > 0 else ""
        metadata['ref_count'] = self._count_references(last_page_text)
        
        return metadata
    
    def _extract_full_text(self, doc: fitz.Document) -> str:
        """提取PDF全文文本"""
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        return full_text
    
    def _analyze_paper_structure(self, full_text: str, doc: fitz.Document) -> Dict:
        """
        分析论文结构质量
        
        返回结构特征字典:
        - section_count: 检测到的标准节数
        - has_abstract: 是否有摘要
        - has_introduction: 是否有引言
        - has_methodology: 是否有方法论部分
        - has_results: 是否有结果部分
        - has_conclusion: 是否有结论
        - has_references: 是否有参考文献
        - has_appendix: 是否有附录
        - formula_count: 公式/方程估计数量
        - table_count: 表格估计数量
        - figure_caption_count: 图表标题数量
        - total_word_count: 总字数
        - paragraph_count: 段落数
        - avg_paragraph_length: 平均段落长度
        - structure_completeness: 结构完整度 (0-1)
        - citation_count: 文中引用次数
        """
        text_lower = full_text.lower()
        
        # 标准节检测
        section_patterns = {
            'has_abstract': r'\babstract\b',
            'has_introduction': r'\bintroduction\b',
            'has_methodology': r'\b(method|methodology|approach|model\s+formulation|mathematical\s+model)\b',
            'has_results': r'\b(results?|analysis|findings|discussion)\b',
            'has_conclusion': r'\b(conclusion|summary|concluding\s+remarks)\b',
            'has_references': r'\b(references?|bibliography)\b',
            'has_appendix': r'\b(appendix|appendices)\b',
        }
        
        structure = {}
        section_count = 0
        for key, pattern in section_patterns.items():
            found = bool(re.search(pattern, text_lower))
            structure[key] = found
            if found and key != 'has_appendix':  # 附录不算核心节
                section_count += 1
        
        structure['section_count'] = section_count
        
        # 公式检测: 寻找常见的公式模式
        # LaTeX公式: $...$, \[...\], equation环境等
        # 或者独立行带有 = 号的模式
        formula_patterns = [
            r'\$[^$]+\$',                          # 内联公式 $...$
            r'\\\[.*?\\\]',                         # 显示公式 \[...\]
            r'\\begin\{equation\}',                 # equation环境
            r'(?m)^\s*[\w\s]*\s*=\s*[^=]',          # 独立等式行
        ]
        formula_count = 0
        for pattern in formula_patterns:
            formula_count += len(re.findall(pattern, full_text))
        # 去重估计（很多重复匹配）
        formula_count = max(1, formula_count // 2) if formula_count > 0 else 0
        structure['formula_count'] = formula_count
        
        # 表格检测: 多种格式, 只匹配标题行（排除引用处的 "Table 1"）
        # "Table 1:", "Table 1.", "TABLE I", "Tab. 1" 等标题格式
        table_caption_patterns = [
            r'(?mi)^\s*Table\s+\d+[\s.:：]',              # 行首 Table 1: / Table 1.
            r'(?mi)^\s*TABLE\s+[IVXLCD]+[\s.:：]',         # 行首 TABLE I:
            r'(?mi)^\s*Tab\.?\s+\d+[\s.:：]',              # Tab. 1:
        ]
        table_ids = set()
        for pat in table_caption_patterns:
            for m in re.finditer(pat, full_text):
                # 提取标识符进行去重
                table_ids.add(m.group().strip().lower())
        
        # 补充: 计算文中所有不同编号的 Table 引用，取标题数和引用编号数的较大值
        table_ref_nums = set()
        for m in re.finditer(r'\bTable\s+(\d+)\b', full_text, re.IGNORECASE):
            table_ref_nums.add(m.group(1))
        
        structure['table_count'] = max(len(table_ids), len(table_ref_nums))
        
        # 图表标题检测: 多种格式, 行首匹配优先
        figure_caption_patterns = [
            r'(?mi)^\s*Figure\s+\d+[\s.:：]',             # 行首 Figure 1:
            r'(?mi)^\s*FIGURE\s+\d+[\s.:：]',             # 行首 FIGURE 1:
            r'(?mi)^\s*Fig\.?\s+\d+[\s.:：]',              # Fig. 1: / Fig 1.
        ]
        figure_ids = set()
        for pat in figure_caption_patterns:
            for m in re.finditer(pat, full_text):
                figure_ids.add(m.group().strip().lower())
        
        # 补充: 不同编号的 Figure 引用
        figure_ref_nums = set()
        for m in re.finditer(r'\b(?:Figure|Fig\.?)\s+(\d+)\b', full_text, re.IGNORECASE):
            figure_ref_nums.add(m.group(1))
        
        structure['figure_caption_count'] = max(len(figure_ids), len(figure_ref_nums))
        
        # 总字数
        words = full_text.split()
        structure['total_word_count'] = len(words)
        
        # 段落统计
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip() and len(p.strip()) > 30]
        structure['paragraph_count'] = len(paragraphs)
        structure['avg_paragraph_length'] = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0
        
        # 文中引用次数 [1], [2,3], [1-5] 等
        citations = re.findall(r'\[\d+(?:[,\-–]\s*\d+)*\]', full_text)
        structure['citation_count'] = len(citations)
        
        # 结构完整度: 核心节占比
        core_sections = ['has_abstract', 'has_introduction', 'has_methodology', 
                        'has_results', 'has_conclusion', 'has_references']
        present_count = sum(1 for s in core_sections if structure.get(s, False))
        structure['structure_completeness'] = present_count / len(core_sections)
        
        # 额外: 检测是否有sensitivity analysis, model validation等高级内容
        advanced_patterns = {
            'has_sensitivity_analysis': r'\bsensitivity\s+analysis\b',
            'has_model_validation': r'\b(model\s+validation|cross[- ]?validation|validation)\b',
            'has_strengths_weaknesses': r'\b(strength|weakness|limitation)\b',
            'has_future_work': r'\b(future\s+work|further\s+research|future\s+direction)\b',
        }
        advanced_count = 0
        for key, pattern in advanced_patterns.items():
            found = bool(re.search(pattern, text_lower))
            structure[key] = found
            if found:
                advanced_count += 1
        structure['advanced_section_count'] = advanced_count
        
        return structure
    
    def _count_references(self, text: str) -> int:
        """
        统计参考文献数量
        
        简单策略：统计 [1], [2] 或 1. 2. 这样的编号数量
        """
        # 查找 [1], [2], [3] ... 这样的引用
        pattern1 = r'\[\d+\]'
        matches1 = re.findall(pattern1, text)
        
        # 查找 1. 2. 3. ... 这样的编号（在 References 部分）
        pattern2 = r'^\s*\d+\.\s'
        matches2 = re.findall(pattern2, text, re.MULTILINE)
        
        return max(len(set(matches1)), len(matches2))
    
    def batch_parse(self, pdf_dir: str, recursive: bool = True) -> List[Dict]:
        """
        批量解析 PDF 文件
        
        参数：
            pdf_dir: PDF 文件目录
            recursive: 是否递归搜索子目录
        
        返回：
            解析结果列表
        """
        pdf_dir = Path(pdf_dir)
        
        if not pdf_dir.exists():
            print(f"目录不存在: {pdf_dir}")
            return []
        
        # 查找所有 PDF 文件
        if recursive:
            pdf_files = list(pdf_dir.rglob("*.pdf"))
        else:
            pdf_files = list(pdf_dir.glob("*.pdf"))
        
        print(f"找到 {len(pdf_files)} 个 PDF 文件")
        
        # 批量解析
        results = []
        for pdf_file in pdf_files:
            print(f"\n解析: {pdf_file.name}")
            result = self.parse(str(pdf_file))
            result['file_path'] = str(pdf_file)
            results.append(result)
        
        # 统计
        success_count = sum(1 for r in results if r['success'])
        print(f"\n解析完成: 成功 {success_count} / {len(pdf_files)}")
        
        return results


def extract_paper_content(pdf_path: str, config_path: str = "config.yaml") -> Dict:
    """
    便捷函数：提取单篇论文的内容
    
    参数:
        pdf_path: PDF文件路径
        config_path: 配置文件路径
    
    返回:
        dict: 包含abstract, images, metadata的字典
    """
    parser = PDFParser(config_path)
    return parser.parse(pdf_path)


def main():
    """测试 PDF 解析器"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python pdf_parser.py <pdf_file_or_dir>")
        return
    
    path = sys.argv[1]
    parser = PDFParser()
    
    if Path(path).is_file():
        # 解析单个文件
        result = parser.parse(path)
        
        print("\n" + "="*60)
        print("解析结果:")
        print("="*60)
        print(f"摘要 ({len(result['abstract'])} 字符):")
        print(result['abstract'][:500] + "...")
        print(f"\n图片数量: {len(result['images'])}")
        print(f"元数据: {result['metadata']}")
    
    else:
        # 批量解析
        results = parser.batch_parse(path)
        
        print("\n" + "="*60)
        print("批量解析统计:")
        print("="*60)
        
        for result in results:
            if result['success']:
                print(f"{result['metadata'].get('file_name', 'unknown')}: "
                      f"摘要 {len(result['abstract'])} 字符, "
                      f"{len(result['images'])} 张图片")


if __name__ == "__main__":
    main()
