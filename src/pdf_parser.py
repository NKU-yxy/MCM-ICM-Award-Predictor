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
        1. 遍历每一页，提取嵌入的位图（PNG/JPG）
        2. 过滤太小的图片（可能是 logo 或装饰）
        3. 过滤纯色图片（可能是背景遮罩）
        4. 过滤极端宽高比图片（可能是分隔线或页眉装饰）
        5. 对矢量图密集但嵌入图少的页面，渲染后边缘检测提取图表区域
        """
        images = []
        embedded_total = 0
        filtered_total = 0
        rendered_total = 0
        pages_with_embeds = 0
        pages_supplemented = 0

        for page_num in range(len(doc)):
            if len(images) >= self.max_images:
                break

            page = doc[page_num]
            image_list = page.get_images()
            page_embedded = 0

            for img_index, img in enumerate(image_list):
                if len(images) >= self.max_images:
                    break

                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes))

                    if pil_image.width < self.min_image_size or pil_image.height < self.min_image_size:
                        filtered_total += 1
                        continue

                    aspect = pil_image.width / max(pil_image.height, 1)
                    if aspect > 15 or aspect < 1/15:
                        filtered_total += 1
                        continue

                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')

                    if self._is_trivial_image(pil_image):
                        filtered_total += 1
                        continue

                    images.append(pil_image)
                    page_embedded += 1
                    embedded_total += 1

                except Exception as e:
                    continue

            if page_embedded > 0:
                pages_with_embeds += 1

        # 矢量图补充：对嵌入图很少但有大量矢量绘图的页面，渲染并提取图表区域
        for page_num in range(len(doc)):
            if len(images) >= self.max_images:
                break

            page = doc[page_num]

            # 已经有不少嵌入图了，跳过
            embeds_on_page = sum(1 for img in page.get_images())
            if embeds_on_page >= 3:
                continue

            # 检查是否有大量矢量绘图
            try:
                drawings = page.get_drawings()
                if len(drawings) > 30:
                    rendered = self._extract_figures_from_page_pixmap(
                        page, existing_count=len(images)
                    )
                    if rendered:
                        images.extend(rendered)
                        rendered_total += len(rendered)
                        pages_supplemented += 1
            except Exception:
                pass

        if pages_supplemented > 0:
            print(f"  从 {len(doc)} 页中找到 {embedded_total + filtered_total} 个嵌入图像，"
                  f"过滤后保留 {embedded_total} 个，页面渲染补充 {rendered_total} 个（{pages_supplemented} 页矢量图密集）")
        else:
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
            return total_std < 2.0  # 方差极小 → 纯色（放宽以避免误杀灰度图/热力图）
        except Exception:
            return False

    def _extract_figures_from_page_pixmap(
        self, page, existing_count: int = 0
    ) -> List[Image.Image]:
        """
        将页面渲染为位图，通过滑动窗口边缘检测分割出图表区域。

        适用于 matplotlib/seaborn/Matlab 生成的矢量图密集页面，
        这类页面的图形不会出现在 get_images() 结果中。
        """
        images = []
        try:
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            gray_np = np.array(img.convert("L"), dtype=np.float32)

            # 边缘检测（简化 Sobel）
            dx = np.abs(np.diff(gray_np, axis=1))
            dy = np.abs(np.diff(gray_np, axis=0))
            edge_map = np.zeros_like(gray_np)
            edge_map[:, :-1] += dx
            edge_map[:-1, :] += dy

            # 滑动窗口检测高边缘密度区域
            step = 60
            window = 120
            h, w = edge_map.shape
            candidates = []

            for y in range(0, max(h - window, 1), step):
                for x in range(0, max(w - window, 1), step):
                    y_end = min(y + window, h)
                    x_end = min(x + window, w)
                    block = edge_map[y:y_end, x:x_end]
                    density = float(np.mean(block))
                    if density > 12.0:
                        candidates.append((x, y, x_end, y_end))

            # 合并重叠候选区域
            merged = self._merge_rects(candidates, img.width, img.height)

            for x1, y1, x2, y2 in merged:
                if len(images) + existing_count >= self.max_images:
                    break
                if x2 - x1 < self.min_image_size or y2 - y1 < self.min_image_size:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                if not self._is_trivial_image(crop):
                    images.append(crop)
        except Exception:
            pass
        return images

    @staticmethod
    def _merge_rects(rects, img_w, img_h):
        """合并重叠/相邻的矩形区域（用于图表区域去重）"""
        if not rects:
            return []

        rects = sorted(rects, key=lambda r: (r[0], r[1]))
        used = [False] * len(rects)
        merged = []

        for i, (x1, y1, x2, y2) in enumerate(rects):
            if used[i]:
                continue
            cx1, cy1, cx2, cy2 = x1, y1, x2, y2
            changed = True
            while changed:
                changed = False
                for j, (ox1, oy1, ox2, oy2) in enumerate(rects):
                    if used[j]:
                        continue
                    margin = 30
                    if (
                        cx1 - margin < ox2 and cx2 + margin > ox1
                        and cy1 - margin < oy2 and cy2 + margin > oy1
                    ):
                        cx1 = min(cx1, ox1)
                        cy1 = min(cy1, oy1)
                        cx2 = max(cx2, ox2)
                        cy2 = max(cy2, oy2)
                        used[j] = True
                        changed = True

            cx1 = max(0, cx1 - 10)
            cy1 = max(0, cy1 - 10)
            cx2 = min(img_w, cx2 + 10)
            cy2 = min(img_h, cy2 + 10)

            used[i] = True
            merged.append((cx1, cy1, cx2, cy2))

        return merged
    
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
        
        # 表格检测: 三种方式融合
        # 方式1: 标题行匹配
        table_caption_patterns = [
            r'(?mi)^\s*Table\s+\d+[\s.:：]',
            r'(?mi)^\s*TABLE\s+[IVXLCD]+[\s.:：]',
            r'(?mi)^\s*Tab\.?\s+\d+[\s.:：]',
        ]
        table_ids = set()
        for pat in table_caption_patterns:
            for m in re.finditer(pat, full_text):
                table_ids.add(m.group().strip().lower())

        # 方式2: 基于文本块位置的表格布局检测
        layout_table_count = self._detect_tables_by_layout(doc)

        # 方式3: 基于视觉线条网格的表格检测（三线表等）
        visual_table_count = self._detect_tables_by_lines(doc)

        # 综合: 标题匹配 + max(布局检测, 视觉检测)
        # 标题匹配是最可靠的（有明确 "Table X" 标题），布局/视觉补充无标题表格
        structure['table_count'] = max(len(table_ids), layout_table_count, visual_table_count)
        
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

        structure['figure_caption_count'] = len(figure_ids)
        
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

        # 深度内容质量检测（超越计数，关注"写得多好"）
        quality_patterns = {
            'has_assumption_justification': r'\b(assum(e|ption).{0,50}(justif|reasonab|valid|realistic|appropriate))',
            'has_model_comparison': r'\b(compar(e|ison|ing).{0,30}(model|approach|method)|alternative\s+model|baseline\s+model)',
            'has_error_analysis': r'\b(error\s+analysis|uncertainty\s+quantification|confidence\s+interval|RMSE|MAE|residual)',
            'has_dimensional_analysis': r'\b(dimensional\s+analysis|unit\s+analysis|dimensionless|normalization|standardization)',
            'has_convergence_check': r'\b(convergence|stability|robustness|grid\s+independence)',
            'has_data_preprocessing': r'\b(data\s+(cleaning|preprocessing|normalization|standardization)|outlier|missing\s+value)',
        }
        quality_count = 0
        for key, pattern in quality_patterns.items():
            found = bool(re.search(pattern, text_lower))
            structure[key] = found
            if found:
                quality_count += 1
        structure['quality_section_count'] = quality_count
        
        return structure

    def _detect_tables_by_layout(self, doc: fitz.Document) -> int:
        """
        通过文本块位置检测表格布局。

        表格特征: 多个文本行在列方向上对齐，行间距很小（< 8pt）。
        这与普通段落有明显区别——段落行间距大且有缩进变化。
        """
        total_tables = 0

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                text_spans = []
                for block in blocks:
                    if block["type"] == 0:
                        for line in block["lines"]:
                            spans_text = []
                            for span in line["spans"]:
                                if span["text"].strip():
                                    spans_text.append(span)
                            if spans_text:
                                bbox = line["bbox"]
                                text_spans.append({
                                    "x0": bbox[0], "y0": bbox[1],
                                    "x1": bbox[2], "y1": bbox[3],
                                })

                if len(text_spans) < 4:
                    continue

                text_spans.sort(key=lambda s: s["y0"])

                # 寻找连续紧凑行组（行间距 < 8pt = 可能是表格行）
                row_groups = []
                current = [text_spans[0]]

                for i in range(1, len(text_spans)):
                    gap = text_spans[i]["y0"] - text_spans[i - 1]["y1"]
                    if abs(gap) < 8:
                        current.append(text_spans[i])
                    else:
                        if len(current) >= 3:
                            row_groups.append(current)
                        current = [text_spans[i]]

                if len(current) >= 3:
                    row_groups.append(current)

                # 验证: 表格行应该有多列（同一行内有多个水平分离的文本块）
                for group in row_groups:
                    rows = {}
                    for span in group:
                        y_key = round(span["y0"] / 10) * 10
                        rows.setdefault(y_key, []).append(span)

                    multi_col_rows = sum(1 for s in rows.values() if len(s) >= 2)
                    if multi_col_rows >= 2 and multi_col_rows >= len(rows) * 0.5:
                        total_tables += 1
            except Exception:
                pass

        return total_tables

    def _detect_tables_by_lines(self, doc: fitz.Document) -> int:
        """
        通过水平线和垂直线形成的网格检测表格（三线表等）。

        三线表特征: 至少 2 条长水平线（顶线、底线、分隔线）。
        """
        total_tables = 0

        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                drawings = page.get_drawings()

                h_lines = []
                for d in drawings:
                    items = d.get("items", [])
                    for item in items:
                        if item[0] == "l":
                            x1, y1 = item[1], item[2]
                            x2, y2 = item[3], item[4]
                            length = abs(x2 - x1)
                            thickness = abs(y2 - y1)
                            if thickness < 3 and length > 50:
                                h_lines.append(length)

                # 至少 2 条长水平线 → 可能是三线表
                long_lines = [l for l in h_lines if l > 50]
                if len(long_lines) >= 2:
                    # 估算表格数: 每 3-4 条长水平线 = 1 个三线表
                    total_tables += max(1, len(long_lines) // 3)
            except Exception:
                pass

        return total_tables

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
