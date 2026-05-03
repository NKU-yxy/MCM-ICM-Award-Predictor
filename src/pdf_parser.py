"""
PDF 解析器模块
从论文 PDF 中提取摘要文本、图片和元数据
"""

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import re
from contextlib import redirect_stderr, redirect_stdout
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
        self.max_images = int(os.environ.get("PDF_MAX_IMAGES", self.pdf_config['max_images_per_paper']))
        self.render_vector_figures = bool(self.pdf_config.get('render_vector_figures', False))
        self.max_image_pixels = int(self.pdf_config.get('max_image_pixels', 8000000))
        self.abstract_keywords = self.pdf_config['abstract_keywords']
        self.intro_keywords = self.pdf_config['intro_keywords']
        self._last_image_stats = {
            "filtered_image_count": 0,
            "rendered_vector_figure_count": 0,
        }
    
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
            abstract_info = self.extract_abstract_info(doc)
            abstract = abstract_info["text"]
            
            # 提取图片
            images = self.extract_images(doc)
            
            # 提取元数据
            metadata = self.extract_metadata(doc, pdf_path, full_text=full_text)
            metadata.update(
                {
                    "abstract_word_count": abstract_info["word_count"],
                    "abstract_extraction_method": abstract_info["method"],
                    "abstract_confidence": abstract_info["confidence"],
                    **self._last_image_stats,
                }
            )
            
            # 提取论文结构特征
            structure = self._analyze_paper_structure(full_text, doc)
            self._sync_abstract_structure_flag(
                structure,
                abstract,
                has_abstract_heading=abstract_info["confidence"] >= 0.5,
            )
            self._attach_visual_evidence_counts(metadata, structure)
            
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
            print(f"解析 PDF 失败 {Path(pdf_path).name}")
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
        """Return the best abstract candidate text."""
        return self.extract_abstract_info(doc)["text"]

    def extract_abstract_info(self, doc: fitz.Document) -> Dict:
        """Extract abstract text plus method, confidence and English token count."""
        text = "\n".join(doc[page_num].get_text() for page_num in range(min(2, len(doc))))
        candidates = []

        for match, label, method_base in self._iter_abstract_heading_matches(text):
            end_match = self._abstract_end_pattern().search(text, match.end())
            end = end_match.start() if end_match else len(text)
            candidate = self._clean_text(text[match.end():end])
            word_count = self._count_english_tokens(candidate)
            if word_count < 20:
                continue

            confidence = self._abstract_confidence(
                word_count=word_count,
                has_boundary=end_match is not None,
                method_base=method_base,
                label=label,
            )
            candidates.append(
                {
                    "text": candidate,
                    "word_count": word_count,
                    "method": f"{method_base}_{label.lower()}",
                    "confidence": confidence,
                }
            )

        if not candidates:
            summary_sheet_info = self._extract_summary_sheet_first_page_info(doc)
            if summary_sheet_info:
                return summary_sheet_info
            return {
                "text": "",
                "word_count": 0,
                "method": "not_found",
                "confidence": 0.0,
            }

        return max(candidates, key=lambda item: (item["confidence"], item["word_count"]))

    def _extract_by_keywords(self, text: str) -> Optional[str]:
        """Backward-compatible keyword extractor for older callers."""
        candidates = []
        for match, label, method_base in self._iter_abstract_heading_matches(text):
            end_match = self._abstract_end_pattern().search(text, match.end())
            end = end_match.start() if end_match else len(text)
            candidate = self._clean_text(text[match.end():end])
            word_count = self._count_english_tokens(candidate)
            if word_count >= 20:
                candidates.append(
                    (
                        self._abstract_confidence(
                            word_count=word_count,
                            has_boundary=end_match is not None,
                            method_base=method_base,
                            label=label,
                        ),
                        word_count,
                        candidate,
                    )
                )
        if not candidates:
            return None
        return max(candidates)[2]

    def _extract_summary_sheet_first_page_info(self, doc: fitz.Document) -> Optional[Dict]:
        """Use the MCM/ICM Summary Sheet first page as the abstract when no heading exists."""
        if len(doc) <= 0:
            return None

        text = doc[0].get_text() or ""
        if not self._looks_like_mcm_summary_sheet(text):
            return None

        candidate = self._clean_summary_sheet_text(text)
        word_count = self._count_english_tokens(candidate)
        if word_count < 20:
            return None

        confidence = 0.82 if 250 <= word_count <= 650 else 0.72
        if word_count < 80 or word_count > 850:
            confidence -= 0.12
        return {
            "text": candidate,
            "word_count": word_count,
            "method": "summary_sheet_first_page",
            "confidence": round(max(0.0, min(confidence, 0.95)), 2),
        }

    @staticmethod
    def _looks_like_mcm_summary_sheet(text: str) -> bool:
        """Return true only for contestant summary sheets, not COMAP problem statements."""
        text = text or ""
        has_summary_sheet = bool(re.search(r"\bsummary\s+sheet\b", text, re.I))
        has_team_control = bool(
            re.search(r"\bteam\s+control\s+number\b", text, re.I)
            or re.search(r"\bteam\s*#\s*\d{5,8}\b", text, re.I)
        )
        has_problem_chosen = bool(re.search(r"\bproblem\s+chosen\b", text, re.I))
        return has_summary_sheet and (has_team_control or has_problem_chosen)

    def _clean_summary_sheet_text(self, text: str) -> str:
        """Remove contest cover metadata from the first-page Summary Sheet."""
        lines = []
        skip_next_team_id = False

        for raw_line in (text or "").splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue

            lower = line.lower()
            if re.match(r"^team\s*#\s*\d+", line, re.I):
                continue
            if re.match(r"^page\s+\d+\s+of\s+\d+", line, re.I):
                continue
            if re.match(r"^problem\s+chosen\b", line, re.I):
                continue
            if re.match(r"^team\s+control\s+number\b", line, re.I):
                skip_next_team_id = True
                continue
            if skip_next_team_id and re.fullmatch(r"\d{5,8}", line):
                skip_next_team_id = False
                continue
            skip_next_team_id = False

            if lower in {"mcm/icm", "summary sheet"}:
                continue
            if "summary sheet" in lower and len(line.split()) <= 5:
                continue
            if re.fullmatch(r"[A-F]", line):
                continue
            if re.fullmatch(r"20\d{2}", line):
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", line):
                continue

            lines.append(line)

        return self._clean_text("\n".join(lines))

    @staticmethod
    def _iter_abstract_heading_matches(text: str):
        exact = re.compile(r"(?im)^\s*(abstract|summary)\s*[:：]?\s*$")
        inline = re.compile(r"(?im)^\s*(abstract|summary)\s*[:：]\s+(?=\S)")

        matches = []
        for match in exact.finditer(text or ""):
            matches.append((match.start(), match, match.group(1), "heading"))
        for match in inline.finditer(text or ""):
            matches.append((match.start(), match, match.group(1), "inline_heading"))

        for _, match, label, method_base in sorted(matches, key=lambda item: item[0]):
            line_start = (text or "").rfind("\n", 0, match.start()) + 1
            line_end = (text or "").find("\n", match.end())
            if line_end == -1:
                line_end = len(text or "")
            line = (text or "")[line_start:line_end].strip().lower()
            if "summary sheet" in line or "control number" in line or "team" in line:
                continue
            yield match, label, method_base

    @staticmethod
    def _abstract_end_pattern():
        return re.compile(
            r"(?im)^\s*(?:keywords?|index\s+terms?|table\s+of\s+contents|contents|"
            r"introduction|(?:\d+|[ivxlcdm]+)\.?\s+(?:introduction|problem|"
            r"restatement|assumptions?|notation|model)|problem\s+restatement|"
            r"restatement\s+of\s+the\s+problem|assumptions?|notations?|"
            r"model\s+(?:overview|formulation|establishment))\b"
        )

    @staticmethod
    def _count_english_tokens(text: str) -> int:
        return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text or ""))

    @staticmethod
    def _abstract_confidence(
        *,
        word_count: int,
        has_boundary: bool,
        method_base: str,
        label: str,
    ) -> float:
        confidence = 0.72 if method_base == "heading" else 0.62
        if has_boundary:
            confidence += 0.12
        if 250 <= word_count <= 650:
            confidence += 0.10
        elif 80 <= word_count <= 850:
            confidence += 0.05
        elif word_count < 50:
            confidence -= 0.25
        else:
            confidence -= 0.08
        if label.lower() == "summary":
            confidence -= 0.02
        return round(max(0.0, min(confidence, 0.98)), 2)

    @staticmethod
    def _sync_abstract_structure_flag(
        structure: Dict,
        abstract: str,
        *,
        has_abstract_heading: bool = True,
    ) -> None:
        """Treat either Abstract or Summary extraction as a valid abstract section."""
        if not has_abstract_heading or PDFParser._count_english_tokens(abstract or "") < 50:
            return

        was_present = bool(structure.get("has_abstract"))
        structure["has_abstract"] = True

        core_sections = [
            "has_abstract",
            "has_introduction",
            "has_methodology",
            "has_results",
            "has_conclusion",
            "has_references",
        ]
        if not was_present:
            structure["section_count"] = sum(
                1 for key in core_sections if structure.get(key, False)
            )
        structure["structure_completeness"] = sum(
            1 for key in core_sections if structure.get(key, False)
        ) / len(core_sections)

    @staticmethod
    def _has_abstract_or_summary_heading(doc: fitz.Document) -> bool:
        """Detect an extractable Abstract/Summary heading or MCM/ICM Summary Sheet."""
        text = ""
        for page_num in range(min(2, len(doc))):
            text += doc[page_num].get_text() + "\n"
        return any(PDFParser._iter_abstract_heading_matches(text)) or PDFParser._looks_like_mcm_summary_sheet(text)
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # Join PDF line-break hyphenation before whitespace normalization.
        text = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])', '', text)
        text = re.sub(r'(?<=[A-Za-z])-\s+(?=[a-z])', '', text)
        text = re.sub(r'(?<=\d)-\s+(?=[A-Za-z])', ' ', text)
        # 移除页眉页脚（通常包含 Page, Team 等）
        lines = []
        for line in text.splitlines():
            line = re.sub(r'\s+', ' ', line).strip()
            if not line:
                continue
            if re.match(r'^(Page|Team|Control Number|Problem Chosen)\b', line, re.I):
                continue
            if re.match(r'^\d+\s*/\s*\d+$', line):
                continue
            lines.append(line)

        return re.sub(r'\s+', ' ', ' '.join(lines)).strip()
    
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
        self._last_image_stats = {
            "filtered_image_count": 0,
            "rendered_vector_figure_count": 0,
        }

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
                    width = int(base_image.get("width", 0) or 0)
                    height = int(base_image.get("height", 0) or 0)
                    if width * height > self.max_image_pixels:
                        filtered_total += 1
                        continue
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

        if not self.render_vector_figures:
            print(f"  Extracted {len(images)} embedded figures (vector page rendering disabled)")
            self._last_image_stats = {
                "filtered_image_count": len(images),
                "rendered_vector_figure_count": 0,
            }
            return images

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

        self._last_image_stats = {
            "filtered_image_count": len(images),
            "rendered_vector_figure_count": rendered_total,
        }
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

    @staticmethod
    def _attach_visual_evidence_counts(metadata: Dict, structure: Dict) -> None:
        """Copy parse counts into metadata and synthesize scoring-safe visual evidence."""
        filtered_image_count = int(metadata.get("filtered_image_count", 0) or 0)
        rendered_vector_count = int(metadata.get("rendered_vector_figure_count", 0) or 0)
        figure_caption_count = int(structure.get("figure_caption_count", 0) or 0)
        table_caption_count = int(structure.get("table_caption_count", 0) or 0)
        pymupdf_table_count = int(structure.get("pymupdf_table_count", 0) or 0)
        table_count = int(structure.get("table_count", 0) or 0)
        visual_evidence_count = max(
            figure_caption_count,
            filtered_image_count,
            rendered_vector_count,
        ) + table_count

        metadata.update(
            {
                "image_count": visual_evidence_count,
                "visual_evidence_count": visual_evidence_count,
                "figure_caption_count": figure_caption_count,
                "table_caption_count": table_caption_count,
                "pymupdf_table_count": pymupdf_table_count,
                "rendered_vector_figure_count": rendered_vector_count,
            }
        )
        structure["visual_evidence_count"] = visual_evidence_count
    
    def extract_metadata(
        self,
        doc: fitz.Document,
        pdf_path: str,
        full_text: Optional[str] = None,
    ) -> Dict:
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
        metadata['raw_image_count'] = sum(len(page.get_images()) for page in doc)
        
        # 从路径提取年份、赛道、题目
        from src.utils import parse_problem_path, get_paper_info_from_filename
        
        path_info = parse_problem_path(pdf_path)
        metadata.update(path_info)
        
        file_info = get_paper_info_from_filename(Path(pdf_path).name)
        metadata.update(file_info)
        
        # 统计参考文献数量。优先使用全文中的 References/Bibliography 区段；
        # 旧逻辑只看最后一页，遇到跨页参考文献时容易误报 0。
        reference_source = full_text
        if reference_source is None:
            start_page = max(0, len(doc) - 3)
            reference_source = "\n".join(doc[i].get_text() for i in range(start_page, len(doc)))
        metadata['ref_count'] = self._count_references(reference_source)
        
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
            'has_abstract': r'(?m)^\s*(abstract|summary)\b(?!\s+sheet)\s*[:：]?\s*$',
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
        
        caption_ids = self._extract_caption_ids(full_text)
        table_caption_count = len(caption_ids["table"])
        figure_caption_count = len(caption_ids["figure"])
        pymupdf_table_count = self._detect_tables_by_pymupdf(doc)
        structure['table_caption_count'] = table_caption_count
        structure['pymupdf_table_count'] = pymupdf_table_count
        structure['table_count'] = max(table_caption_count, pymupdf_table_count)
        structure['figure_caption_count'] = figure_caption_count
        
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

    @staticmethod
    def _extract_caption_ids(full_text: str) -> Dict[str, set]:
        """Detect Figure/Fig./Table/Tab. captions and dedupe by normalized number."""
        caption_pattern = re.compile(
            r"(?mi)^\s*(figure|fig\.?|table|tab\.?)\s*"
            r"([A-Za-z]?\d+(?:\s*[\(\[]\s*[A-Za-z0-9]+\s*[\)\]])?"
            r"(?:[A-Za-z])?|[IVXLCDM]+)(?=[\s.:：\)-]|$)"
        )
        ids = {"figure": set(), "table": set()}
        for match in caption_pattern.finditer(full_text or ""):
            label = match.group(1).lower().rstrip(".")
            number = re.sub(r"\s+", "", match.group(2).lower())
            kind = "figure" if label in {"figure", "fig"} else "table"
            ids[kind].add(number)
        return ids

    def _detect_tables_by_pymupdf(self, doc: fitz.Document) -> int:
        """Use PyMuPDF table detection with line and text strategies."""
        table_boxes = set()
        strategies = (("lines", {}), ("text", {"strategy": "text"}))

        for page_num in range(len(doc)):
            page = doc[page_num]
            for strategy_name, kwargs in strategies:
                try:
                    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                        tables_obj = page.find_tables(**kwargs)
                except Exception:
                    continue

                tables = getattr(tables_obj, "tables", tables_obj)
                for table in tables or []:
                    bbox = getattr(table, "bbox", None)
                    if not bbox:
                        continue
                    rect = fitz.Rect(bbox)
                    if rect.width < 40 or rect.height < 20:
                        continue

                    row_count = int(getattr(table, "row_count", 0) or 0)
                    col_count = int(getattr(table, "col_count", 0) or 0)
                    if row_count and row_count < 2:
                        continue
                    if col_count and col_count < 2:
                        continue
                    if strategy_name == "text":
                        if rect.height > page.rect.height * 0.40:
                            continue
                        if col_count > 12:
                            continue

                    key = (
                        page_num,
                        round(rect.x0 / 4) * 4,
                        round(rect.y0 / 4) * 4,
                        round(rect.x1 / 4) * 4,
                        round(rect.y1 / 4) * 4,
                    )
                    table_boxes.add(key)

        return len(table_boxes)

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
        
        统计 References/Bibliography 区段中的 [1]、1.、1) 等编号；
        对 APA/作者-年份等非编号格式，用年份/DOI/URL/期刊线索做保守估计。
        """
        has_reference_heading = bool(
            re.search(
                r'(?im)^\s*(references?|bibliography|works\s+cited|literature\s+cited)\s*$',
                text or "",
            )
        )
        section = self._reference_section_text(text)
        if not section.strip():
            return 0

        numbered_counts = [
            len(set(re.findall(r'(?m)^\s*\[\s*(\d{1,3})\s*\]', section))),
            len(set(re.findall(r'(?m)^\s*(\d{1,3})\s*[\.\)]\s+', section))),
        ]
        numbered_count = max(numbered_counts)
        if numbered_count > 0:
            return numbered_count

        lines = [
            re.sub(r'\s+', ' ', line).strip()
            for line in section.splitlines()
            if len(line.strip()) >= 12
        ]
        author_year_entries = 0
        for line in lines:
            has_year = re.search(r'\b(?:19|20)\d{2}[a-z]?\b', line)
            has_reference_marker = re.search(
                r'\b(doi|https?://|journal|proceedings|conference|press|vol\.|pp\.|arxiv|retrieved)\b',
                line,
                re.IGNORECASE,
            )
            starts_like_citation = re.search(
                r'^[A-Z][A-Za-z\'’\-]+,\s+(?:[A-Z]\.\s*)+',
                line,
            ) or re.search(
                r'^[A-Z][A-Za-z\'’\-]+(?:\s+and\s+|,\s+)[A-Z][A-Za-z\'’\-]+',
                line,
            )
            if has_year and (has_reference_marker or starts_like_citation):
                author_year_entries += 1

        if author_year_entries > 0:
            return author_year_entries

        doi_or_url_count = len(re.findall(r'\b(?:doi:\s*10\.|10\.\d{4,9}/|https?://)', section, re.IGNORECASE))
        if doi_or_url_count > 0:
            return doi_or_url_count

        years = re.findall(r'\b(?:19|20)\d{2}[a-z]?\b', section)
        return len(years) if has_reference_heading and len(years) >= 2 else 0

    @staticmethod
    def _reference_section_text(text: str) -> str:
        if not text:
            return ""

        headings = list(
            re.finditer(
                r'(?im)^\s*(references?|bibliography|works\s+cited|literature\s+cited)\s*$',
                text,
            )
        )
        if headings:
            start = headings[-1].end()
            tail = text[start:]
        else:
            tail = text[-12000:]

        end_match = re.search(
            r'(?im)^\s*(appendix|appendices|acknowledg(?:e)?ments?|supporting\s+materials?)\b',
            tail,
        )
        if end_match:
            tail = tail[:end_match.start()]
        return tail
    
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
