"""
MCM/ICM 论文获奖预测 — Web 服务

安全特性:
- 文件类型白名单 (仅 PDF)
- 文件大小限制 (20MB)
- 请求频率限制 (IP 级别)
- 临时文件自动清理
- 安全响应头 (CSP, X-Frame-Options, etc.)
- 输入消毒

启动: uvicorn web_app:app --host 0.0.0.0 --port 8000
生产: 通过 nginx 反向代理 + HTTPS 终端
"""

import os
import sys
import uuid
import time
import json
import logging
import hashlib
import tempfile
import threading
import re
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# --- project path ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_parser import PDFParser
from src.image_features import ImageFeatureExtractor
from src.llm_rubric_scorer import DeepSeekRubricScorer
from src.problem_detector import ProblemDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("mcm-web")

# ============================================================================
# Constants
# ============================================================================
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_MIME = {"application/pdf"}
TEMP_DIR = Path(tempfile.gettempdir()) / "mcm_predictor"
TEMP_DIR.mkdir(mode=0o700, exist_ok=True)
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Rate limiting
RATE_LIMIT_WINDOW = 60 * 60       # seconds
RATE_LIMIT_MAX_REQUESTS = 5       # per identity per hour
rate_limit_store: dict[str, list[float]] = defaultdict(list)
rate_limit_lock = threading.Lock()

# Public deployment settings
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "0").strip().lower() in {"1", "true", "yes"}
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "0").strip().lower() in {"1", "true", "yes"}
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
    if host.strip()
]

# Lightweight evaluator singletons
_evaluator_lock = threading.Lock()
_pdf_parser: PDFParser | None = None
_image_extractor: ImageFeatureExtractor | None = None
_rubric_scorer: DeepSeekRubricScorer | None = None
_problem_detector: ProblemDetector | None = None

# Usage statistics (in-memory, resets on restart)
STATS_FILE = Path(__file__).parent / "data" / "usage_stats_deepseek.json"
_stats_lock = threading.Lock()
_usage_stats: dict = {
    "total_predictions": 0,
    "today_predictions": 0,
    "today_date": str(date.today()),
    "award_counts": {"O": 0, "F": 0, "M": 0, "H": 0, "S": 0},
    "recent_scores": [],  # [{score, problem, timestamp}, ...] max 50
}


def _load_stats():
    """从磁盘加载持久化统计"""
    global _usage_stats
    try:
        if STATS_FILE.exists():
            saved = json.loads(STATS_FILE.read_text("utf-8"))
            _usage_stats.update(saved)
            # 检查日期
            if _usage_stats.get("today_date") != str(date.today()):
                _usage_stats["today_predictions"] = 0
                _usage_stats["today_date"] = str(date.today())
    except Exception:
        pass


def _write_stats_to_disk(stats_copy: dict):
    """将统计写入磁盘（无锁，仅供异步调用）"""
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(stats_copy, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass


def record_prediction(score: float, problem: str, best_award: str):
    """线程安全地记录一次预测，立即持久化到磁盘。"""
    best_award = "S" if best_award in {"S/U", "U"} else best_award
    with _stats_lock:
        _usage_stats["total_predictions"] += 1
        if _usage_stats.get("today_date") == str(date.today()):
            _usage_stats["today_predictions"] += 1
        else:
            _usage_stats["today_predictions"] = 1
            _usage_stats["today_date"] = str(date.today())
        if best_award in _usage_stats["award_counts"]:
            _usage_stats["award_counts"][best_award] += 1
        _usage_stats["recent_scores"].append({
            "score": round(score, 1),
            "problem": problem,
            "timestamp": datetime.now().isoformat(timespec="minutes"),
        })
        if len(_usage_stats["recent_scores"]) > 50:
            _usage_stats["recent_scores"] = _usage_stats["recent_scores"][-50:]
        # 拷贝快照，释放锁后再写盘，避免 I/O 阻塞事件循环
        stats_snapshot = {
            "total_predictions": _usage_stats["total_predictions"],
            "today_predictions": _usage_stats["today_predictions"],
            "today_date": _usage_stats["today_date"],
            "award_counts": dict(_usage_stats["award_counts"]),
            "recent_scores": list(_usage_stats["recent_scores"]),
        }
    # 锁外写盘，不阻塞 stats 查询
    _write_stats_to_disk(stats_snapshot)


def public_rubric_payload(rubric: dict) -> dict:
    """Return only fields needed by the browser, excluding provider fingerprinting."""
    allowed = {
        "status",
        "score",
        "details",
        "award_prediction",
        "probabilities",
        "strengths",
        "weaknesses",
        "comments",
    }
    return {key: value for key, value in (rubric or {}).items() if key in allowed}


def get_evaluators():
    """线程安全的 AI 评估组件单例。"""
    global _pdf_parser, _image_extractor, _rubric_scorer, _problem_detector
    if _pdf_parser is None:
        with _evaluator_lock:
            if _pdf_parser is None:
                logger.info("初始化 AI 评估组件...")
                _pdf_parser = PDFParser()
                _image_extractor = ImageFeatureExtractor()
                _rubric_scorer = DeepSeekRubricScorer()
                _problem_detector = ProblemDetector()
                logger.info("AI 评估组件初始化完成")
    return _pdf_parser, _image_extractor, _rubric_scorer, _problem_detector


def detect_summary_sheet_problem(full_text: str) -> str | None:
    """Read the official MCM/ICM summary sheet problem marker when present."""
    match = re.search(r"(?is)\bProblem\s+Chosen\s*[:：]?\s*([A-F])\b", full_text or "")
    return match.group(1).upper() if match else None


# ============================================================================
# Security Middleware
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """添加安全响应头"""

    async def dispatch(self, request: Request, call_next):
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
        if REQUIRE_HTTPS and request.url.path != "/health":
            is_https = request.url.scheme == "https" or (
                TRUST_PROXY_HEADERS and forwarded_proto.split(",")[0].strip() == "https"
            )
            if not is_https:
                return JSONResponse({"detail": "HTTPS required"}, status_code=403)

        response = await call_next(request)
        # 防止 MIME 嗅探
        response.headers["X-Content-Type-Options"] = "nosniff"
        # 防止点击劫持
        response.headers["X-Frame-Options"] = "DENY"
        # 内容安全策略
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if request.url.scheme == "https" or (
            TRUST_PROXY_HEADERS and forwarded_proto.split(",")[0].strip() == "https"
        ):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """IP 级别请求频率限制 (仅限制 POST /api/predict)"""

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """获取客户端 IP；只有显式信任反向代理时才读取 X-Forwarded-For。"""
        forwarded = request.headers.get("X-Forwarded-For") if TRUST_PROXY_HEADERS else None
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    @classmethod
    def _client_identity(cls, request: Request) -> str:
        ip = cls._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "")[:200]
        digest = hashlib.sha256(f"{ip}|{user_agent}".encode("utf-8")).hexdigest()[:16]
        return digest

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/api/predict" and request.method == "POST":
            client_id = self._client_identity(request)
            now = time.time()
            window_start = now - RATE_LIMIT_WINDOW

            with rate_limit_lock:
                stale_keys = [
                    key for key, values in rate_limit_store.items()
                    if not values or max(values) <= window_start
                ]
                for key in stale_keys:
                    rate_limit_store.pop(key, None)

                rate_limit_store[client_id] = [
                    t for t in rate_limit_store[client_id] if t > window_start
                ]

                if len(rate_limit_store[client_id]) >= RATE_LIMIT_MAX_REQUESTS:
                    wait = int(RATE_LIMIT_WINDOW - (now - rate_limit_store[client_id][0]))
                    return JSONResponse(
                        {"detail": f"一小时内最多提交 5 次，请 {max(wait, 1)} 秒后再试"},
                        status_code=429,
                    )

                rate_limit_store[client_id].append(now)

        return await call_next(request)


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="MCM/ICM Award Reviewer",
    description="基于 AI 的美赛论文严苛评审",
    version="4.0",
)

# 注册中间件
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS,
)


# ============================================================================
# Routes
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def index():
    """主页面"""
    template_path = TEMPLATES_DIR / "index.html"
    if not template_path.exists():
        return HTMLResponse("<h1>模板文件缺失</h1>", status_code=500)
    return template_path.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "mode": "ai_rubric",
        "evaluator_loaded": _pdf_parser is not None,
    }


@app.get("/api/stats")
async def stats():
    """返回使用统计"""
    with _stats_lock:
        return {
            "total_predictions": _usage_stats["total_predictions"],
            "today_predictions": _usage_stats["today_predictions"],
            "award_counts": dict(_usage_stats["award_counts"]),
            "recent_scores": list(_usage_stats["recent_scores"]),
        }


@app.post("/api/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    problem: str = Form("auto"),
    year: str = Form("auto"),
):
    """
    接收 PDF 文件并返回预测结果。

    安全措施:
    - 仅接受 PDF MIME type
    - 文件大小限制
    - 频率限制 (中间件)
    - 临时文件自动清理
    """
    # ---- 1. 文件类型校验 ----
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"仅接受 PDF 文件，当前类型: {file.content_type}",
        )

    # ---- 2. 文件名校验 ----
    original_name = file.filename or "upload.pdf"
    if not original_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅接受 .pdf 后缀的文件")

    # 临时落盘只使用 ASCII 文件名，避免中文名在部分客户端/终端编码下变成非法路径。
    safe_name = "upload.pdf"

    # ---- 3. 读取 & 大小校验 ----
    chunks: list[bytes] = []
    total_size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件过大，上限 20MB",
            )
        chunks.append(chunk)
    contents = b"".join(chunks)

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大 ({len(contents)/1024/1024:.1f}MB)，上限 20MB",
        )

    # ---- 4. PDF 魔术字节校验 ----
    if contents[:5] != b"%PDF-":
        raise HTTPException(status_code=400, detail="文件不是有效的 PDF 格式")

    # ---- 5. 写入临时文件 ----
    file_id = uuid.uuid4().hex[:12]
    tmp_path = TEMP_DIR / f"{file_id}_{safe_name}"
    tmp_path.write_bytes(contents)

    # ---- 6. 解析参数 ----
    problem_arg = problem.strip().upper() if problem.strip().lower() != "auto" else None
    year_arg = int(year.strip()) if year.strip().lower() != "auto" and year.strip().isdigit() else None

    # ---- 7. 解析 PDF 并调用 AI 评分 ----
    try:
        pdf_parser, image_extractor, rubric_scorer, problem_detector = get_evaluators()
        parsed = pdf_parser.parse(str(tmp_path))
        if not parsed.get("success"):
            logger.warning("PDF 解析失败")
            raise HTTPException(status_code=400, detail="PDF 解析失败，请确认文件未损坏且未加密")

        abstract = parsed.get("abstract", "")
        full_text = parsed.get("full_text", "")
        images = parsed.get("images", [])
        metadata = parsed.get("metadata", {})
        structure = parsed.get("structure", {})
        detection = problem_detector.detect(full_text, str(tmp_path), year_arg)
        summary_problem = detect_summary_sheet_problem(full_text)
        problem_detected = problem_arg or summary_problem or detection.get("problem", "auto")
        year_detected = year_arg or detection.get("year") or problem_detector.detect_year(full_text, str(tmp_path)) or "auto"
        contest_detected = detection.get(
            "contest",
            "MCM" if str(problem_detected).upper() in {"A", "B", "C"} else "ICM",
        )
        image_result = image_extractor.extract(images)
        page_count = int(metadata.get("page_count", 0) or 0)
        ref_count = int(metadata.get("ref_count", 0) or 0)
        raw_image_count = int(metadata.get("raw_image_count", 0) or 0)
        figure_caption_count = int(structure.get("figure_caption_count", 0) or 0)
        filtered_image_count = len(images)
        display_image_count = max(filtered_image_count, raw_image_count, figure_caption_count)
        # 释放原始图片内存（PIL Image 无循环引用，del 后引用计数立即回收）
        del images

        # 截断全文以降低内存压力（LLM 评分仅使用前 ~18k 字符的证据片段）
        full_text = full_text[:100000] if len(full_text) > 100000 else full_text

        llm_rubric = rubric_scorer.score(
            abstract=abstract,
            full_text=full_text,
            structure=structure,
            image_result=image_result,
            image_count=display_image_count,
            page_count=page_count,
            ref_count=ref_count,
            problem=problem_detected,
            contest=contest_detected,
            year=year_detected,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("预测失败")
        raise HTTPException(status_code=500, detail="预测失败，请稍后重试")
    finally:
        _cleanup_temp_file(tmp_path)

    # ---- 8. 记录统计 ----
    record_prediction(
        float(llm_rubric.get("score", 0)),
        str(problem_detected or "?"),
        str(llm_rubric.get("award_prediction", "S/U")),
    )

    # ---- 9. 返回结果 ----
    return JSONResponse({
        "success": True,
        "is_mcm": True,
        "mode": "ai_rubric",
        "llm_rubric": public_rubric_payload(llm_rubric),
        "problem": problem_detected,
        "contest": contest_detected,
        "year": year_detected,
        "metadata": {
            "abstract_length": len(abstract),
            "abstract_word_count": len(abstract.split()),
            "full_text_word_count": len(full_text.split()),
            "image_count": display_image_count,
            "filtered_image_count": filtered_image_count,
            "raw_image_count": raw_image_count,
            "page_count": page_count,
            "ref_count": ref_count,
        },
        "structure": structure,
    })


def _cleanup_temp_file(path: Path):
    """后台上传文件清理"""
    try:
        if path.exists():
            path.unlink()
            logger.info(f"临时文件已清理: {path.name}")
    except Exception as e:
        logger.warning(f"清理临时文件失败: {e}")


# ============================================================================
# Startup / Shutdown
# ============================================================================


@app.on_event("startup")
async def startup():
    """初始化轻量组件"""
    _load_stats()
    logger.info(f"已加载统计: {_usage_stats['total_predictions']} 次历史预测")

    # 生产环境启动检查
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.warning("!!! 未设置 DEEPSEEK_API_KEY 环境变量 — AI 评分将不可用 !!!")
    else:
        logger.info(f"DeepSeek API Key 已配置 ({api_key[:8]}***)")

    default_hosts = {"127.0.0.1", "localhost", "testserver"}
    if set(ALLOWED_HOSTS) == default_hosts:
        logger.warning("ALLOWED_HOSTS 使用默认值，生产环境请设置为你的 Render 域名")

    logger.info("服务启动，使用 AI 作为主评审...")
    get_evaluators()
    # 清理旧临时文件
    for f in TEMP_DIR.glob("*.pdf"):
        try:
            f.unlink()
        except Exception:
            pass
    port = os.environ.get("PORT", "8000")
    logger.info(f"服务就绪 — 端口 {port}")


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "web_app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info",
    )
