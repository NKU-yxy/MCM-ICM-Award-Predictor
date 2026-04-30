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
import tempfile
import threading
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

# --- project path ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_award import AwardPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("mcm-web")

# ============================================================================
# Constants
# ============================================================================
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
ALLOWED_MIME = {"application/pdf"}
TEMP_DIR = Path(tempfile.gettempdir()) / "mcm_predictor"
TEMP_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Rate limiting
RATE_LIMIT_WINDOW = 60       # seconds
RATE_LIMIT_MAX_REQUESTS = 10  # per window per IP
rate_limit_store: dict[str, list[float]] = defaultdict(list)

# Predictor singleton (loaded once at startup)
_predictor: AwardPredictor | None = None
_predictor_lock = threading.Lock()

# Usage statistics (in-memory, resets on restart)
STATS_FILE = Path(__file__).parent / "data" / "usage_stats.json"
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


def _save_stats():
    """持久化统计到磁盘"""
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATS_FILE.write_text(json.dumps(_usage_stats, ensure_ascii=False, indent=2), "utf-8")
    except Exception:
        pass


def record_prediction(score: float, problem: str, best_award: str):
    """线程安全地记录一次预测"""
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
    # 异步持久化（每 10 次保存一次，减少 I/O）
    if _usage_stats["total_predictions"] % 10 == 0:
        _save_stats()


def get_predictor() -> AwardPredictor:
    """线程安全的预测器单例"""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                logger.info("加载预测模型 (首次启动)...")
                model_path = os.environ.get("MODEL_PATH", "models/scoring_model.pkl")
                _predictor = AwardPredictor(model_path=model_path)
                logger.info("模型加载完成")
    return _predictor


# ============================================================================
# Security Middleware
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """添加安全响应头"""

    async def dispatch(self, request: Request, call_next):
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
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """IP 级别请求频率限制 (仅限制 POST /api/predict)"""

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """获取真实客户端 IP（处理代理/ Render 的 X-Forwarded-For 头）"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host
        return "unknown"

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/api/predict" and request.method == "POST":
            client_ip = self._get_client_ip(request)
            now = time.time()
            window_start = now - RATE_LIMIT_WINDOW

            rate_limit_store[client_ip] = [
                t for t in rate_limit_store[client_ip] if t > window_start
            ]

            if len(rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
                wait = int(RATE_LIMIT_WINDOW - (now - rate_limit_store[client_ip][0]))
                raise HTTPException(
                    status_code=429,
                    detail=f"请求过于频繁，请 {wait} 秒后再试",
                )

            rate_limit_store[client_ip].append(now)

        return await call_next(request)


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="MCM/ICM Award Predictor",
    description="美赛论文获奖等级预测",
    version="3.0",
)

# 注册中间件
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # 生产环境改为实际域名
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
    return {"status": "ok", "predictor_loaded": _predictor is not None}


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
    background_tasks: BackgroundTasks,
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

    # 消毒文件名 (移除路径遍历字符)
    safe_name = Path(original_name).name.replace("..", "_")

    # ---- 3. 读取 & 大小校验 ----
    contents = await file.read()

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

    # 注册清理任务
    background_tasks.add_task(_cleanup_temp_file, tmp_path)

    # ---- 6. 解析参数 ----
    problem_arg = problem.strip().upper() if problem.strip().lower() != "auto" else None
    year_arg = int(year.strip()) if year.strip().lower() != "auto" and year.strip().isdigit() else None

    # ---- 7. 运行预测 ----
    try:
        predictor = get_predictor()
        result = predictor.predict(
            str(tmp_path),
            problem=problem_arg,
            year=year_arg,
            verbose=False,
        )
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")

    # ---- 8. 记录统计 ----
    best_award = max(result["probabilities"], key=result["probabilities"].get) if result.get("probabilities") else "S"
    record_prediction(float(result.get("score", 0)), result.get("problem", "?"), best_award)

    # ---- 9. 返回结果 ----
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "预测失败"))

    return JSONResponse({
        "success": True,
        "score": round(result["score"], 1),
        "probabilities": {
            k: round(v * 100, 1) for k, v in result["probabilities"].items()
        },
        "aspect_scores": result.get("aspect_scores", {}),
        "aspect_details": result.get("aspect_details", {}),
        "similarity": round(result["similarity"], 4),
        "problem": result["problem"],
        "contest": result["contest"],
        "year": result["year"],
        "quality_tier": result["quality_tier"],
        "emoji": result["emoji"],
        "description": result["description"],
        "metadata": {
            "abstract_length": result["metadata"]["abstract_length"],
            "image_count": result["metadata"]["image_count"],
            "page_count": result["metadata"]["page_count"],
            "ref_count": result["metadata"]["ref_count"],
        },
        "structure": result["metadata"].get("structure", {}),
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
    """预热模型"""
    _load_stats()
    logger.info(f"已加载统计: {_usage_stats['total_predictions']} 次历史预测")
    logger.info("服务启动，预热预测模型...")
    try:
        get_predictor()
    except FileNotFoundError:
        logger.warning(
            "模型文件不存在！请先运行: python scripts/train_scoring_model.py"
        )
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
