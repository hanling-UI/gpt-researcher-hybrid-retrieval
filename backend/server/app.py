import asyncio
import json
import os
import re
from typing import Dict, List, Any
import time
import logging
import sys
import warnings
from pathlib import Path

# Suppress Pydantic V2 migration warnings
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, File, UploadFile, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict

# Add the parent directory to sys.path to make sure we can import from server
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from server.websocket_manager import WebSocketManager
from server.server_utils import (
    get_config_dict, sanitize_filename,
    update_environment_variables, handle_file_upload, handle_file_deletion,
    execute_multi_agents, handle_websocket_communication
)
from server.agent_discovery import build_agent_discovery_document

from server.websocket_manager import run_agent
from utils import write_md_to_word, write_md_to_pdf
from gpt_researcher.utils.enum import Tone
from chat.chat import ChatAgentWithMemory

from server.report_store import ReportStore

# MongoDB services removed - no database persistence needed

# Setup logging
logger = logging.getLogger(__name__)

# Don't override parent logger settings
logger.propagate = True

# Silence uvicorn reload logs
logging.getLogger("uvicorn.supervisors.ChangeReload").setLevel(logging.WARNING)

# Models


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    report_source: str
    tone: str
    headers: dict | None = None
    repo_name: str
    branch_name: str
    generate_in_background: bool = True


class ExportRequest(BaseModel):
    """按需导出：正文由前端传上来，服务端不依赖任何持久化文件。

    为什么这样设计：Render 这类无状态容器每次重新部署/休眠唤醒都会清空
    outputs/ 目录。如果导出依赖服务端文件，用户过一会儿再点下载就是 404。
    把正文放在请求体里，服务端只做"转换并返回流"，重启多少次都不影响。
    """
    markdown: str
    format: str = "pdf"
    title: str = "report"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # Allow extra fields in the request
    
    report: str
    messages: List[Dict[str, Any]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs("outputs", exist_ok=True)
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
    
    # Mount frontend static files
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
    if os.path.exists(frontend_path):
        app.mount("/site", StaticFiles(directory=frontend_path), name="frontend")
        logger.debug(f"Frontend mounted from: {frontend_path}")
        
        # Also mount the static directory directly for assets referenced as /static/
        static_path = os.path.join(frontend_path, "static")
        if os.path.exists(static_path):
            app.mount("/static", StaticFiles(directory=static_path), name="static")
            logger.debug(f"Static assets mounted from: {static_path}")
    else:
        logger.warning(f"Frontend directory not found: {frontend_path}")
    
    logger.info("GPT Researcher API ready - local mode (no database persistence)")
    yield
    # Shutdown
    logger.info("Research API shutting down")

# App initialization
app = FastAPI(lifespan=lifespan)

# Configure allowed origins for CORS
allowed_origins_env = os.getenv("CORS_ALLOW_ORIGINS")
ALLOWED_ORIGINS = (
    [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    if allowed_origins_env
    else [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://app.gptr.dev",
    ]
)

# Standard JSON response - no custom MongoDB encoding needed

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use default JSON response class

# Mount static files for frontend
# Get the absolute path to the frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

# Mount static directories
app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")
app.mount("/site", StaticFiles(directory=frontend_dir), name="site")

# WebSocket manager
manager = WebSocketManager()

report_store = ReportStore(Path(os.getenv('REPORT_STORE_PATH', os.path.join('data', 'reports.json'))))

# Constants
DOC_PATH = os.getenv("DOC_PATH", "./my-docs")

# Startup event


# Lifespan events now handled in the lifespan context manager above


# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend HTML page."""
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    index_path = os.path.join(frontend_dir, "index.html")
    
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return HTMLResponse(content=content)


@app.get("/.well-known/agent-discovery.json")
async def agent_discovery(request: Request):
    """Advertise GPT Researcher services via the Agent Discovery Protocol."""
    origin = str(request.base_url).rstrip("/")
    domain = request.url.hostname or request.headers.get("host", "")
    contact = os.getenv("AGENT_DISCOVERY_CONTACT")

    document = build_agent_discovery_document(origin=origin, domain=domain, contact=contact)
    response = JSONResponse(content=document)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.get("/report/{research_id}")
async def read_report(request: Request, research_id: str):
    docx_path = os.path.join('outputs', f"{research_id}.docx")
    if not os.path.exists(docx_path):
        return {"message": "Report not found."}
    return FileResponse(docx_path)


# Simplified API routes - no database persistence
_EXPORT_JOBS: dict[str, dict] = {}
_EXPORT_JOB_LIMIT = 20  # 只保留最近 20 个任务，避免内存无限增长


def _remember_job(job_id: str, payload: dict) -> None:
    _EXPORT_JOBS[job_id] = payload
    if len(_EXPORT_JOBS) > _EXPORT_JOB_LIMIT:
        for old in list(_EXPORT_JOBS.keys())[: len(_EXPORT_JOBS) - _EXPORT_JOB_LIMIT]:
            _EXPORT_JOBS.pop(old, None)


@app.post("/api/export")
async def start_export(payload: ExportRequest):
    """提交导出任务，立刻返回 job_id。

    免费实例只有 0.1 CPU，中文 PDF 要渲染几分钟。同步请求会撞上反向代理超时
    （表现为 502/白屏）。改成"提交任务 + 轮询"，每个请求都是毫秒级。
    """
    text = payload.markdown or ""
    if len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="报告正文为空，无法导出")

    fmt = (payload.format or "").strip().lower()
    if fmt not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="format 只支持 pdf 或 docx")

    import uuid

    job_id = uuid.uuid4().hex[:16]
    _remember_job(job_id, {
        "status": "running",
        "path": "",
        "error": "",
        "format": fmt,
        "markdown": text,          # 预览页要用
        "title": payload.title or "report",
    })

    async def _run() -> None:
        try:
            path = await (write_md_to_pdf(text, job_id) if fmt == "pdf"
                          else write_md_to_word(text, job_id))
            if path:
                _EXPORT_JOBS[job_id].update(status="done", path=path)
            else:
                _EXPORT_JOBS[job_id].update(status="failed", error=f"{fmt.upper()} 生成失败")
        except Exception as exc:  # noqa: BLE001
            _EXPORT_JOBS[job_id].update(status="failed", error=str(exc))

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "running"}


@app.get("/api/export/job/{job_id}")
async def export_job_status(job_id: str):
    job = _EXPORT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return {"status": job["status"], "error": job.get("error", "")}


@app.get("/api/export/file/{job_id}")
async def export_job_file(job_id: str):
    """PDF 内联预览；DOCX 作为附件下载。"""
    job = _EXPORT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"任务未完成（当前 {job['status']}）")

    target = Path("outputs") / Path(job["path"]).name
    if not target.exists():
        raise HTTPException(status_code=404, detail="生成的文件已丢失，请重新导出")

    if target.suffix == ".pdf":
        return FileResponse(target, media_type="application/pdf")
    return FileResponse(
        target,
        filename="report.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


_PREVIEW_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - Word 预览</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #525659; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }}
  .toolbar {{
    position: sticky; top: 0; z-index: 10; display: flex; align-items: center; gap: 12px;
    padding: 12px 20px; background: #202124; color: #e8eaed; box-shadow: 0 2px 8px rgba(0,0,0,.4);
  }}
  .toolbar .name {{ flex: 1; font-size: 14px; opacity: .9; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .btn {{
    display: inline-flex; align-items: center; gap: 6px; padding: 8px 16px; border-radius: 6px;
    background: #0b57d0; color: #fff; text-decoration: none; font-size: 14px; border: none; cursor: pointer;
  }}
  .btn:hover {{ background: #0842a0; }}
  .btn.secondary {{ background: #3c4043; }}
  .btn.secondary:hover {{ background: #4a4e51; }}
  .paper {{
    max-width: 820px; margin: 24px auto 64px; background: #fff; padding: 64px 72px;
    box-shadow: 0 4px 18px rgba(0,0,0,.35); line-height: 1.9; color: #1a1a1a;
    font-family: Georgia, "Times New Roman", "Songti SC", "SimSun", serif; font-size: 16px;
  }}
  .paper h1 {{ font-size: 26px; line-height: 1.5; margin: 0 0 24px; }}
  .paper h2 {{ font-size: 20px; margin: 32px 0 14px; }}
  .paper h3 {{ font-size: 17px; margin: 24px 0 10px; }}
  .paper p {{ margin: 12px 0; text-align: justify; }}
  .paper table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }}
  .paper th, .paper td {{ border: 1px solid #d0d0d0; padding: 8px 10px; text-align: left; }}
  .paper th {{ background: #f3f4f6; }}
  .paper a {{ color: #0b57d0; word-break: break-all; }}
  .paper blockquote {{ margin: 16px 0; padding-left: 14px; border-left: 3px solid #d0d0d0; color: #555; }}
  .hint {{ max-width: 820px; margin: 16px auto 0; color: #cfd2d6; font-size: 13px; text-align: center; }}
  @media print {{
    .toolbar, .hint {{ display: none; }}
    body {{ background: #fff; }}
    .paper {{ box-shadow: none; margin: 0; padding: 0; max-width: none; }}
  }}
</style>
</head>
<body>
<div class="toolbar">
  <span class="name">{title}</span>
  <a class="btn" href="/api/export/file/{job_id}">下载 Word (.docx)</a>
  <button class="btn secondary" onclick="window.print()">打印 / 另存为 PDF</button>
</div>
<div class="paper">{body}</div>
<p class="hint">这是 Word 文档的网页预览效果，点上方按钮可下载 .docx 原件。</p>
</body>
</html>"""


@app.get("/api/export/preview/{job_id}", response_class=HTMLResponse)
async def export_preview(job_id: str):
    """Word 的网页预览页。

    浏览器无法原生预览 .docx，所以这里把同一份 Markdown 渲染成排好版的网页，
    顶部给"下载 Word"和"打印/另存为 PDF"两个按钮——效果等价于 PDF 预览器的保存按钮。
    """
    job = _EXPORT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"任务未完成（当前 {job['status']}）")

    import html as html_module

    import mistune

    body_html = mistune.html(job.get("markdown", ""))
    return HTMLResponse(
        _PREVIEW_TEMPLATE.format(
            title=html_module.escape(job.get("title") or "report"),
            job_id=job_id,
            body=body_html,
        )
    )


# ==================== 会话持久化 ====================
# 浏览器 localStorage 只在本机生效；这里再落一份到服务器，
# 换浏览器/换设备时也能拉回来。
# 注意：Render 免费层没有持久磁盘，重新部署会清空 data/ 目录。
_SESSIONS_FILE = Path("data/sessions.json")


def _load_sessions_file() -> list:
    try:
        if _SESSIONS_FILE.exists():
            data = json.loads(_SESSIONS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取会话文件失败：%s", exc)
    return []


def _save_sessions_file(items: list) -> None:
    _SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SESSIONS_FILE.write_text(
        json.dumps(items, ensure_ascii=False),
        encoding="utf-8",
    )


@app.get("/api/sessions")
async def list_sessions():
    """返回服务端保存的会话列表（按更新时间倒序）。"""
    items = _load_sessions_file()
    items.sort(key=lambda s: str(s.get("updatedAt") or s.get("createdAt") or ""), reverse=True)
    return {"sessions": items}


@app.post("/api/sessions")
async def save_sessions(payload: dict):
    """整体保存会话列表（单用户场景，后写覆盖先写）。"""
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise HTTPException(status_code=400, detail="sessions 必须是数组")

    # 只保留必要字段，避免把任意内容写进磁盘
    cleaned = []
    for item in sessions[:200]:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            "id": str(item.get("id") or ""),
            "title": str(item.get("title") or "未命名研究")[:120],
            "report": str(item.get("report") or ""),
            "messages": item.get("messages") if isinstance(item.get("messages"), list) else [],
            "links": item.get("links") if isinstance(item.get("links"), dict) else {},
            "createdAt": item.get("createdAt") or "",
            "updatedAt": item.get("updatedAt") or "",
        })

    _save_sessions_file(cleaned)
    return {"ok": True, "count": len(cleaned)}


@app.get("/api/export/{stem}")
async def export_report(stem: str, format: str = "pdf"):
    """按需把 outputs/<stem>.md 转换成 pdf/docx 并返回。

    为什么按需：免费实例只有 0.1 CPU，WeasyPrint 渲染中文长报告要几分钟。
    如果在研究结束时就同步生成，用户要干等几分钟才能看到下载按钮。
    改成点击时才生成：
      * 按钮在报告结束的瞬间就可用；
      * 不点就不花 CPU；
      * 同一份报告第二次点击直接命中已生成的文件。
    """
    safe_stem = re.sub(r"[^\w\-]", "", stem)
    if not safe_stem:
        raise HTTPException(status_code=400, detail="非法的报告标识")

    outputs = Path("outputs")
    md_path = outputs / f"{safe_stem}.md"
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="报告正文不存在，请重新研究")

    if format not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="format 只支持 pdf 或 docx")

    target = outputs / f"{safe_stem}.{format}"
    if not target.exists():
        text = md_path.read_text(encoding="utf-8")
        if format == "pdf":
            await write_md_to_pdf(text, safe_stem)
        else:
            await write_md_to_word(text, safe_stem)

    if not target.exists():
        raise HTTPException(status_code=500, detail=f"{format.upper()} 生成失败（详见服务端日志）")

    # PDF 不设 filename：浏览器内联预览，右上角仍可下载。
    # DOCX 浏览器无法预览，设 filename 直接下载。
    if format == "pdf":
        return FileResponse(target, media_type="application/pdf")
    return FileResponse(target, filename=target.name)


@app.get("/api/reports")
async def get_all_reports(report_ids: str = None):
    report_ids_list = report_ids.split(",") if report_ids else None
    reports = await report_store.list_reports(report_ids_list)
    return {"reports": reports}


@app.get("/api/reports/{research_id}")
async def get_report_by_id(research_id: str):
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report": report}


@app.post("/api/reports")
async def create_or_update_report(request: Request):
    try:
        data = await request.json()
        research_id = data.get("id", "temp_id")

        now_ms = int(time.time() * 1000)
        existing = await report_store.get_report(research_id)
        incoming_timestamp = data.get("timestamp")
        timestamp = incoming_timestamp if isinstance(incoming_timestamp, int) else now_ms
        if existing and isinstance(existing.get("timestamp"), int):
            timestamp = max(timestamp, existing["timestamp"])

        report = {
            "id": research_id,
            "question": data.get("question"),
            "answer": data.get("answer"),
            "orderedData": data.get("orderedData") or [],
            "chatMessages": data.get("chatMessages") or [],
            "timestamp": timestamp,
        }

        await report_store.upsert_report(research_id, report)
        return {"success": True, "id": research_id}
    except Exception as e:
        logger.error(f"Error processing report creation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/reports/{research_id}")
async def update_report(research_id: str, request: Request):
    existing = await report_store.get_report(research_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Report not found")

    data = await request.json()
    now_ms = int(time.time() * 1000)

    updated = {
        **existing,
        **{k: v for k, v in data.items() if v is not None},
        "id": research_id,
        "timestamp": now_ms,
    }

    await report_store.upsert_report(research_id, updated)
    return {"success": True, "id": research_id}


@app.delete("/api/reports/{research_id}")
async def delete_report(research_id: str):
    existed = await report_store.delete_report(research_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"success": True}


@app.get("/api/reports/{research_id}/chat")
async def get_report_chat(research_id: str):
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"chatMessages": report.get("chatMessages") or []}


@app.post("/api/reports/{research_id}/chat")
async def research_report_chat(research_id: str, request: Request):
    """Chat against a stored report: LLM reply + persist user/assistant messages."""
    report = await report_store.get_report(research_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    try:
        data = await request.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    # Accept either a bare chat message object or {messages, report, message}
    incoming_messages = data.get("messages")
    if not isinstance(incoming_messages, list):
        if any(key in data for key in ("role", "content")):
            incoming_messages = [data]
        elif isinstance(data.get("message"), dict):
            incoming_messages = [data["message"]]
        else:
            incoming_messages = []

    report_text = data.get("report") or report.get("answer") or ""
    history = report.get("chatMessages") or []
    if not isinstance(history, list):
        history = []

    messages_for_llm = list(history)
    for msg in incoming_messages:
        if isinstance(msg, dict):
            messages_for_llm.append(msg)

    try:
        chat_agent = ChatAgentWithMemory(
            report=report_text if isinstance(report_text, str) else str(report_text),
            config_path="default",
            headers=None,
        )
        response_content, tool_calls_metadata = await chat_agent.chat(messages_for_llm, None)
    except Exception as e:
        logger.error(f"Error in research report chat: {str(e)}", exc_info=True)
        return {"error": str(e)}

    now_ms = int(time.time() * 1000)
    response_message = {
        "role": "assistant",
        "content": response_content,
        "timestamp": now_ms,
        "metadata": {"tool_calls": tool_calls_metadata} if tool_calls_metadata else None,
    }

    new_chat = list(history)
    for msg in incoming_messages:
        if isinstance(msg, dict):
            new_chat.append(msg)
    new_chat.append(response_message)

    updated = {
        **report,
        "chatMessages": new_chat,
        "timestamp": now_ms,
    }
    await report_store.upsert_report(research_id, updated)
    return {"success": True, "id": research_id, "response": response_message}




async def write_report(research_request: ResearchRequest, research_id: str = None):
    report_information = await run_agent(
        task=research_request.task,
        report_type=research_request.report_type,
        report_source=research_request.report_source,
        source_urls=[],
        document_urls=[],
        tone=Tone[research_request.tone],
        websocket=None,
        stream_output=None,
        headers=research_request.headers,
        query_domains=[],
        config_path="",
        return_researcher=True
    )

    docx_path = await write_md_to_word(report_information[0], research_id)
    pdf_path = await write_md_to_pdf(report_information[0], research_id)
    if research_request.report_type != "multi_agents":
        report, researcher = report_information
        response = {
            "research_id": research_id,
            "research_information": {
                "source_urls": researcher.get_source_urls(),
                "research_costs": researcher.get_costs(),
                "visited_urls": list(researcher.visited_urls),
                "research_images": researcher.get_research_images(),
                # "research_sources": researcher.get_research_sources(),  # Raw content of sources may be very large
            },
            "report": report,
            "docx_path": docx_path,
            "pdf_path": pdf_path
        }
    else:
        response = { "research_id": research_id, "report": "", "docx_path": docx_path, "pdf_path": pdf_path }

    return response

@app.post("/report/")
async def generate_report(research_request: ResearchRequest, background_tasks: BackgroundTasks):
    research_id = sanitize_filename(f"task_{int(time.time())}_{research_request.task}")

    if research_request.generate_in_background:
        background_tasks.add_task(write_report, research_request=research_request, research_id=research_id)
        return {"message": "Your report is being generated in the background. Please check back later.",
                "research_id": research_id}
    else:
        response = await write_report(research_request, research_id)
        return response


@app.get("/files/")
async def list_files():
    if not os.path.exists(DOC_PATH):
        os.makedirs(DOC_PATH, exist_ok=True)
    files = os.listdir(DOC_PATH)
    print(f"Files in {DOC_PATH}: {files}")
    return {"files": files}


@app.post("/api/multi_agents")
async def run_multi_agents():
    return await execute_multi_agents(manager)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    return await handle_file_upload(file, DOC_PATH)


@app.delete("/files/{filename}")
async def delete_file(filename: str):
    return await handle_file_deletion(filename, DOC_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await handle_websocket_communication(websocket, manager)
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected with code {e.code} and reason: '{e.reason}'")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {str(e)}")
    finally:
        # Release the connection's queue and sender task however the loop ended.
        await manager.disconnect(websocket)

@app.post("/api/chat")
async def chat(chat_request: ChatRequest):
    """Process a chat request with a report and message history.

    Args:
        chat_request: ChatRequest object containing report text and message history

    Returns:
        JSON response with the assistant's message and any tool usage metadata
    """
    try:
        logger.info(f"Received chat request with {len(chat_request.messages)} messages")

        # Create chat agent with the report
        chat_agent = ChatAgentWithMemory(
            report=chat_request.report,
            config_path="default",
            headers=None
        )

        # Process the chat and get response with metadata
        response_content, tool_calls_metadata = await chat_agent.chat(chat_request.messages, None)
        logger.info(f"response_content: {response_content}")
        logger.info(f"Got chat response of length: {len(response_content) if response_content else 0}")
        
        if tool_calls_metadata:
            logger.info(f"Tool calls used: {json.dumps(tool_calls_metadata)}")

        # Format response as a ChatMessage object with role, content, timestamp and metadata
        response_message = {
            "role": "assistant",
            "content": response_content,
            "timestamp": int(time.time() * 1000),  # Current time in milliseconds
            "metadata": {
                "tool_calls": tool_calls_metadata
            } if tool_calls_metadata else None
        }

        logger.info(f"Returning formatted response: {json.dumps(response_message)[:100]}...")
        return {"response": response_message}
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return {"error": str(e)}

