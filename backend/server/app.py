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
@app.post("/api/export")
async def export_report_inline(payload: ExportRequest):
    """把前端传来的 Markdown 即时转换成 PDF / DOCX 并返回文件流。

    这是无状态导出：不读 outputs/、不依赖之前生成过什么文件，
    所以容器重启、休眠唤醒都不会 404。
    """
    text = payload.markdown or ""
    if len(text.strip()) < 20:
        raise HTTPException(status_code=400, detail="报告正文为空，无法导出")

    fmt = (payload.format or "").strip().lower()
    if fmt not in ("pdf", "docx"):
        raise HTTPException(status_code=400, detail="format 只支持 pdf 或 docx")

    # 唯一临时名，避免并发导出互相覆盖
    import uuid

    stem = f"export_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    if fmt == "pdf":
        path = await write_md_to_pdf(text, stem)
    else:
        path = await write_md_to_word(text, stem)

    if not path:
        raise HTTPException(status_code=500, detail=f"{fmt.upper()} 生成失败（详见服务端日志）")

    target = Path("outputs") / Path(path).name
    if not target.exists():
        raise HTTPException(status_code=500, detail="生成的文件未落盘")

    if fmt == "pdf":
        # 不设 filename：浏览器内联预览，右上角仍可另存
        return FileResponse(target, media_type="application/pdf")
    safe_title = re.sub(r"[^\w\-]", "", payload.title or "report") or "report"
    return FileResponse(
        target,
        filename=f"{safe_title}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


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

