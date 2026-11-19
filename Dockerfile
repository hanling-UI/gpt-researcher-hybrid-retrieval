# 精简版镜像：去掉浏览器（本方案用 bs 解析 HTML）与 weasyprint（PDF 导出用不到）
# 原版 Dockerfile 见 git 历史：git show HEAD~1:Dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 只装运行期必需的系统包。numpy / pymupdf / pandas 都有预编译 wheel，
# 不需要 build-essential 去现场编译。
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# 依赖单独一层：只要 requirements.txt 没变，改代码时不会重装依赖
COPY requirements.txt ./
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 应用代码
COPY . .

# 运行时目录 + 非 root 用户
RUN useradd -m appuser \
    && mkdir -p /app/outputs /app/logs /app/data \
    && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000

# 健康检查：平台靠它判断容器是否活着
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/reports" || exit 1

# 用 sh -c 是为了让 ${PORT} 能被展开（Render 之类的平台会注入自己的端口）
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]