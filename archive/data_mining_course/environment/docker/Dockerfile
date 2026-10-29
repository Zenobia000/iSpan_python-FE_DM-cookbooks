# 使用 uv 官方映像取得 uv 執行檔，再以 slim Python 為基礎
FROM python:3.11-slim

# 從官方映像複製 uv / uvx 執行檔
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先複製依賴清單以善用 Docker layer 快取
# 注意：build context 為 repo 根目錄（見 docker-compose.yml）
COPY pyproject.toml uv.lock README.md ./

# 依 lockfile 安裝核心依賴（不含 dev 工具）
RUN uv sync --frozen --no-dev --no-install-project

# 複製其餘專案內容
COPY . .

EXPOSE 8888

# 透過 uv 啟動 JupyterLab
CMD ["uv", "run", "--no-dev", "jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
