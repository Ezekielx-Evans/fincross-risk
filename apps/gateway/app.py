# Gateway 服务入口。
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

# 配置 Data Service 地址。
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data-service:8000")

# 创建 FastAPI 应用。
app = FastAPI(title="fincross-risk Gateway")

# 保留下游 404 / 409 / 422 / 503，避免前端把失败当作成功。
def forward_response(response: httpx.Response):
    try:
        content = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="invalid JSON from data service")
    return JSONResponse(status_code=response.status_code, content=content)

# Gateway 健康检查。
@app.get("/health")
def health():
    return {"service": "gateway", "status": "ok"}

# 提交交易。
@app.post("/api/transactions")
async def create_transaction(payload: dict):
    try:
        # 异步请求 Data Service。
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(f"{DATA_SERVICE_URL}/transactions", json=payload)
    except httpx.HTTPError as exc:
        # 下游不可用时返回 503。
        raise HTTPException(status_code=503, detail=f"data service unavailable: {exc}")
    return forward_response(response)

# 查询交易。
@app.get("/api/transactions/{external_id}")
async def get_transaction(external_id: str):
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{DATA_SERVICE_URL}/transactions/{external_id}")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"data service unavailable: {exc}")
    return forward_response(response)
