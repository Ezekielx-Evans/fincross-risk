import os
from pathlib import Path

# 加载模型，创建接口并返回 HTTP 错误
import joblib
from fastapi import FastAPI, HTTPException

# 固定九个特征的排列顺序，与训练时一致
FEATURE_NAMES = [
    "amount_paid", "amount_received", "amount_paid_log",
    "amount_received_log", "hour", "weekday", "cross_bank_flag",
    "cross_currency_flag", "payment_format_code",
]

# 读取环境变量；未设置时使用默认值
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "aml-model-v1")

# 创建模型服务的 API 应用
app = FastAPI(title="fincross-risk Model Service")

# 启动时加载一次模型；文件不存在时暂记为 None
model = joblib.load(MODEL_PATH) if Path(MODEL_PATH).exists() else None

# 健康检查接口，供外部判断模型服务是否就绪
@app.get("/health")
def health():
    # 模型未加载时返回 503，表示服务尚不可用
    if model is None:
        raise HTTPException(status_code=503, detail="model is not loaded; train model first")

    # 模型已加载，返回服务状态和版本标识
    return {
        "service": "model-service",
        "status": "ok",
        "loaded": model is not None,
        "model_version": MODEL_VERSION,
    }

# 预测接口，payload 接收请求中的 JSON 特征数据
@app.post("/predict")
def predict(payload: dict):
    # 没有可用模型时拒绝预测
    if model is None:
        raise HTTPException(status_code=503, detail="model is not loaded; train model first")

    # 检查九个特征是否齐全，缺失时返回 400 和字段名
    missing = [name for name in FEATURE_NAMES if name not in payload]
    if missing:
        raise HTTPException(status_code=400, detail={"missing": missing})

    # 按训练时的顺序取值并转为浮点数，外层列表表示一批交易
    row = [[float(payload[name]) for name in FEATURE_NAMES]]

    # 计算概率；[0] 取第一笔交易，[1] 取风险类别的概率
    probability = float(model.predict_proba(row)[0][1])

    # 根据风险概率设定风险等级
    risk_level = "HIGH" if probability >= 0.70 else "MEDIUM" if probability >= 0.30 else "LOW"

    # 返回概率、等级和版本；数据库写回由 Worker 负责
    return {
        "risk_probability": probability,
        "risk_level": risk_level,
        "model_version": MODEL_VERSION,
    }