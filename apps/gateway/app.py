import json
import math
import os
from datetime import datetime

import psycopg
from fastapi import FastAPI, HTTPException
from kafka import KafkaProducer
from pydantic import BaseModel

# 配置 PostgreSQL 和 Kafka。
DATABASE_URL = os.environ["DATABASE_URL"]
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transaction-analysis")

# 支付方式编码，与训练脚本保持一致。
PAYMENT_FORMAT_CODES = {
    "ACH": 1,
    "CREDIT CARD": 2,
    "CHEQUE": 3,
    "CASH": 4,
    "WIRE": 5,
}

# 创建 FastAPI 应用。
app = FastAPI(title="fincross-risk Data Service")

# 创建 Kafka 生产者。
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    # 将任务序列化为 JSON 字节。
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

# 定义交易请求模型。
class TransactionIn(BaseModel):
    external_id: str
    transaction_timestamp: datetime
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_received: float
    receiving_currency: str
    amount_paid: float
    payment_currency: str
    payment_format: str
    # 可选的真实标签。
    is_laundering: bool | None = None

# 创建数据库连接。
def db():
    return psycopg.connect(DATABASE_URL)

# 将支付方式转换为数值编码，未知方式记为 0。
def payment_format_code(value: str) -> float:
    return float(PAYMENT_FORMAT_CODES.get(value.strip().upper(), 0))

# 生成交易特征。
def make_features(item: TransactionIn) -> dict:
    # 提取交易小时和星期。
    hour = item.transaction_timestamp.hour
    weekday = item.transaction_timestamp.weekday()
    return {
        # 保留原始金额，并计算非负金额的对数特征。
        "amount_paid": float(item.amount_paid),
        "amount_received": float(item.amount_received),
        "amount_paid_log": math.log1p(max(item.amount_paid, 0.0)),
        "amount_received_log": math.log1p(max(item.amount_received, 0.0)),
        "hour": float(hour),
        "weekday": float(weekday),
        # 生成跨行和跨币种标记。
        "cross_bank_flag": float(item.from_bank != item.to_bank),
        "cross_currency_flag": float(
            item.payment_currency.strip().upper()
            != item.receiving_currency.strip().upper()
        ),
        "payment_format_code": payment_format_code(item.payment_format),
    }

# Data Service 健康检查。
@app.get("/health")
def health():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    return {"service": "data-service", "status": "ok"}

# 提交交易。
@app.post("/transactions")
def create_transaction(item: TransactionIn):
    features = make_features(item)
    # 保存交易，初始状态为 RECEIVED。
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO transactions
                    (external_id, source_type, transaction_timestamp,
                     from_bank, from_account, to_bank, to_account,
                     amount_received, receiving_currency, amount_paid,
                     payment_currency, payment_format, is_laundering, status)
                    VALUES (%s, 'ONLINE', %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, 'RECEIVED')
                    RETURNING id""",
                    (
                        item.external_id,
                        item.transaction_timestamp,
                        item.from_bank,
                        item.from_account,
                        item.to_bank,
                        item.to_account,
                        item.amount_received,
                        item.receiving_currency,
                        item.amount_paid,
                        item.payment_currency,
                        item.payment_format,
                        item.is_laundering,
                    ),
                )
                # 获取内部交易编号并提交事务。
                transaction_id = cur.fetchone()[0]
                conn.commit()
    except psycopg.errors.UniqueViolation:
        # 交易编号重复时返回 409。
        raise HTTPException(status_code=409, detail="external_id already exists")

    # 组装交易分析任务。
    message = {
        "transaction_id": transaction_id,
        "external_id": item.external_id,
        "features": features,
    }
    try:
        # 发布任务并等待 Kafka 确认。
        producer.send(KAFKA_TOPIC, message).get(timeout=10)
    except Exception as exc:
        # Kafka 发布失败时返回 503。
        raise HTTPException(status_code=503, detail=f"kafka publish failed: {exc}")

    # 将 RECEIVED 更新为 QUEUED，避免覆盖已完成状态。
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE transactions SET status='QUEUED' WHERE id=%s AND status='RECEIVED'",
                (transaction_id,),
            )
            conn.commit()
    # 返回任务受理信息。
    return {
        "external_id": item.external_id,
        "transaction_id": transaction_id,
        "status": "QUEUED",
    }

# 查询交易。
@app.get("/transactions/{external_id}")
def get_transaction(external_id: str):
    # 查询交易和最新预测，尚未预测的交易也保留。
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT t.external_id, t.transaction_timestamp,
                          t.from_bank, t.from_account, t.to_bank, t.to_account,
                          t.amount_received, t.receiving_currency,
                          t.amount_paid, t.payment_currency, t.payment_format,
                          t.is_laundering, t.status,
                          p.risk_probability, p.risk_level,
                          p.model_version, p.predicted_at
                   FROM transactions t
                   LEFT JOIN LATERAL (
                       SELECT * FROM predictions p0
                       WHERE p0.transaction_id=t.id
                       ORDER BY p0.predicted_at DESC LIMIT 1
                   ) p ON TRUE
                   WHERE t.external_id=%s""",
                (external_id,),
            )
            row = cur.fetchone()
    if row is None:
        # 交易不存在时返回 404。
        raise HTTPException(status_code=404, detail="transaction not found")
    # 按查询字段顺序组装响应字典。
    keys = [
        "external_id", "transaction_timestamp", "from_bank", "from_account",
        "to_bank", "to_account", "amount_received", "receiving_currency",
        "amount_paid", "payment_currency", "payment_format", "is_laundering",
        "status", "risk_probability", "risk_level", "model_version",
        "predicted_at",
    ]
    return dict(zip(keys, row))