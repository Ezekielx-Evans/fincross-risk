# 1. 导入依赖：标准库负责 JSON、特征计算和环境配置。
import json
import math
import os
from datetime import datetime

# 第三方库分别负责数据库访问、HTTP 接口、Kafka 消息和请求校验。
import psycopg
from fastapi import FastAPI, HTTPException
from kafka import KafkaProducer
from pydantic import BaseModel

# 2. 读取连接配置：数据库地址必填，Kafka 地址和主题提供默认值。
DATABASE_URL = os.environ["DATABASE_URL"]
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transaction-analysis")

# 支付方式编码需与训练脚本一致，避免训练与在线推理使用不同含义。
PAYMENT_FORMAT_CODES = {
    "ACH": 1,
    "CREDIT CARD": 2,
    "CHEQUE": 3,
    "CASH": 4,
    "WIRE": 5,
}

# 3. 创建接口应用和可复用的 Kafka 生产者。
app = FastAPI(title="fincross-risk Data Service")
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVERS,
    # 将任务字典序列化为 JSON 字节，供 Worker 读取。
    value_serializer=lambda value: json.dumps(value).encode("utf-8"),
)

# 4. 定义交易请求：Pydantic 校验必填字段和类型，不符合要求时返回 422。
class TransactionIn(BaseModel):
    # 业务交易编号和时间；external_id 用于去重与结果查询。
    external_id: str
    transaction_timestamp: datetime
    # 付款方与收款方的银行、账户信息。
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    # 收付金额、币种与支付方式，用于保存交易和生成特征。
    amount_received: float
    receiving_currency: str
    amount_paid: float
    payment_currency: str
    payment_format: str
    # 离线数据有标签；在线预测请求通常没有真实标签。
    is_laundering: bool | None = None

# 5. 创建数据库连接：由调用处的 with 管理事务及连接关闭。
def db():
    return psycopg.connect(DATABASE_URL)

# 统一支付方式的大小写和首尾空格，未知方式编码为 0。
def payment_format_code(value: str) -> float:
    return float(PAYMENT_FORMAT_CODES.get(value.strip().upper(), 0))

# 6. 生成 9 项模型特征，不将账户编号或真实标签作为预测输入。
def make_features(item: TransactionIn) -> dict:
    # 提取小时和星期；星期一为 0，星期日为 6。
    hour = item.transaction_timestamp.hour
    weekday = item.transaction_timestamp.weekday()
    return {
        # 保留原始金额，并用 log1p 压缩大额数值；取对数前将负值截为 0。
        "amount_paid": float(item.amount_paid),
        "amount_received": float(item.amount_received),
        "amount_paid_log": math.log1p(max(item.amount_paid, 0.0)),
        "amount_received_log": math.log1p(max(item.amount_received, 0.0)),
        "hour": float(hour),
        "weekday": float(weekday),
        # 银行或币种不同记为 1，否则为 0；不直接代表跨国交易。
        "cross_bank_flag": float(item.from_bank != item.to_bank),
        "cross_currency_flag": float(
            item.payment_currency.strip().upper()
            != item.receiving_currency.strip().upper()
        ),
        "payment_format_code": payment_format_code(item.payment_format),
    }

# 7. 健康检查：执行简单 SQL，确认服务能连接并访问数据库。
@app.get("/health")
def health():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
    return {"service": "data-service", "status": "ok"}

# 8. 提交交易：生成特征 → 保存交易 → 发布任务 → 更新状态。
@app.post("/transactions")
def create_transaction(item: TransactionIn):
    # 先生成特征，避免特征计算失败后仍留下待处理交易。
    features = make_features(item)
    # 先写入 RECEIVED 状态；%s 参数与 SQL 分开传入，不拼接用户输入。
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
                # 获取数据库内部编号，并在发布 Kafka 任务前提交交易。
                transaction_id = cur.fetchone()[0]
                conn.commit()
    except psycopg.errors.UniqueViolation:
        # 重复的业务交易编号返回 409，不再发布任务。
        raise HTTPException(status_code=409, detail="external_id already exists")

    # 消息携带内部编号、业务编号和特征，供 Worker 推理并写回结果。
    message = {
        "transaction_id": transaction_id,
        "external_id": item.external_id,
        "features": features,
    }
    # 等待 Kafka 确认消息发送；失败返回 503，已保存的交易仍为 RECEIVED。
    try:
        producer.send(KAFKA_TOPIC, message).get(timeout=10)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"kafka publish failed: {exc}")

    # 发布成功后标为 QUEUED；仅更新 RECEIVED，避免覆盖 Worker 已写入的 SCORED。
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE transactions SET status='QUEUED' WHERE id=%s AND status='RECEIVED'",
                (transaction_id,),
            )
            conn.commit()
    # 返回任务受理信息，不代表模型已处理完成；最终状态需调用查询接口。
    return {
        "external_id": item.external_id,
        "transaction_id": transaction_id,
        "status": "QUEUED",
    }

# 9. 查询交易：按业务编号读取交易及最新一次预测结果。
@app.get("/transactions/{external_id}")
def get_transaction(external_id: str):
    # LEFT JOIN 保留尚无预测的交易；LATERAL 子查询只取最新一条预测。
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
    # 未找到交易返回 404；尚未预测的交易仍正常返回，预测字段为 null。
    if row is None:
        raise HTTPException(status_code=404, detail="transaction not found")
    # 字段名与 SELECT 顺序一一对应，将查询元组转换为响应字典。
    keys = [
        "external_id", "transaction_timestamp", "from_bank", "from_account",
        "to_bank", "to_account", "amount_received", "receiving_currency",
        "amount_paid", "payment_currency", "payment_format", "is_laundering",
        "status", "risk_probability", "risk_level", "model_version",
        "predicted_at",
    ]
    return dict(zip(keys, row))