import json
import os
import time

import psycopg
import redis
import requests
from kafka import KafkaConsumer


# 读取数据库、Redis、Kafka 和模型服务的连接配置。
DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transaction-analysis")
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:8000")


# 创建 Redis 客户端，将读取的内容解码为字符串。
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


# 创建 PostgreSQL 连接。
def db():
    return psycopg.connect(DATABASE_URL)


# 处理一条交易分析任务。
def process_message(message):
    # 读取业务交易编号和数据库内部编号。
    payload = message.value
    external_id = payload["external_id"]
    transaction_id = int(payload["transaction_id"])

    # 使用业务交易编号生成 Redis 完成标记的键名。
    done_key = f"fincross:transaction:{external_id}:done"

    # 已有完成标记时跳过预测和数据库写入。
    if redis_client.exists(done_key):
        print(f"skip duplicated message external_id={external_id}", flush=True)
        return

    # 将任务中的九个特征发送给模型服务，等待预测结果。
    response = requests.post(
        f"{MODEL_SERVICE_URL}/predict",
        json=payload["features"],
        timeout=15,
    )

    # HTTP 请求失败时抛出异常，成功时读取 JSON 结果。
    response.raise_for_status()
    result = response.json()

    # 在同一个事务中保存预测结果并更新交易状态。
    with db() as conn:
        with conn.cursor() as cur:
            # 将模型版本、风险概率和风险等级写入预测表。
            cur.execute(
                """INSERT INTO predictions
                   (transaction_id, model_version, risk_probability, risk_level)
                   VALUES (%s,%s,%s,%s)""",
                (transaction_id, result["model_version"],
                 result["risk_probability"], result["risk_level"]),
            )

            # 将交易标记为已完成评分。
            cur.execute(
                "UPDATE transactions SET status='SCORED' WHERE id=%s",
                (transaction_id,),
            )

        # 两项写入均成功后提交事务；发生异常则回滚。
        conn.commit()

    # 数据库提交成功后保存完成标记，有效期为一天。
    redis_client.set(done_key, "1", ex=86400)

    # 输出处理结果，flush=True 让容器日志及时显示。
    print(
        f"scored external_id={external_id} "
        f"probability={result['risk_probability']:.6f}",
        flush=True,
    )


# 启动消费者，持续读取 Kafka 任务。
def main():
    consumer = KafkaConsumer(
        # 订阅 Data Service 发布任务的主题。
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,

        # 同一消费组的 Worker 按分区分担任务。
        group_id="fincross-risk-worker",

        # 关闭自动提交，由代码在处理成功后提交进度。
        enable_auto_commit=False,

        # 没有有效消费进度时，从仍保留的最早消息开始读取。
        auto_offset_reset="earliest",

        # 将消息中的 JSON 字节解码为 Python 字典。
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    print("worker started", flush=True)

    # 逐条处理收到的消息。
    for message in consumer:
        try:
            process_message(message)

            # 处理完成后提交消费进度，供后续恢复消费使用。
            consumer.commit()
        except Exception as exc:
            # 记录错误并等待三秒；此处不会自动重试当前消息。
            print(f"processing failed: {exc}", flush=True)
            time.sleep(3)


# 直接运行 worker.py 时启动消费循环。
if __name__ == "__main__":
    main()