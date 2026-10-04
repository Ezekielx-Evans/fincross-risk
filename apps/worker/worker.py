import json
import os

import psycopg
import redis
import requests
from kafka import KafkaConsumer
from kafka.structs import OffsetAndMetadata, TopicPartition

# 从环境变量读取连接配置；默认主机名对应 Compose 中的服务名。
# DATABASE_URL 必须配置，其余连接参数提供默认值。
DATABASE_URL = os.environ["DATABASE_URL"]
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:19092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transaction-analysis")
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:8000")

# 创建 Redis 客户端，设置连接和读取超时，故障时回退到数据库。
redis_client = redis.from_url(
    REDIS_URL, decode_responses=True,
    socket_connect_timeout=2, socket_timeout=2,
)

def db():
    # 建立数据库连接，交给调用处的 with 语句管理关闭。
    return psycopg.connect(DATABASE_URL)

def cache_completed(done_key):
    # 仅在数据库确认完成后调用；标记保留 24 小时。
    try:
        redis_client.set(done_key, "1", ex=86400)
    except redis.RedisError as exc:
        # 缓存故障不影响已保存的结果，也不阻止提交消费进度。
        print(f"completion cache write unavailable: {exc}", flush=True)

def process_message(message):
    # 消息已解析为字典；external_id 是业务编号，transaction_id 是数据库主键。
    payload = message.value
    external_id = payload["external_id"]
    transaction_id = int(payload["transaction_id"])
    # 同时使用数据库主键和业务编号；v2 命名空间不读取旧版完成标记。
    done_key = f"fincross:completion:v2:{transaction_id}:{external_id}"

    # 命中已完成标记时直接跳过，不再查询数据库和调用模型。
    try:
        if redis_client.get(done_key) == "1":
            print(f"skip cached transaction external_id={external_id}", flush=True)
            return
    except redis.RedisError as exc:
        # 缓存不可用时继续查库，不能据此认定交易尚未处理。
        print(f"completion cache read unavailable: {exc}", flush=True)

    # 标记过期、缺失或缓存故障时，以数据库状态确认是否完成。
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM transactions WHERE id=%s", (transaction_id,))
            row = cur.fetchone()
    # 交易不存在时终止处理；已完成的交易直接跳过，避免重复调用模型。
    if row is None:
        raise ValueError(f"transaction not found: {transaction_id}")
    if row[0] == "SCORED":
        # 数据库已确认完成，补写缓存以便后续重复消息直接跳过。
        cache_completed(done_key)
        print(f"skip completed transaction external_id={external_id}", flush=True)
        return

    # 将交易特征作为 JSON 发送到模型接口，连接和读取超时均设为 15 秒。
    response = requests.post(
        f"{MODEL_SERVICE_URL}/predict",
        json=payload["features"],
        timeout=15,
    )
    # HTTP 4xx/5xx 响应抛出异常；成功后读取概率、等级和模型版本。
    response.raise_for_status()
    result = response.json()

    # 预测结果和交易状态在同一事务中写入，任一步失败都会回滚。
    with db() as conn:
        with conn.cursor() as cur:
            # 同一交易、同一模型版本已有预测时，不再重复插入。
            cur.execute(
                """INSERT INTO predictions
                   (transaction_id, model_version, risk_probability, risk_level)
                   VALUES (%s,%s,%s,%s)
                   ON CONFLICT (transaction_id, model_version) DO NOTHING""",
                (transaction_id, result["model_version"],
                 result["risk_probability"], result["risk_level"]),
            )
            # 将交易标记为已评分，同时更新处理时间。
            cur.execute(
                "UPDATE transactions SET status='SCORED', updated_at=NOW() WHERE id=%s",
                (transaction_id,),
            )
        # 两项写入均成功后提交，确保状态与预测结果一致。
        conn.commit()

    # 必须先提交数据库，再写完成标记，避免未落库的任务被提前跳过。
    cache_completed(done_key)
    # 输出交易编号与风险概率，并立即刷新日志。
    print(
        f"scored external_id={external_id} "
        f"probability={result['risk_probability']:.6f}",
        flush=True,
    )

def main():
    # 订阅交易分析主题，持续接收待评分的交易任务。
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        # 同一消费组内的多个 Worker 可以分担不同分区。
        group_id="fincross-risk-worker",
        # 关闭自动提交，待业务处理成功后再记录消费进度。
        enable_auto_commit=False,
        # 没有有效的已提交位点时，从分区中最早保留的消息开始读取。
        auto_offset_reset="earliest",
        # 将消息的 UTF-8 字节解码，再把 JSON 解析为 Python 字典。
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    print("worker started", flush=True)
    try:
        for message in consumer:
            # 按顺序处理消息；异常向外抛出并退出，当前消息不提交消费进度。
            process_message(message)
            # offset 表示消息在分区中的位置；提交值为下次应读取的位置。
            # 只更新当前分区，避免提交其他分区尚未处理的消息。
            partition = TopicPartition(message.topic, message.partition)
            consumer.commit({partition: OffsetAndMetadata(message.offset + 1, "")})
    finally:
        # 不在关闭时提交未处理消息；Compose 会重启异常退出的 Worker。
        consumer.close(autocommit=False)

# 直接运行脚本时启动消费循环。
if __name__ == "__main__":
    main()
