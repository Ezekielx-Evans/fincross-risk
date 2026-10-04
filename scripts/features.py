import numpy as np
import pandas as pd

# 固定九个特征的名称和顺序，训练与预测时保持一致。
FEATURE_NAMES = [
    "amount_paid", "amount_received", "amount_paid_log",
    "amount_received_log", "hour", "weekday", "cross_bank_flag",
    "cross_currency_flag", "payment_format_code",
]

# 将支付方式转换为固定编号；未知值在转换时统一填为 0。
# 编号只用于区分支付方式，不表示金额大小或风险高低。
PAYMENT_FORMAT_CODES = {
    "ACH": 1,
    "CREDIT CARD": 2,
    "CHEQUE": 3,
    "CASH": 4,
    "WIRE": 5,
    "REINVESTMENT": 6,
    "BITCOIN": 7,
}

# 将交易表转换为九列特征，每行仍对应原来的一笔交易。
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # 沿用原表的行索引，使特征与交易及训练标签一一对应。
    result = pd.DataFrame(index=df.index)

    # 将金额转换为数值，无法转换的内容及缺失值填为 0。
    result["amount_paid"] = pd.to_numeric(
        df["amount_paid"], errors="coerce"
    ).fillna(0)
    result["amount_received"] = pd.to_numeric(
        df["amount_received"], errors="coerce"
    ).fillna(0)

    # 增加金额对数特征，压缩金额之间的数量级差异。
    # clip 将负数截为 0，log1p 计算 ln(1 + x)，零金额也能计算。
    result["amount_paid_log"] = np.log1p(result["amount_paid"].clip(lower=0))
    result["amount_received_log"] = np.log1p(
        result["amount_received"].clip(lower=0)
    )

    # 解析交易时间；无时区的时间按 UTC 处理，已有时区的时间转换为 UTC。
    # 无法解析的时间记为缺失值，提取特征时填为 0。
    timestamp = pd.to_datetime(
        df["transaction_timestamp"], errors="coerce", utc=True
    )

    # 提取小时（0—23）和星期（周一为 0，周日为 6）。
    result["hour"] = timestamp.dt.hour.fillna(0)
    result["weekday"] = timestamp.dt.weekday.fillna(0)

    # 比较双方银行编号：不同记为 1，相同记为 0。
    result["cross_bank_flag"] = (
        df["from_bank"].astype("string").str.strip()
        != df["to_bank"].astype("string").str.strip()
    ).astype(float)

    # 去除币种两端空格并统一为大写，再判断是否跨币种。
    # 支付币种与接收币种不同记为 1，相同记为 0。
    result["cross_currency_flag"] = (
        df["payment_currency"].astype("string").str.strip().str.upper()
        != df["receiving_currency"].astype("string").str.strip().str.upper()
    ).astype(float)

    # 统一支付方式的文本格式，按编码表转换，未知或缺失值填为 0。
    result["payment_format_code"] = (
        df["payment_format"].astype("string").str.strip().str.upper()
        .map(PAYMENT_FORMAT_CODES).fillna(0)
    )

    # 按固定顺序返回九列特征，不包含交易编号、来源信息和训练标签。
    return result[FEATURE_NAMES]