import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# 根据当前脚本的位置定位项目根目录。
ROOT = Path(__file__).resolve().parents[1]

# 明确指定原始交易文件，不自动读取目录中的其他数据文件。
SOURCE_FILE = ROOT / "data/source/ibm-aml-data/HI-Small_Trans.csv"

# 保存标准化后的数据和处理统计信息。
OUT = ROOT / "data/raw/aml_transactions.parquet"

# 每次读取十万行，避免一次性加载全部交易。
CHUNK_SIZE = 100_000


# 原始字段名与程序内部字段名的对应关系。
# 原始文件中有两列 Account，pandas 会将第二列读取为 Account.1。
COLUMN_MAP = {
    "Timestamp": "transaction_timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}


def normalize_chunk(df, first_row):
    # 清洗当前数据块，返回字段和类型统一的 DataFrame。

    # 检查表头，避免误把账户表或其他文件当作交易表读取。
    if list(df.columns) != list(COLUMN_MAP):
        raise ValueError(f"交易表头不匹配：{list(df.columns)}")

    # 统一字段名称，复制数据后再进行后续处理。
    result = df.rename(columns=COLUMN_MAP).copy()

    # 去除文本字段两端的空格，并将空字符串视为缺失值。
    for name in result.columns:
        result[name] = result[name].str.strip()
    result = result.replace("", pd.NA)

    # 记录原始 CSV 行号，便于之后追溯清洗后的记录。
    # CSV 第 1 行是表头，因此第一条交易记录从第 2 行开始。
    rows = np.arange(first_row, first_row + len(result))

    # external_id 用文件名和原始行号组成，保证记录具有稳定标识。
    result.insert(
        0,
        "external_id",
        [f"{SOURCE_FILE.stem}-{row_number}" for row_number in rows],
    )
    result.insert(1, "source_file", SOURCE_FILE.name)
    result.insert(2, "source_row_number", rows)

    # 将交易时间转换为统一的 UTC 时间。
    # 无法解析的时间会变成缺失值，后续统一删除。
    result["transaction_timestamp"] = pd.to_datetime(
        result["transaction_timestamp"],
        format="%Y/%m/%d %H:%M",
        errors="coerce",
        utc=True,
    )

    # 金额字段转成浮点数，便于后续特征计算。
    # 银行编号和账户编号不参与数值转换，避免丢失前导零。
    for name in ["amount_paid", "amount_received"]:
        result[name] = pd.to_numeric(
            result[name],
            errors="coerce",
        ).astype("float64")

    # 统一币种和支付方式的大小写。
    for name in [
        "payment_currency",
        "receiving_currency",
        "payment_format",
    ]:
        result[name] = result[name].str.upper()

    # 标签只能是 0 或 1。
    # 0 表示正常交易，1 表示风险交易。
    labels = result["is_laundering"]
    if not labels.isin(["0", "1"]).all():
        raise ValueError("Is Laundering 存在缺失值或非 0/1 标签")

    # 将字符串标签转换为布尔值，True 表示风险交易。
    result["is_laundering"] = labels.eq("1").astype(bool)

    # 删除关键字段缺失的记录。
    result = result.dropna(subset=list(COLUMN_MAP.values()))

    # 删除金额不是有限数值或小于零的记录。
    amounts = result[["amount_paid", "amount_received"]]
    valid_amounts = (
        np.isfinite(amounts).all(axis=1)
        & amounts.ge(0).all(axis=1)
    )

    return result.loc[valid_amounts].reset_index(drop=True)


def main():
    # 检查原始交易文件是否存在。
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(SOURCE_FILE)

    # 创建输出目录。
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # 先写入临时文件，处理成功后再替换正式文件。
    # 如果中途发生错误，不会覆盖已有的完整结果。
    temp = OUT.with_suffix(".parquet.tmp")

    writer = None
    read_rows = 0
    saved_rows = 0
    risk_rows = 0
    formats = {}

    try:
        # dtype="string" 让银行编号、账户编号等字段按文本读取，
        # 从源头保留类似 00123 这样的编号格式。
        for chunk in pd.read_csv(
            SOURCE_FILE,
            dtype="string",
            chunksize=CHUNK_SIZE,
        ):
            clean = normalize_chunk(
                chunk,
                first_row=read_rows + 2,
            )
            read_rows += len(chunk)

            # 当前数据块清洗后没有有效记录时，直接处理下一块。
            if clean.empty:
                continue

            # 将 pandas DataFrame 转换为 Arrow 表。
            table = pa.Table.from_pandas(
                clean,
                preserve_index=False,
            )

            # 第一个有效数据块用于确定 Parquet 文件的字段结构。
            if writer is None:
                writer = pq.ParquetWriter(
                    temp,
                    table.schema,
                    compression="snappy",
                )

            # 逐块写入同一个 Parquet 文件。
            writer.write_table(table)

            saved_rows += len(clean)
            risk_rows += int(clean["is_laundering"].sum())

            # 统计各种支付方式的数量。
            for name, count in clean["payment_format"].value_counts().items():
                formats[name] = formats.get(name, 0) + int(count)

            print(
                f"已读取 {read_rows:,} 行，"
                f"已保留 {saved_rows:,} 行",
                flush=True,
            )

    finally:
        # 无论处理成功还是失败，都关闭文件写入器。
        if writer is not None:
            writer.close()

    # 如果没有任何有效记录，不生成正式输出文件。
    if not saved_rows:
        raise ValueError("没有有效交易，未生成正式结果")

    # 临时文件完整生成后，再替换正式 Parquet 文件。
    temp.replace(OUT)

    # 计算原始 CSV 的 SHA-256，用于确认输入文件是否发生变化。
    with SOURCE_FILE.open("rb") as stream:
        digest = hashlib.file_digest(
            stream,
            "sha256",
        ).hexdigest()

    # 保存本次标准化处理的统计信息。
    info = {
        "source_file": SOURCE_FILE.name,
        "source_bytes": SOURCE_FILE.stat().st_size,
        "source_sha256": digest,
        "read_rows": read_rows,
        "saved_rows": saved_rows,
        "dropped_rows": read_rows - saved_rows,
        "risk_rows": risk_rows,
        "positive_rate": risk_rows / saved_rows,
        "payment_formats": formats,
    }

    # 将统计信息写入与 Parquet 文件同名的 JSON 文件。
    OUT.with_suffix(".info.json").write_text(
        json.dumps(
            info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("输出:", OUT)


if __name__ == "__main__":
    main()