import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score, precision_recall_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# 复用统一的特征转换函数和列顺序。
from scripts.features import build_features, FEATURE_NAMES

# 定位项目根目录、标准化数据和模型输出目录。
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/aml_transactions.parquet"
MODEL_DIR = ROOT / "models"

# 核对数据来源，并限定 Small 数据集的主要交易时段。
SOURCE_NAME = "HI-Small_Trans.csv"
PERIOD_START = pd.Timestamp("2022-09-01", tz="UTC")
# 结束边界取次日零点，筛选时不包含该边界，确保保留 9 月 10 日全天。
PERIOD_END = pd.Timestamp("2022-09-11", tz="UTC")
# 每批读取十万笔，固定随机种子用于模型参数和结果复现。
BATCH_SIZE = 100_000
SEED = 42


# 分批读取数据，保留主要交易时段内的全部记录，并记录筛选数量。
def load_training_data():
    # 读取标准化阶段的统计文件，核对原始数据文件名。
    info = json.loads(DATA.with_suffix(".info.json").read_text(encoding="utf-8"))
    if info["source_file"] != SOURCE_NAME:
        raise ValueError("源文件与训练配置不一致，请核对文件名和主要交易时段")
    # 累计读取总行数和主要交易时段内的交易数。
    selected_parts = []
    scanned = eligible = 0
    # 只读取特征计算、标签和排序需要的列。
    columns = [
        "transaction_timestamp", "source_row_number", "amount_paid",
        "amount_received", "from_bank", "to_bank", "payment_currency",
        "receiving_currency", "payment_format", "is_laundering",
    ]

    # 分批读取并筛选；后续仍需将保留的全部交易合并到内存。
    for batch in pq.ParquetFile(DATA).iter_batches(batch_size=BATCH_SIZE, columns=columns):
        part = batch.to_pandas()
        scanned += len(part)
        # 按时间戳筛选，不依据正常或洗钱标签排除记录。
        time = part["transaction_timestamp"]
        part = part.loc[(time >= PERIOD_START) & (time < PERIOD_END)].copy()
        eligible += len(part)
        if part.empty:
            continue
        # 保存当前时段内的交易块，后面合并为完整训练数据。
        selected_parts.append(part)

    # 核对全文件行数，并检查主要交易时段内是否存在可用记录。
    if scanned != info["saved_rows"]:
        raise ValueError("Parquet 行数与来源记录不一致，请重新标准化")
    if not selected_parts:
        raise ValueError("指定的主要交易时段内没有可用交易")
    # 合并主要交易时段内的全部记录，再按时间排序。
    data = pd.concat(selected_parts, ignore_index=True).sort_values(
        ["transaction_timestamp", "source_row_number"], kind="stable",
    ).reset_index(drop=True)
    # 记录来源哈希、日期边界、筛选数量和批处理配置，便于复现实验。
    metadata = {
        "source_file": SOURCE_NAME,
        "source_sha256": info["source_sha256"],
        "scanned_rows": scanned,
        "eligible_rows": eligible,
        "excluded_by_period": scanned - eligible,
        "training_rows": len(data),
        "batch_size": BATCH_SIZE,
        "period_start_inclusive": PERIOD_START.isoformat(),
        "period_end_exclusive": PERIOD_END.isoformat(),
        "selection": "all records in the primary period",
    }
    return data, metadata


# 按时间顺序划分训练集、验证集和测试集，比例约为 60%、20%、20%。
def split_by_time(df):
    # 找到约 60% 和 80% 处的时间，再回到该时间首次出现的位置。
    # 同一分钟的交易不会拆到两个集合中，因此实际比例可能略有偏差。
    if df.empty:
        raise ValueError("主要交易时段内没有可用交易")
    times = df["transaction_timestamp"]
    a = int(times.searchsorted(times.iloc[int(len(df) * 0.6)], side="left"))
    b = int(times.searchsorted(times.iloc[int(len(df) * 0.8)], side="left"))
    parts = {"train": df.iloc[:a], "validation": df.iloc[a:b], "test": df.iloc[b:]}
    # 三个集合都需要包含正常和风险交易，才能完成后续训练与评估。
    for name, part in parts.items():
        if part.empty or part["is_laundering"].nunique() != 2:
            raise ValueError(f"{name} 缺少某一类别；请检查各时间段的类别分布")
    return parts


# 使用验证集选出的阈值计算分类指标，并记录概率评估指标。
def metrics(y_true, y_prob, threshold):
    # 概率达到阈值记为风险，否则记为正常。
    # 阈值影响 Precision、Recall 和 F1，不影响 AP、ROC-AUC 或原始概率。
    y_pred = (y_prob >= threshold).astype(int)
    return {
        # 记录这组分类指标使用的阈值。
        "threshold": float(threshold),
        # 真实标签中风险交易的比例。
        "positive_rate": float(y_true.mean()),
        # Precision：判为风险的交易中，实际为风险的比例。
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        # Recall：实际风险交易中，被识别出来的比例。
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        # F1：综合 Precision 和 Recall；无法计算时返回 0。
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # ROC-AUC 和 AP 直接使用概率，评估不同阈值下的区分能力。
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        # 沿用 pr_auc 字段名，实际保存的是 Average Precision（AP）。
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


# 仅使用验证集，为已选中的模型寻找 F1 最高的阈值。
def select_threshold(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    # 标签和概率必须一一对应，且验证集同时包含正常和风险交易。
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("标签和概率必须是长度相同的一维数组")
    if np.unique(y_true).tolist() != [0, 1]:
        raise ValueError("阈值选择需要同时包含正常和风险交易")
    if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("预测概率必须是 0 到 1 之间的有限数")

    # 按各个不同的预测概率计算 P、R，同分交易使用同一个阈值。
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # 最后一组 P、R 没有对应阈值，去掉后再计算各阈值的 F1。
    precision = precision[:-1]
    recall = recall[:-1]
    scores = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) > 0,
    )
    # thresholds 从小到大排列；F1 并列最高时选较高阈值，固定选择规则。
    best_index = np.flatnonzero(scores == scores.max())[-1]
    return float(thresholds[best_index]), int(len(thresholds))


def main():
    # 创建输出目录，读取样本并按时间划分。
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df, metadata = load_training_data()
    parts = split_by_time(df)
    # 分别记录三个集合的交易数、风险交易数和时间范围。
    metadata["splits"] = {
        name: {
            "rows": len(part),
            "risk_rows": int(part["is_laundering"].sum()),
            "start": part["transaction_timestamp"].min().isoformat(),
            "end": part["transaction_timestamp"].max().isoformat(),
        }
        for name, part in parts.items()
    }
    # X 保存九列输入特征，y 保存真实标签：正常为 0，风险为 1。
    X_train = build_features(parts["train"])
    y_train = parts["train"]["is_laundering"].astype(int)
    X_val = build_features(parts["validation"])
    y_val = parts["validation"]["is_laundering"].astype(int).to_numpy()

    # 使用相同的数据比较三个模型；预处理和参数由各模型分别配置。
    models = {
        # 逻辑回归先缩放特征；缩放参数只从训练集学习，并随模型一起保存。
        # balanced 提高少数类别的权重，max_iter 限制优化迭代次数。
        "logistic-regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced"),
        ),
        # 随机森林使用 150 棵树，启用类别权重，并使用全部可用 CPU 核心。
        "random-forest": RandomForestClassifier(
            n_estimators=150, random_state=SEED, class_weight="balanced", n_jobs=-1,
        ),
        # XGBoost 先使用最多 150 轮的初始配置；正式实验需比较不同轮数。
        # 树深度最多为 5，学习率控制每轮的分数调整幅度。
        # 每轮采样 90% 的记录和特征，logloss 用于评估概率预测的误差。
        "xgboost": XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=SEED,
        ),
    }

    # 依次训练三个模型；先计算不依赖分类阈值的指标，按验证集 AP 选模型。
    all_metrics = {}
    for name, model in models.items():
        # 使用训练特征和真实标签学习模型参数。
        model.fit(X_train[FEATURE_NAMES], y_train)
        # 在验证集上预测；[:, 1] 取出每笔交易对应标签 1 的风险概率。
        probability = model.predict_proba(X_val[FEATURE_NAMES])[:, 1]
        result = {
            "positive_rate": float(y_val.mean()),
            "roc_auc": float(roc_auc_score(y_val, probability)),
            # pr_auc 字段保存 AP，用于比较候选模型。
            "pr_auc": float(average_precision_score(y_val, probability)),
            "features": FEATURE_NAMES,
        }
        all_metrics[name] = result
        # 分别保存候选模型，便于后续比较和加载。
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")
        print(name, json.dumps(result, ensure_ascii=False, indent=2))

    # 仅按验证集 AP 选择最佳模型，并保存为服务加载的 model.pkl。
    best_name = max(all_metrics, key=lambda name: all_metrics[name]["pr_auc"])
    best_model = models[best_name]
    joblib.dump(best_model, MODEL_DIR / "model.pkl")

    # 只对选中的模型搜索阈值，使用验证集概率，不重新训练模型。
    validation_probability = best_model.predict_proba(X_val[FEATURE_NAMES])[:, 1]
    selected_threshold, candidate_count = select_threshold(y_val, validation_probability)
    validation_selected = metrics(y_val, validation_probability, selected_threshold)

    # 模型和阈值确定后才评估测试集，仅使用验证集选出的阈值。
    X_test = build_features(parts["test"])
    y_test = parts["test"]["is_laundering"].astype(int).to_numpy()
    test_probability = best_model.predict_proba(X_test[FEATURE_NAMES])[:, 1]
    test_metrics = metrics(y_test, test_probability, selected_threshold)

    # 保存阈值的选择依据、验证结果和最终测试指标，便于复现实验。
    output = {
        "best_model": best_name,
        "selection_metric": "validation average precision (pr_auc field)",
        "models": all_metrics,
        "threshold_selection": {
            "dataset": "validation",
            "metric": "f1",
            "candidate_rule": "unique validation predicted probabilities",
            "candidate_count": candidate_count,
            "tie_break": "highest threshold among equal maximum F1 scores",
            "selected_threshold": selected_threshold,
            "validation_selected": validation_selected,
        },
        "test": test_metrics,
        "data": metadata,
        # 记录训练环境，用于核对部署依赖和复现实验。
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in
                ["scikit-learn", "xgboost", "numpy", "scipy", "joblib", "pandas", "pyarrow"]
            },
        },
    }
    # 将评估结果写入 JSON，保留中文并缩进排版。
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("best_model:", best_name)
    print("selected_threshold:", selected_threshold)
    print("test:", test_metrics)
    print("data:", metadata)


# 以模块方式运行时执行训练入口。
if __name__ == "__main__":
    main()
