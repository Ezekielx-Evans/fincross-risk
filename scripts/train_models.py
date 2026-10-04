import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
# 先比较这些树数；XGBoost 搜索到指定轮数为止。
RF_COUNTS = [50, 100, 150, 200, 300]
XGB_MAX_ROUNDS = 300


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
    batch_no = 0
    for batch in pq.ParquetFile(DATA).iter_batches(batch_size=BATCH_SIZE, columns=columns):
        batch_no += 1
        part = batch.to_pandas()
        scanned += len(part)
        # 按时间戳筛选，不依据正常或洗钱标签排除记录。
        time = part["transaction_timestamp"]
        part = part.loc[(time >= PERIOD_START) & (time < PERIOD_END)].copy()
        eligible += len(part)
        if batch_no == 1 or batch_no % 10 == 0:
            print(f"[数据读取] 已扫描 {scanned:,} 行，保留 {eligible:,} 行", flush=True)
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
    print("[数据读取] 正在合并并排序筛选后的交易", flush=True)
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
    print(f"[数据读取] 完成，共保留 {len(data):,} 行", flush=True)
    return data, metadata


# 按时间顺序划分训练集、验证集和测试集，比例约为 60%、20%、20%。
def split_by_time(df, check_test=False):
    # 找到约 60% 和 80% 处的时间，再回到该时间首次出现的位置。
    # 同一分钟的交易不会拆到两个集合中，因此实际比例可能略有偏差。
    if df.empty:
        raise ValueError("主要交易时段内没有可用交易")
    times = df["transaction_timestamp"]
    a = int(times.searchsorted(times.iloc[int(len(df) * 0.6)], side="left"))
    b = int(times.searchsorted(times.iloc[int(len(df) * 0.8)], side="left"))
    parts = {"train": df.iloc[:a], "validation": df.iloc[a:b], "test": df.iloc[b:]}
    # 选择阶段只检查训练集和验证集，不使用测试集标签选型。
    for name in ("train", "validation"):
        part = parts[name]
        if part.empty or part["is_laundering"].nunique() != 2:
            raise ValueError(f"{name} 缺少某一类别；请检查各时间段的类别分布")
    if check_test:
        test = parts["test"]
        if test.empty or test["is_laundering"].nunique() != 2:
            raise ValueError("test 缺少某一类别；无法完成最终评估")
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


# sklearn 的 AP 作为 XGBoost 每一轮的验证指标。
def validation_ap(y_true, y_prob):
    return average_precision_score(y_true, y_prob)


# 左图比较随机森林候选树数，右图展示 XGBoost 每一轮的 AP。
def plot_validation_ap(search, best_by_family):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    baseline = best_by_family["logistic-regression"]["pr_auc"]
    for ax, name, label in [
        (axes[0], "random-forest", "Random forest"),
        (axes[1], "xgboost", "XGBoost"),
    ]:
        xs = [item["n_estimators"] for item in search[name]]
        ys = [item["pr_auc"] for item in search[name]]
        ax.plot(xs, ys, marker="o" if name == "random-forest" else None,
                markersize=4, label=label)
        chosen = best_by_family[name]
        chosen_ap = (ys[chosen["n_estimators"] - 1] if name == "xgboost"
                     else chosen["pr_auc"])
        ax.scatter(chosen["n_estimators"], chosen_ap, s=90,
                   facecolors="none", edgecolors="black", linewidths=1.5,
                   zorder=3, label="Selected")
        ax.axhline(baseline, color="gray", linestyle="--",
                   label="Logistic regression")
        ax.set_xlim(0, max(xs) * 1.04)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Number of trees" if name == "random-forest"
                      else "Boosting round")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_xticks(RF_COUNTS)
    axes[0].set_ylabel("Validation AP")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "validation-ap-by-estimators.png", dpi=180)
    plt.close(fig)


# 使用验证集每个不同的预测概率对应的阈值绘制 F1 曲线。
def plot_validation_f1(y_true, y_prob, selected_threshold):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    p, r = precision[:-1], recall[:-1]
    scores = np.divide(2 * p * r, p + r,
                       out=np.zeros_like(p), where=(p + r) > 0)
    best_index = np.searchsorted(thresholds, selected_threshold)
    # 点数过多时仅为作图抽稀；最佳阈值点始终保留。
    step = max(1, len(thresholds) // 2000)
    shown = np.unique(np.r_[np.arange(0, len(thresholds), step),
                            best_index, len(thresholds) - 1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds[shown], scores[shown], label="Validation F1")
    ax.scatter(selected_threshold, scores[best_index], color="tab:orange",
               zorder=3, label=f"Selected: {selected_threshold:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Validation F1")
    ax.set_title("Validation F1 by threshold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "validation-f1-by-threshold.png", dpi=180)
    plt.close(fig)


def main_select():
    # 第一阶段仅用训练集与验证集确定模型、树数/轮数和阈值。
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[select] 开始读取标准化数据", flush=True)
    df, metadata = load_training_data()
    parts = split_by_time(df)
    print(
        f"[select] 已按时间划分：训练集 {len(parts['train']):,} 行，"
        f"验证集 {len(parts['validation']):,} 行",
        flush=True,
    )
    # 只记录训练集和验证集统计，测试集留到最终评估阶段。
    metadata["splits"] = {
        name: {
            "rows": len(parts[name]),
            "risk_rows": int(parts[name]["is_laundering"].sum()),
            "start": parts[name]["transaction_timestamp"].min().isoformat(),
            "end": parts[name]["transaction_timestamp"].max().isoformat(),
        }
        for name in ("train", "validation")
    }
    # X 保存九列输入特征，y 保存真实标签：正常为 0，风险为 1。
    X_train = build_features(parts["train"])
    y_train = parts["train"]["is_laundering"].astype(int)
    X_val = build_features(parts["validation"])
    y_val = parts["validation"]["is_laundering"].astype(int).to_numpy()
    print("[select] 特征构建完成，开始比较候选模型", flush=True)

    # 逻辑回归作为基线；随机森林比较预设的候选树数。
    candidate_counts = {
        "logistic-regression": [None],
        "random-forest": RF_COUNTS,
    }
    all_metrics = {}
    parameter_search = {}
    X_train_values = X_train[FEATURE_NAMES]
    X_val_values = X_val[FEATURE_NAMES]

    for name, counts in candidate_counts.items():
        trials = []
        best_result = None
        for count in counts:
            model_label = "逻辑回归" if name == "logistic-regression" else f"随机森林（{count} 棵树）"
            print(f"[select] 开始训练 {model_label}", flush=True)
            if name == "logistic-regression":
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=500, class_weight="balanced"),
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=count, random_state=SEED,
                    class_weight="balanced", n_jobs=-1,
                )
            model.fit(X_train_values, y_train)
            probability = model.predict_proba(X_val_values)[:, 1]
            result = {
                "positive_rate": float(y_val.mean()),
                "roc_auc": float(roc_auc_score(y_val, probability)),
                "pr_auc": float(average_precision_score(y_val, probability)),
                "features": FEATURE_NAMES,
            }
            if count is not None:
                result["n_estimators"] = count
            trials.append(result)
            print(
                f"[select] {model_label} 完成，验证集 AP={result['pr_auc']:.4f}",
                flush=True,
            )
            # AP 并列时不更新，保留较少的树数。
            if best_result is None or result["pr_auc"] > best_result["pr_auc"]:
                best_result = result
                joblib.dump(model, MODEL_DIR / f"{name}.pkl")
        parameter_search[name] = trials
        all_metrics[name] = best_result

    # XGBoost 训练到搜索上限，每轮均在同一验证集计算 AP。
    print(f"[select] 开始训练 XGBoost，共 {XGB_MAX_ROUNDS} 轮；每 50 轮输出一次验证结果", flush=True)
    xgb_params = dict(max_depth=5, learning_rate=0.08,
                      subsample=0.9, colsample_bytree=0.9, random_state=SEED)
    search_model = XGBClassifier(n_estimators=XGB_MAX_ROUNDS,
                                 eval_metric=validation_ap, **xgb_params)
    search_model.fit(X_train_values, y_train,
                     eval_set=[(X_val_values, y_val)], verbose=50)
    ap_history = search_model.evals_result()["validation_0"]["validation_ap"]
    if len(ap_history) != XGB_MAX_ROUNDS or not np.isfinite(ap_history).all():
        raise ValueError("XGBoost 逐轮验证指标缺失或无效")
    # 并列时取第一个最大值，即较少的轮数。
    best_round = int(np.argmax(ap_history)) + 1
    print(f"[select] XGBoost 搜索完成，最佳轮数为 {best_round}；正在按该轮数重新训练", flush=True)
    parameter_search["xgboost"] = [
        {"n_estimators": i, "pr_auc": float(ap)}
        for i, ap in enumerate(ap_history, start=1)
    ]
    # 将最佳轮数重新训练并保存，部署预测也只使用选中的轮数。
    xgb_model = XGBClassifier(n_estimators=best_round,
                              eval_metric="logloss", **xgb_params)
    xgb_model.fit(X_train_values, y_train)
    xgb_prob = xgb_model.predict_proba(X_val_values)[:, 1]
    all_metrics["xgboost"] = {
        "n_estimators": best_round,
        "positive_rate": float(y_val.mean()),
        "roc_auc": float(roc_auc_score(y_val, xgb_prob)),
        "pr_auc": float(average_precision_score(y_val, xgb_prob)),
        "features": FEATURE_NAMES,
    }
    joblib.dump(xgb_model, MODEL_DIR / "xgboost.pkl")
    print(f"[select] XGBoost 训练完成，验证集 AP={all_metrics['xgboost']['pr_auc']:.4f}", flush=True)

    # 每种模型先选出最佳配置，再比较三个模型的验证集 AP。
    print("[select] 正在比较三个模型并选择最终模型", flush=True)
    plot_validation_ap(parameter_search, all_metrics)
    best_name = max(all_metrics, key=lambda name: all_metrics[name]["pr_auc"])
    best_model = joblib.load(MODEL_DIR / f"{best_name}.pkl")
    joblib.dump(best_model, MODEL_DIR / "model.pkl")

    # 只对选中的模型搜索阈值，使用验证集概率，不重新训练模型。
    validation_probability = best_model.predict_proba(X_val[FEATURE_NAMES])[:, 1]
    selected_threshold, candidate_count = select_threshold(y_val, validation_probability)
    print(f"[select] 已选模型：{best_name}；正在选择分类阈值", flush=True)
    plot_validation_f1(y_val, validation_probability, selected_threshold)
    validation_selected = metrics(y_val, validation_probability, selected_threshold)

    # 记录模型文件摘要，最终评估时核对是否仍为选中的模型。
    with (MODEL_DIR / "model.pkl").open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    # 先保存验证集选择结果；此阶段不计算测试集指标。
    output = {
        "best_model": best_name,
        "model_sha256": model_sha256,
        "selection_metric": "validation average precision (pr_auc field)",
        "models": all_metrics,
        "parameter_search": parameter_search,
        "threshold_selection": {
            "dataset": "validation",
            "metric": "f1",
            "candidate_rule": "unique validation predicted probabilities",
            "candidate_count": candidate_count,
            "tie_break": "highest threshold among equal maximum F1 scores",
            "selected_threshold": selected_threshold,
            "validation_selected": validation_selected,
        },
        "data": metadata,
        # 记录训练环境，用于核对部署依赖和复现实验。
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in
                ["scikit-learn", "xgboost", "numpy", "scipy", "joblib", "pandas", "pyarrow", "matplotlib"]
            },
        },
    }
    # 将评估结果写入 JSON，保留中文并缩进排版。
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("[select] 验证集选择完成，结果和图像已保存", flush=True)
    print("best_model:", best_name)
    print("selected_threshold:", selected_threshold)
    print("验证集结果已保存；确认候选范围后再运行 test。")


# 第二阶段只加载已选好的模型和阈值，评估一次测试集。
def main_test():
    result_path = MODEL_DIR / "metrics.json"
    if not result_path.is_file() or not (MODEL_DIR / "model.pkl").is_file():
        raise FileNotFoundError("请先运行 select，生成模型和验证集记录")
    print("[test] 开始加载模型并准备测试集", flush=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "test" in result:
        raise ValueError("测试集已评估；不要用测试结果反复调整参数")
    with (MODEL_DIR / "model.pkl").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != result["model_sha256"]:
        raise ValueError("模型文件与验证集选择记录不一致，请重新运行 select")

    df, metadata = load_training_data()
    parts = split_by_time(df, check_test=True)
    previous = dict(result["data"])
    previous_splits = previous.pop("splits")
    if metadata != previous:
        raise ValueError("数据来源或筛选条件发生变化，请重新运行 select")
    for name in ("train", "validation"):
        part = parts[name]
        saved = previous_splits[name]
        if (len(part) != saved["rows"] or
                part["transaction_timestamp"].min().isoformat() != saved["start"] or
                part["transaction_timestamp"].max().isoformat() != saved["end"]):
            raise ValueError("数据划分发生变化，请重新运行 select")

    # 不再重新训练或选阈值；只用已有模型预测测试集。
    best_model = joblib.load(MODEL_DIR / "model.pkl")
    threshold = result["threshold_selection"]["selected_threshold"]
    test = parts["test"]
    X_test = build_features(test)
    y_test = test["is_laundering"].astype(int).to_numpy()
    print(f"[test] 开始预测测试集，共 {len(test):,} 行", flush=True)
    probability = best_model.predict_proba(X_test[FEATURE_NAMES])[:, 1]
    result["test"] = metrics(y_test, probability, threshold)
    result["data"]["splits"]["test"] = {
        "rows": len(test),
        "risk_rows": int(test["is_laundering"].sum()),
        "start": test["transaction_timestamp"].min().isoformat(),
        "end": test["transaction_timestamp"].max().isoformat(),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("[test] 测试集评估完成，结果已写入 metrics.json", flush=True)
    print("test:", result["test"])


# 显式指定阶段，避免选型时提前查看测试结果。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "test"])
    args = parser.parse_args()
    if args.stage == "select":
        main_select()
    else:
        main_test()