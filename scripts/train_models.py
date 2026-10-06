import argparse
import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, fbeta_score, precision_score,
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
# F2 中 Recall 的权重高于 Precision，用于优先减少风险交易漏报。
F_BETA = 2.0
# 随机森林的默认候选树数。
RF_COUNTS = [50, 100, 150, 200, 300]
DEFAULT_ROUNDS = {"logistic-regression": 500, "random-forest": 300, "xgboost": 500}


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
    # 阈值影响 Precision、Recall 和 F2，不影响 AP、ROC-AUC 或原始概率。
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
        # F2：让 Recall 的权重高于 Precision；无法计算时返回 0。
        "f2": float(fbeta_score(y_true, y_pred, beta=F_BETA, zero_division=0)),
        # ROC-AUC 和 AP 直接使用概率，评估不同阈值下的区分能力。
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        # 沿用 pr_auc 字段名，实际保存的是 Average Precision（AP）。
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


# 仅使用验证集，为已选中的模型寻找 F2 最高的阈值。
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
    # 最后一组 P、R 没有对应阈值，去掉后再计算各阈值的 F2。
    precision = precision[:-1]
    recall = recall[:-1]
    beta_squared = F_BETA ** 2
    scores = np.divide(
        (1 + beta_squared) * precision * recall,
        beta_squared * precision + recall,
        out=np.zeros_like(precision),
        where=(beta_squared * precision + recall) > 0,
    )
    # thresholds 从小到大排列；F2 并列最高时选较高阈值，固定选择规则。
    best_index = np.flatnonzero(scores == scores.max())[-1]
    return float(thresholds[best_index]), int(len(thresholds))


# sklearn 的 AP 作为 XGBoost 每一轮的验证指标。
def validation_ap(y_true, y_prob):
    return average_precision_score(y_true, y_prob)


# 根据数据范围缩放纵轴，并允许命令行指定范围。
def set_score_axis(ax, values, y_min=None, y_max=None):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("绘图指标为空或包含无效数值")
    spread = float(values.max() - values.min())
    padding = max(spread * 0.1, 0.001)
    lower = (max(0.0, float(values.min()) - padding) if y_max is None else 0.0) if y_min is None else y_min
    upper = (min(1.0, float(values.max()) + padding) if y_min is None else 1.0) if y_max is None else y_max
    if not 0 <= lower < upper <= 1:
        raise ValueError("纵轴范围必须在 0 到 1 之间，且最小值小于最大值")
    ax.set_ylim(lower, upper)
    if lower > 0 or upper < 1:
        ax.text(0.99, 0.02, "Y axis zoomed", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="gray")


# 每个模型独立绘制验证集 AP；单点模型用散点显示。
def plot_validation_ap(name, search, chosen, output, y_min=None, y_max=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [item["n_estimators"] for item in search if "n_estimators" in item]
    ys = [item["pr_auc"] for item in search]
    if xs:
        ax.plot(xs, ys, marker="o" if name == "random-forest" else None,
                markersize=4, label="Validation AP")
        best_x = chosen["n_estimators"]
        ax.scatter(best_x, ys[xs.index(best_x)], s=90, facecolors="none",
                   edgecolors="black", linewidths=1.5, zorder=3,
                   label=f"Selected: {best_x}")
        ax.set_xlim(0, max(xs) * 1.04)
        ax.set_xlabel("Number of trees" if name == "random-forest"
                      else "Boosting round")
        if name == "random-forest":
            ax.set_xticks(xs)
    else:
        ax.scatter([1], ys, label=f"Validation AP: {ys[0]:.4f}")
        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1], ["Logistic regression"])
    set_score_axis(ax, ys, y_min, y_max)
    ax.set_ylabel("Validation AP")
    ax.set_title(f"{name} - Validation AP")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"[绘图] AP: {output}", flush=True)


# 对 F2 曲线抽稀，保证自动生成的图像清晰。
def f2_plot_data(y_true, y_prob, selected_threshold):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    p, r = precision[:-1], recall[:-1]
    beta_squared = F_BETA ** 2
    scores = np.divide(
        (1 + beta_squared) * p * r, beta_squared * p + r,
        out=np.zeros_like(p), where=(beta_squared * p + r) > 0,
    )
    best_index = int(np.searchsorted(thresholds, selected_threshold))
    step = max(1, len(thresholds) // 2000)
    shown = np.unique(np.r_[np.arange(0, len(thresholds), step),
                            best_index, len(thresholds) - 1])
    return {
        "thresholds": thresholds[shown].tolist(),
        "scores": scores[shown].tolist(),
        "selected_threshold": selected_threshold,
        "selected_f2": float(scores[best_index]),
    }


# 根据验证集曲线绘制 F2 图。
def plot_validation_f2(data, output, y_min=None, y_max=None):
    thresholds = data["thresholds"]
    scores = data["scores"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, scores, label="Validation F2")
    ax.scatter(data["selected_threshold"], data["selected_f2"],
               color="tab:orange", zorder=3,
               label=f"Selected: {data['selected_threshold']:.4f}")
    ax.set_xlim(0, 1)
    set_score_axis(ax, scores, y_min, y_max)
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Validation F2")
    ax.set_title("Validation F2 by threshold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"[绘图] F2: {output}", flush=True)


# 图和指标都按模型命名，避免不同实验互相覆盖。
def model_path(name):
    return MODEL_DIR / f"{name}.pkl"


def metrics_path(name):
    return MODEL_DIR / f"{name}-metrics.json"


def draw_charts(name, trials, curve, chosen, y_min=None, y_max=None):
    plot_validation_ap(
        name, trials, chosen,
        MODEL_DIR / f"validation-ap-{name}.png", y_min, y_max,
    )
    plot_validation_f2(
        curve, MODEL_DIR / f"validation-f2-{name}.png", y_min, y_max,
    )


def model_result(y_val, probability, rounds=None):
    result = {
        "positive_rate": float(y_val.mean()),
        "roc_auc": float(roc_auc_score(y_val, probability)),
        "pr_auc": float(average_precision_score(y_val, probability)),
        "features": FEATURE_NAMES,
    }
    if rounds is not None:
        result["n_estimators"] = rounds
    return result


# 逻辑回归仅训练一次，rounds 为最大迭代次数。
def train_logistic(X_train, y_train, X_val, y_val, rounds):
    print(f"[训练] 逻辑回归：最大迭代 {rounds} 次", flush=True)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=rounds,
                                             class_weight="balanced"))
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_val)[:, 1]
    result = model_result(y_val, probability)
    joblib.dump(model, model_path("logistic-regression"))
    print(f"[训练] 逻辑回归完成，验证集 AP={result['pr_auc']:.6f}", flush=True)
    return result, [result], probability


# 随机森林比较不超过 rounds 的候选树数，保留 AP 最好的模型。
def train_random_forest(X_train, y_train, X_val, y_val, rounds):
    counts = sorted(set([n for n in RF_COUNTS if n <= rounds] + [rounds]))
    best = None
    best_probability = None
    trials = []
    for count in counts:
        print(f"[训练] 随机森林：{count} 棵树", flush=True)
        model = RandomForestClassifier(
            n_estimators=count, random_state=SEED,
            class_weight="balanced", n_jobs=-1,
        )
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_val)[:, 1]
        result = model_result(y_val, probability, count)
        trials.append(result)
        print(f"[训练] {count} 棵树，验证集 AP={result['pr_auc']:.6f}", flush=True)
        if best is None or result["pr_auc"] > best["pr_auc"]:
            best = result
            best_probability = probability
            joblib.dump(model, model_path("random-forest"))
    print(f"[训练] 随机森林完成，最佳树数={best['n_estimators']}", flush=True)
    return best, trials, best_probability


# XGBoost 按验证集逐轮 AP 选轮数，再按选中轮数重新训练。
def train_xgboost(X_train, y_train, X_val, y_val, rounds):
    print(f"[训练] XGBoost：最多 {rounds} 轮，每 50 轮输出验证指标", flush=True)
    params = dict(max_depth=5, learning_rate=0.08,
                  subsample=0.9, colsample_bytree=0.9, random_state=SEED)
    search = XGBClassifier(n_estimators=rounds,
                           eval_metric=validation_ap, **params)
    search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    history = search.evals_result()["validation_0"]["validation_ap"]
    if len(history) != rounds or not np.isfinite(history).all():
        raise ValueError("XGBoost 逐轮验证指标缺失或无效")
    best_round = int(np.argmax(history)) + 1
    trials = [{"n_estimators": i, "pr_auc": float(ap)}
              for i, ap in enumerate(history, start=1)]
    print(f"[训练] 最高逐轮 AP={history[best_round-1]:.6f}，第 {best_round} 轮；重新训练", flush=True)
    del search
    model = XGBClassifier(n_estimators=best_round,
                          eval_metric="logloss", **params)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_val)[:, 1]
    result = model_result(y_val, probability, best_round)
    joblib.dump(model, model_path("xgboost"))
    print(f"[训练] XGBoost 完成，验证集 AP={result['pr_auc']:.6f}", flush=True)
    return result, trials, probability


TRAINERS = {
    "logistic-regression": train_logistic,
    "random-forest": train_random_forest,
    "xgboost": train_xgboost,
}


# 选择阶段只训练指定模型，阈值仅由验证集决定。
def main_select(name, rounds, y_min=None, y_max=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[select] 模型={name}，轮数上限={rounds}", flush=True)
    df, metadata = load_training_data()
    parts = split_by_time(df)
    print(f"[select] 训练集 {len(parts['train']):,} 行；验证集 {len(parts['validation']):,} 行", flush=True)
    metadata["splits"] = {
        key: {
            "rows": len(parts[key]),
            "risk_rows": int(parts[key]["is_laundering"].sum()),
            "start": parts[key]["transaction_timestamp"].min().isoformat(),
            "end": parts[key]["transaction_timestamp"].max().isoformat(),
        }
        for key in ("train", "validation")
    }
    X_train = build_features(parts["train"])[FEATURE_NAMES]
    y_train = parts["train"]["is_laundering"].astype(int)
    X_val = build_features(parts["validation"])[FEATURE_NAMES]
    y_val = parts["validation"]["is_laundering"].astype(int).to_numpy()
    print("[select] 特征构建完成", flush=True)

    chosen, trials, probability = TRAINERS[name](
        X_train, y_train, X_val, y_val, rounds,
    )
    threshold, candidate_count = select_threshold(y_val, probability)
    validation_selected = metrics(y_val, probability, threshold)
    print(f"[select] F2 阈值={threshold:.6f}；验证集 F2={validation_selected['f2']:.6f}", flush=True)

    # 模型专属文件用于独立测试；根目录 model.pkl 保持部署入口不变。
    with model_path(name).open("rb") as stream:
        model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    output = {
        "best_model": name,
        "model_sha256": model_sha256,
        "selection_metric": "model chosen by CLI; rounds selected by validation AP",
        "models": {name: chosen},
        "parameter_search": {name: trials},
        "training_config": {"model": name, "rounds": rounds},
        "threshold_selection": {
            "dataset": "validation",
            "metric": "f2",
            "candidate_rule": "unique validation predicted probabilities",
            "candidate_count": candidate_count,
            "tie_break": "highest threshold among equal maximum F2 scores",
            "selected_threshold": threshold,
            "validation_selected": validation_selected,
        },
        "data": metadata,
        "environment": {
            "python": platform.python_version(),
            "packages": {
                package: version(package) for package in
                ["scikit-learn", "xgboost", "numpy", "scipy", "joblib", "pandas", "pyarrow", "matplotlib"]
            },
        },
    }
    metrics_path(name).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    curve = f2_plot_data(y_val, probability, threshold)
    draw_charts(name, trials, curve, chosen, y_min, y_max)
    print("[select] 正在更新部署模型", flush=True)
    shutil.copyfile(model_path(name), MODEL_DIR / "model.pkl")
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"[select] 完成：{metrics_path(name)}", flush=True)
    print("[select] 测试集尚未评估；运行 test --model 后查看", flush=True)


# 测试阶段只读取对应模型的选择记录，不重新训练或调整阈值。
def main_test(name):
    result_path = metrics_path(name)
    trained_path = model_path(name)
    if not result_path.is_file() or not trained_path.is_file():
        raise FileNotFoundError(f"请先运行 select --model {name}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["best_model"] != name:
        raise ValueError("模型与选择记录不一致")
    if "test" in result:
        raise ValueError("测试集已评估；不要用测试结果反复调整参数")
    print(f"[test] 正在核对 {name} 模型文件", flush=True)
    with trained_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != result["model_sha256"]:
        raise ValueError("模型文件与验证集选择记录不一致，请重新运行 select")

    df, metadata = load_training_data()
    parts = split_by_time(df, check_test=True)
    previous = dict(result["data"])
    previous_splits = previous.pop("splits")
    if metadata != previous:
        raise ValueError("数据来源或筛选条件发生变化，请重新运行 select")
    for key in ("train", "validation"):
        part = parts[key]
        saved = previous_splits[key]
        if (len(part) != saved["rows"] or
                part["transaction_timestamp"].min().isoformat() != saved["start"] or
                part["transaction_timestamp"].max().isoformat() != saved["end"]):
            raise ValueError("数据划分发生变化，请重新运行 select")

    model = joblib.load(trained_path)
    threshold = result["threshold_selection"]["selected_threshold"]
    test = parts["test"]
    X_test = build_features(test)
    y_test = test["is_laundering"].astype(int).to_numpy()
    print(f"[test] 正在预测 {len(test):,} 行", flush=True)
    probability = model.predict_proba(X_test[FEATURE_NAMES])[:, 1]
    result["test"] = metrics(y_test, probability, threshold)
    result["data"]["splits"]["test"] = {
        "rows": len(test),
        "risk_rows": int(test["is_laundering"].sum()),
        "start": test["transaction_timestamp"].min().isoformat(),
        "end": test["transaction_timestamp"].max().isoformat(),
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    # 根目录指标仅在当前部署模型与本次测试模型相同时同步。
    root_path = MODEL_DIR / "metrics.json"
    if root_path.is_file():
        root = json.loads(root_path.read_text(encoding="utf-8"))
        if root.get("model_sha256") == digest and root.get("best_model") == name:
            root_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"[test] 完成：{result_path}", flush=True)
    print("[test] 指标:", result["test"], flush=True)


# 比较已保存的验证集 AP，不重新训练，也不使用测试集。
def main_compare(y_min=None, y_max=None):
    names = list(TRAINERS)
    results = []
    for name in names:
        file = metrics_path(name)
        if not file.is_file():
            raise FileNotFoundError(f"缺少 {file}；请先训练三个模型")
        results.append(json.loads(file.read_text(encoding="utf-8")))
    def comparable_data(result):
        data = dict(result["data"])
        splits = dict(data.pop("splits"))
        splits.pop("test", None)
        return data, splits
    if any(comparable_data(result) != comparable_data(results[0]) for result in results[1:]):
        raise ValueError("三个模型的数据来源或划分不同，不能直接比较")
    scores = [result["models"][name]["pr_auc"]
              for name, result in zip(names, results)]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(names))
    ax.scatter(xs, scores, s=90, color="tab:blue")
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_xticks(xs, names)
    set_score_axis(ax, scores, y_min, y_max)
    ax.set_ylabel("Validation AP")
    ax.set_title("Validation AP by model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = MODEL_DIR / "validation-ap-model-comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    for name, score in zip(names, scores):
        print(f"[比较] {name}: AP={score:.6f}", flush=True)
    print(f"[比较] 最佳验证集 AP：{names[int(np.argmax(scores))]}", flush=True)
    print(f"[比较] 图像：{output}", flush=True)


# 训练和测试均需指定模型，比较只读取保存的验证集指标。
def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("轮数必须为正整数")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description="按模型独立训练、测试和比较；训练后自动生成图像")
    stages = parser.add_subparsers(dest="stage", required=True)
    for stage in ("select", "test", "compare"):
        sub = stages.add_parser(stage)
        if stage != "compare":
            sub.add_argument("--model", required=True, choices=TRAINERS)
        if stage == "select":
            sub.add_argument("--rounds", type=positive_int,
                             help="逻辑回归最大迭代数、随机森林最大树数或 XGBoost 最大轮数")
        if stage != "test":
            sub.add_argument("--y-min", type=float, help="图的纵轴最小值；默认自动缩放")
            sub.add_argument("--y-max", type=float, help="图的纵轴最大值；默认自动缩放")
    args = parser.parse_args()
    if args.stage != "test":
        for bound in (args.y_min, args.y_max):
            if bound is not None and (not np.isfinite(bound) or not 0 <= bound <= 1):
                parser.error("纵轴边界必须是 0 到 1 之间的有限数")
        if args.y_min is not None and args.y_max is not None and args.y_min >= args.y_max:
            parser.error("纵轴最小值必须小于最大值")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "select":
        main_select(args.model, args.rounds or DEFAULT_ROUNDS[args.model],
                    args.y_min, args.y_max)
    elif args.stage == "test":
        main_test(args.model)
    else:
        main_compare(args.y_min, args.y_max)
