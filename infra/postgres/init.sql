-- ============================================================
-- fincross-risk 数据库初始化脚本
-- ============================================================

-- 1. 交易主表：保存 AML-Data 导入记录和在线提交记录。
-- Data Service 先写入这里，Worker 完成评分后再更新 status。
CREATE TABLE IF NOT EXISTS transactions (
    -- 数据库内部主键。AML-Data 本身没有统一交易 ID。
    id BIGSERIAL PRIMARY KEY,
    -- 外部业务编号，唯一约束防止同一交易重复提交。
    external_id VARCHAR(160) UNIQUE NOT NULL,

    -- 数据来源和原始文件定位，用于追溯、排错和复现实验。
    source_type VARCHAR(32) NOT NULL DEFAULT 'ONLINE'
        CHECK (source_type IN ('AML_DATA', 'ONLINE', 'AMLSIM')),
    source_file TEXT,
    source_row_number BIGINT CHECK (source_row_number IS NULL OR source_row_number > 0),

    -- 交易发生时间，带时区，便于跨地区数据统一比较。
    transaction_timestamp TIMESTAMPTZ NOT NULL,
    -- AML-Data 的 From Bank / 第一个 Account：转出方。
    from_bank VARCHAR(32) NOT NULL,
    from_account VARCHAR(64) NOT NULL,
    -- AML-Data 的 To Bank / 第二个 Account：收款方。
    to_bank VARCHAR(32) NOT NULL,
    to_account VARCHAR(64) NOT NULL,

    -- 收款方实际收到的金额。
    amount_received NUMERIC(20, 2) NOT NULL CHECK (amount_received >= 0),
    -- 收款方收到的币种。
    receiving_currency VARCHAR(32) NOT NULL,
    -- 转出方支付的金额。
    amount_paid NUMERIC(20, 2) NOT NULL CHECK (amount_paid >= 0),
    -- 转出方支付的币种。
    payment_currency VARCHAR(32) NOT NULL,
    -- 支付方式，例如 ACH、Wire、Credit Card。
    payment_format VARCHAR(32) NOT NULL,

    -- Is Laundering 是离线训练标签；在线新交易没有真实标签，因此允许 NULL。
    is_laundering BOOLEAN,
    -- 原始行快照，便于核对映射；不替代 data/source 中的原始 CSV。
    raw_record JSONB,

    -- 异步处理状态：接收、排队、完成评分或失败。
    status VARCHAR(32) NOT NULL DEFAULT 'RECEIVED'
        CHECK (status IN ('RECEIVED', 'QUEUED', 'SCORED', 'FAILED')),
    -- 交易入库时间，与 transaction_timestamp 区分。
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- 最近一次状态变更时间，便于排查异步处理延迟。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. 模型预测结果表：一笔交易可以对应多条预测结果，支持模型版本对比。
CREATE TABLE IF NOT EXISTS predictions (
    -- 预测结果数据库内部主键。
    id BIGSERIAL PRIMARY KEY,
    -- 外键：每条预测必须对应一笔已经存在的交易。
    transaction_id BIGINT NOT NULL
        REFERENCES transactions(id) ON DELETE CASCADE,
    -- 产生这条结果的模型版本。
    model_version VARCHAR(128) NOT NULL,
    -- 模型输出的风险概率，范围固定为 0 到 1。
    risk_probability NUMERIC(10, 8) NOT NULL
        CHECK (risk_probability >= 0 AND risk_probability <= 1),
    -- 根据概率转换出的展示等级。
    risk_level VARCHAR(32) NOT NULL
        CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    -- 预测生成时间。
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- 同一交易同一模型版本只保留一条结果，避免重复消费造成重复记录。
    CONSTRAINT uq_predictions_transaction_model UNIQUE (transaction_id, model_version)
);

-- 3. 实验记录表：保存一次训练评估。
CREATE TABLE IF NOT EXISTS experiment_runs (
    -- 实验记录数据库内部主键。
    id BIGSERIAL PRIMARY KEY,
    -- 模型名称，例如 logistic-regression、random-forest、xgboost。
    model_name VARCHAR(128) NOT NULL,
    -- 模型版本，与 .env 和 predictions 中的版本标识保持一致。
    model_version VARCHAR(128) NOT NULL,
    -- 数据集、特征版本、随机种子等可复现信息。
    dataset_reference TEXT,
    feature_version VARCHAR(64),
    random_seed INTEGER,
    -- 保存 Precision、Recall、F1、ROC-AUC、PR-AUC 等指标。
    metrics JSONB NOT NULL,
    -- 实验记录写入时间。
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 索引：覆盖最常用的查询路径。
-- Data Service / Worker 经常按业务编号查询交易。
CREATE INDEX IF NOT EXISTS idx_transactions_external_id
    ON transactions(external_id);
-- 按状态统计积压任务或查询待处理交易。
CREATE INDEX IF NOT EXISTS idx_transactions_status_created_at
    ON transactions(status, created_at);
-- 按交易编号查询预测结果。
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id
    ON predictions(transaction_id);
-- 按模型版本比较实验结果。
CREATE INDEX IF NOT EXISTS idx_experiment_runs_model_version
    ON experiment_runs(model_version);