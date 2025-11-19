import warnings, json, joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# 基础配置
warnings.filterwarnings("ignore")
RANDOM_STATE = 168 # 固定随机种子，保证复现性 same as ANN
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# 1) 读取原始数据
df = pd.read_csv("pipe_data.csv")

# 特征 X：入口速度 + 阀角度 / 目标 y：主管 + 支管质量流量
X = df[["inlet_velocity", "degree"]].values
y = df[["main_mass_flow", "branch_mass_flow"]].values

# 2) 划分训练 / 验证 / 测试集
# 先留 20% 做最终测试集，20% 做验证集 → 60/20/20
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=RANDOM_STATE) # 0.25 * 0.8 = 0.20

# 3) 标准化 (μ=0, σ=1)
scaler = StandardScaler().fit(X_train) # 仅用训练集拟合
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "cnn_scaler.pkl") # 保存归一化器，部署时保持一致

#  4) 调整为 CNN 期望形状
# Conv1D 期望输入: (batch, timesteps, channels)
# 这里 timesteps = 2 (两个特征), channels = 1
X_train = X_train.reshape(-1, 2, 1)
X_val = X_val.reshape( -1, 2, 1)
X_test = X_test.reshape( -1, 2, 1)

#  5) 构建 CNN 模型
# kernel_size 为 1 => Point‑wise 卷积，相当于在“局部”提取非线性组合
model = models.Sequential([
    layers.Input(shape=(2, 1)),
    layers.Conv1D(32, kernel_size=1, activation="relu"), # 输出通道 32
    layers.Conv1D(64, kernel_size=1, activation="relu"), # 输出通道 64
    layers.Flatten(), # 展平成向量
    layers.Dense(64, activation="relu"),
    layers.Dense(2) # 回归两路质量流量
])

model.compile(optimizer="adam", loss="mse")

# 6) 训练模型
early = callbacks.EarlyStopping(
    monitor="val_loss", patience=30, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs= 500, # 先跑500轮
    batch_size=32,
    callbacks=[early],
    verbose=0 # 设为 1 可实时打印 loss 曲线
)

# 7) 在验证 / 测试集评估
def get_metrics(y_true, y_pred):
    """返回 (MAPE, R²) 二元组，越小 / 越接近 1 越好"""
    return (mean_absolute_percentage_error(y_true, y_pred),
            r2_score(y_true, y_pred))

y_val_pred = model.predict(X_val, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

mape_val, r2_val = get_metrics(y_val, y_val_pred)
mape_test, r2_test = get_metrics(y_test, y_test_pred)

# 8) 保存模型与结果
model.save("cnn_model.h5")

# 8‑2 测试集真值 / 预测 / 误差 CSV
pred_df = pd.DataFrame(
    np.hstack([scaler.inverse_transform(X_test.reshape(-1, 2)), # 还原特征原尺度
               y_test, y_test_pred]),
    columns=["inlet_velocity", "degree",
             "main_flow_true", "branch_flow_true",
             "main_flow_pred", "branch_flow_pred"])

# 计算两路 MRE
pred_df["main_mre"] = (pred_df["main_flow_pred"] - pred_df["main_flow_true"]).abs() / pred_df["main_flow_true"]
pred_df["branch_mre"] = (pred_df["branch_flow_pred"] - pred_df["branch_flow_true"]).abs() / pred_df["branch_flow_true"]

pred_df.to_csv("cnn_predictions.csv", index=False)

# 8‑3 json
with open("cnn_model_info.txt", "w", encoding="utf-8") as fp:
    json.dump({
        "architecture": "Conv1D(32,1)->Conv1D(64,1)->Flatten->Dense64->Dense2",
        "scaler": "StandardScaler(mean=0, std=1)",
        "validation": {"mape": float(mape_val), "r2": float(r2_val)},
        "test": {"mape": float(mape_test), "r2": float(r2_test)},
        "early_stopping_epoch": len(history.history["loss"])
    }, fp, indent=2, ensure_ascii=False)

# 8‑4 真值‑预测散点 ＋ 误差直方图（same as ANN)
with PdfPages("cnn_results.pdf") as pdf:
    # 散点图：主管
    plt.figure()
    plt.scatter(y_test[:, 0], y_test_pred[:, 0], s=12)
    lim = [min(y_test[:, 0].min(), y_test_pred[:, 0].min()),
           max(y_test[:, 0].max(), y_test_pred[:, 0].max())]
    plt.plot(lim, lim, "k--")
    plt.title("Main Flow • Truth vs Prediction")
    plt.xlabel("True"); plt.ylabel("Pred")
    plt.tight_layout(); pdf.savefig(); plt.close()

    # 散点图：支管
    plt.figure()
    plt.scatter(y_test[:, 1], y_test_pred[:, 1], s=12)
    lim = [min(y_test[:, 1].min(), y_test_pred[:, 1].min()),
           max(y_test[:, 1].max(), y_test_pred[:, 1].max())]
    plt.plot(lim, lim, "k--")
    plt.title("Branch Flow • Truth vs Prediction")
    plt.xlabel("True"); plt.ylabel("Pred")
    plt.tight_layout(); pdf.savefig(); plt.close()

    # 相对误差分布
    plt.figure()
    plt.hist(pred_df["main_mre"], bins=30, alpha=0.7, label="Main")
    plt.hist(pred_df["branch_mre"], bins=30, alpha=0.7, label="Branch")
    plt.title("Relative Error Distribution (CNN)")
    plt.xlabel("MRE"); plt.ylabel("Count"); plt.legend()
    plt.tight_layout(); pdf.savefig(); plt.close()

print("训练完成")


