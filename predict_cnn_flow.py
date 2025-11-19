import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# 载入模型
scaler = joblib.load("cnn_scaler.pkl")
model = load_model("cnn_model.h5", compile=False)

# 准备输入（列名必须与训练时一致）
# 底下数值随便换
X_new = pd.DataFrame({
    "inlet_velocity": [9.9], #入口风速
    "degree" : [30.0] #阀门开度
})

# 标准化 → reshape → 预测
X_scaled = scaler.transform(X_new)
X_scaled = X_scaled.reshape(-1, 2, 1)

main_flow, branch_flow = model.predict(X_scaled, verbose=0)[0]

print(f"主管质量流量: {main_flow:.3f}")
print(f"支管质量流量: {branch_flow:.3f}")



