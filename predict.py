import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
font_path = "/data1/xyj/SimSun.ttf"
if os.path.exists(font_path):
        my_font = fm.FontProperties(fname=font_path, size=12)
else:
    print(f"字体文件 {font_path} 不存在，将使用默认字体")
    my_font = None
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体显示中文
# plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

torch.manual_seed(42)
np.random.seed(42)

from model import LSTMModel,TimeSeriesTransformer,InceptionTransformer,TransformerWithLearnedPE
from model import MultiScaleConvAdaptiveTransformer,GraphEnhancedTemporalModel,DynamicAttentionExternalModel


### -------- Step 1: 数据读取与预处理 -------- ###
def load_and_preprocess(filepath,output_file):
    df = pd.read_csv(filepath, sep=",")
    df['DateTime'] = pd.to_datetime(df['DateTime'], format="%Y/%m/%d %H:%M")
    df = df.set_index("DateTime")

    # 替换缺失值并转为float
    df = df.replace("?", np.nan).astype(float)
    df = df.fillna(method='ffill').fillna(method='bfill')

    # 计算剩余能耗
    df["sub_metering_remainder"] = (df["Global_active_power"] * 1000 / 60) - (
            df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"]
    )

    # 每天聚合
    daily_df = pd.DataFrame()
    daily_df["Global_active_power"] = df["Global_active_power"].resample('D').sum()
    daily_df["Global_reactive_power"] = df["Global_reactive_power"].resample('D').sum()
    daily_df["Voltage"] = df["Voltage"].resample('D').mean()
    daily_df["Global_intensity"] = df["Global_intensity"].resample('D').mean()
    daily_df["Sub_metering_1"] = df["Sub_metering_1"].resample('D').sum()
    daily_df["Sub_metering_2"] = df["Sub_metering_2"].resample('D').sum()
    daily_df["Sub_metering_3"] = df["Sub_metering_3"].resample('D').sum()
    daily_df["sub_metering_remainder"] = df["sub_metering_remainder"].resample('D').sum()
    daily_df["RR"] = df["RR"].resample("D").first()
    daily_df["NBJRR1"] = df["NBJRR1"].resample("D").first()
    daily_df["NBJRR5"] = df["NBJRR5"].resample("D").first()
    daily_df["NBJRR10"] = df["NBJRR10"].resample("D").first()
    daily_df["NBJBROU"] = df["NBJBROU"].resample("D").first()
    # # 如果提供了输出路径，则保存到文件
    # if output_file is not None:
    #     daily_df.to_csv(output_file, index=True)
    #     print(f"已保存预处理后的数据到: {output_file}")
    return daily_df


### -------- Step 2: 构建 Dataset -------- ###
class PowerDataset(Dataset):
    def __init__(self, data, input_len, output_len):
        self.X, self.y = [], []
        for i in range(len(data) - input_len - output_len):
            self.X.append(data[i:i + input_len])
            self.y.append(data[i + input_len:i + input_len + output_len, 0])  # 只预测 Global_active_power
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


### -------- Step 4（修改）: 模型训练与预测 -------- ###
def train_and_evaluate(train_loader, test_loader, input_dim, output_len, target_scaler, epochs=50, lr=0.001):
    # model = LSTMModel(input_dim=input_dim, output_len=output_len)
    # model = TimeSeriesTransformer(input_dim=input_dim, output_len=output_len)
    # model = InceptionTransformer(input_dim=input_dim, output_len=output_len)
    # model = TransformerWithLearnedPE(input_dim=input_dim, output_len=output_len)
    model = DynamicAttentionExternalModel(input_dim=input_dim, output_len=output_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy()
            all_preds.append(pred)
            all_trues.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)

    # 反归一化
    preds_denorm = target_scaler.inverse_transform(preds)
    trues_denorm = target_scaler.inverse_transform(trues)

    return preds_denorm, trues_denorm


### -------- Step 5（修改）: 主流程 -------- ###
def run_pipeline(train_path, test_path, input_len=90, output_len=90, runs=5, draw=True, model_name="OPS"):
    train_df = load_and_preprocess(train_path, "train_processed.csv")
    test_df = load_and_preprocess(test_path, "test_processed.csv")

    full_df = pd.concat([train_df, test_df])

    #  1. 建立 scaler
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaled = feature_scaler.fit_transform(full_df.values)
    target_scaler.fit(train_df[["Global_active_power"]])  # 只拟合训练集的目标值

    #  2. 拆分数据
    train_scaled = feature_scaled[:len(train_df)]
    test_scaled = feature_scaled[len(train_df) - input_len:]  # 保证前90天输入

    train_dataset = PowerDataset(train_scaled, input_len, output_len)
    test_dataset = PowerDataset(test_scaled, input_len, output_len)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    mse_list, mae_list = [], []
    all_preds = []

    for run in tqdm(range(runs), desc="多轮实验"):
        preds, trues = train_and_evaluate(train_loader, test_loader,
                                          input_dim=feature_scaled.shape[1],
                                          output_len=output_len,
                                          target_scaler=target_scaler)
        all_preds.append(preds)
        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        mse_list.append(mse)
        mae_list.append(mae)
        print(f"Run {run + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    avg_preds = np.mean(np.array(all_preds), axis=0)
    std_preds = np.std(np.array(all_preds), axis=0)

    print(f"\n平均 MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
    print(f"平均 MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")

    if draw:
        plot_predictions(avg_preds, trues, std_preds, output_len, model_name)


### -------- Step 6: 可视化 -------- ###
def plot_predictions(avg_preds, trues, std_preds, output_len,model_name):
    days = np.arange(output_len)

    plt.figure(figsize=(12, 6))
    plt.plot(days, trues[0], label='实际值', color='black')
    plt.plot(days, avg_preds[0], label='预测平均值', color='blue')
    plt.fill_between(days,
                     avg_preds[0] - std_preds[0],
                     avg_preds[0] + std_preds[0],
                     color='blue', alpha=0.2, label='±1 std')

    # plt.xlabel("预测天数")
    # plt.ylabel("Global Active Power (kW)")
    # plt.title("家庭每日总有功功率预测")
    # plt.legend()

    plt.xlabel("预测天数", fontproperties=my_font)
    plt.ylabel("Global Active Power (kW)", fontproperties=my_font)
    plt.title("家庭每日总有功功率预测", fontproperties=my_font)
    plt.legend(prop=my_font)


    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"my_prediction{output_len}_{model_name}.png")
    print()

### -------- Step 7: 启动 -------- ###
if __name__ == "__main__":
    train_file = "train.csv"  # 替换为实际路径
    test_file = "test.csv"
    model_name="DynamicAttentionExternalModel"
    print("短期预测（90 → 90 天）")
    run_pipeline(train_file, test_file, input_len=90, output_len=90,model_name=model_name)

    print("\n长期预测（90 → 365 天）")
    run_pipeline(train_file, test_file, input_len=90, output_len=365,model_name=model_name)
