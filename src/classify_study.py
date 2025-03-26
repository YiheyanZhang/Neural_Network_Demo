"""
This is a program to classify rice seeds using a simple neural network
With the help from this video
https://www.bilibili.com/video/BV1BQR3YuEc7/?spm_id_from=333.1007.tianma.2-2-5.click&vd_source=79934a7c8eba67c8575bcf39607a4313
This program just use to study and practice neutal network
Editor: Yiheyan Zhang
"""
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 数据集预处理
data_df = pd.read_csv("dataset/riceClassification.csv")
data_df.dropna(inplace=True) # 删除缺失值
# axis=1 按列删除 axis=0 按行删除
data_df.drop(["id"], axis=1, inplace=True) # 删除id列

# 复制原始数据集
original_df = data_df.copy()

# 对每一列数据进行归一处理
for column in data_df.columns:
    data_df[column] = data_df[column] / data_df[column].max()

#### 数据集划分 0.7train 0.15 test 0.15 val ####
# 提取所有行和除了最后一列的所有列的数据，并转化为numpy数组
X = np.array(data_df.iloc[:, :-1])
# 提取所有行和最后一列的数据，并转化为numpy数组
Y = np.array(data_df.iloc[:, -1])

# 将数据集划分为训练集和测试集，其中测试集占30%，X为特征数据，Y为标签数据
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.3, 
    random_state=42
    )

# 将测试集划分为测试集和验证集，其中验证集占50%
X_test, X_val, y_test, y_val = train_test_split(
    X_test, y_test,
    test_size=0.5,
    random_state=42
    )

# print(X_train.shape, X_test.shape, X_val.shape)

# 定义数据集类
class dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device=device)
        self.y = torch.tensor(Y, dtype=torch.float32).to(device=device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建数据集
training_data = dataset(X=X_train, Y=y_train)
testing_data = dataset(X=X_test, Y=y_test)
validation_data = dataset(X=X_val, Y=y_val)

# 创建数据加载器
# shuffer=True表示每个epoch都会打乱数据
train_dataloader = DataLoader(training_data, batch_size = 32, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size = 32, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size = 32, shuffle=True)

# for x,y in train_dataloader:
#     print(x)
#     print("--------")
#     print(y)

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # shape[1]表示X的列数
        # 10 - 64 - 1 - Sigmoid
        self.input_layer = nn.Linear(X.shape[1], 64)
        self.linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
model = MyModel().to(device=device)

# 定义损失函数
# BCELoss是二分类交叉熵损失函数
# BCELoss的输入是一个概率值，所以在模型的最后一层需要加上一个Sigmoid函数
criterion = nn.BCELoss()
# 定义优化器
# Adam是一种自适应学习率的优化算法
# model.parameters()表示优化器需要优化的参数
# lr表示学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

epochs = 10
for epoch in range(epochs):
    total_acc_train = 0 # 训练集准确率
    total_loss_train = 0 # 训练集损失
    total_acc_validation = 0 # 验证集准确率
    total_loss_validation = 0 # 验证集损失

    model.train()
    for data in (train_dataloader):
        # 从训练集中取出数据
        inputs, labels = data
        # squeeze()函数用于压缩维度
        # 压缩前：(8, 1)，有额外的维度
        # 压缩后：(8,)
        prediction = model(inputs).squeeze()
        
        batch_loss = criterion(prediction, labels.squeeze())
        total_loss_train += batch_loss.item()

        acc = ((prediction).round() == labels).sum().item()
        total_acc_train += acc

        # 进行反向传播
        # 每次反向传播前需要清空梯度
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    # 验证集
    model.eval()
    with torch.no_grad():
        for data in validation_dataloader:
            inputs, labels = data

            prediction = model(inputs).squeeze()

            batch_loss = criterion(prediction, labels)
            total_loss_validation += batch_loss.item()

            acc = ((prediction).round() == labels).sum().item()
            total_acc_validation += acc
            
    total_loss_train_plot.append(total_loss_train / len(training_data))
    total_acc_train_plot.append(total_acc_train / len(training_data))
    total_loss_validation_plot.append(total_loss_validation / len(validation_data))
    total_acc_validation_plot.append(total_acc_validation / len(validation_data))

    print(f'Epoch{epoch + 1}')
    print("Train Loss: ",round(total_loss_train_plot[-1], 4))
    print("Train Acc: ", round(total_acc_train_plot[-1], 4))
    print("Validation Loss: ", round(total_loss_validation_plot[-1], 4))
    print("Validation Acc: ", round(total_acc_validation_plot[-1], 4))
    print("===========================")

# 保存模型
torch.save(model.state_dict(), "model/classify_study_model.pth")

def main():
    model = MyModel().to(device=device)
    model.load_state_dict(torch.load("model/classify_study_model.pth"))

    with torch.no_grad():
        model.eval()
        total_acc_test = 0
        for data in test_dataloader:
            inputs, labels = data
            prediction = model(inputs).squeeze()
            acc = ((prediction).round() == labels).sum().item()
            total_acc_test += acc
        test_acc = total_acc_test / len(testing_data)
        print("Test Acc: ", round(test_acc, 4))

    # 绘制图像
    fig,axs = plt.subplots(1,2, figsize=(25,10))

    axs[0].plot(total_loss_train_plot, label="Train Loss")
    axs[0].plot(total_loss_validation_plot, label="Validation Loss")
    axs[0].set_title("Train and Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_ylim(0, 1)
    axs[0].legend()

    axs[1].plot(total_acc_train_plot, label="Train Acc")
    axs[1].plot(total_acc_validation_plot, label="Validation Acc")
    axs[1].set_title("Train and Validation Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Acc")
    axs[1].set_ylim(0, 1)
    axs[1].legend()

    plt.show()

if __name__ == "__main__":
    main()