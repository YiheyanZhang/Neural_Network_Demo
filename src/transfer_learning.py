"""
迁移学习
1. 加载数据集
2. 数据预处理
3. 创建数据加载器
4. 定义模型
5. 修改模型的最后一层
6. 定义损失函数和优化器
7. 训练模型
8. 评估模型
9. 模型保存

迁移学习的优势：
1. 减少训练时间
2. 提高模型性能
3. 适用于小数据集
4. 适用于新任务
5. 适用于新数据集
6. 适用于新领域
"""
import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

train_df = pd.read_csv("dataset/Bean Leaf Lesions Classification/train.csv")
val_df = pd.read_csv("dataset/Bean Leaf Lesions Classification/val.csv")

train_df["image:FILE"] = "dataset/Bean Leaf Lesions Classification/" + train_df["image:FILE"]
val_df["image:FILE"] = "dataset/Bean Leaf Lesions Classification/" + val_df["image:FILE"]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(self.dataframe["category"]).to(device)

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        img = Image.open(img_path)
        if(self.transform):
            img = (self.transform(img) / 255.0).to(device)
        return img, label
    
train_dataset = CustomImageDataset(train_df, transform)
val_dataset = CustomImageDataset(val_df, transform)

# 图像可视化
# n_rows = 3
# n_cols = 3
# f, axarr = plt.subplots(n_rows, n_cols, figsize=(16, 16))

# for row in range(n_rows):
#     for col in range(n_cols):
#         image = train_dataset[np.random.randint(0,train_dataset.__len__())][0].cpu()
#         # permute() 方法用于对换张量的维度
#         # Matplotlib的imshow期望的图像格式是高度×宽度×通道，即HWC格式，而PyTorch通常使用CHW格式
#         # squeeze() 方法用于移除所有长度为 1 的维度
#         axarr[row, col].imshow((image*255.0).squeeze().permute(1, 2, 0))
#         axarr[row, col].axis('off')

# plt.tight_layout()
# plt.show()

# 设置超参数
lr = 1e-4
batch_size = 16
epochs = 15

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 定义模型
googlenet_model = models.googlenet(pretrained=True)

for param in googlenet_model.parameters():
    # requires_grad属性默认为True，表示需要计算梯度
    param.requires_grad = True

# print(googlenet_model.fc)会输出模型的最后一层
# Linear(in_features=1024, out_features=1000, bias=True)
# print(googlenet_model.fc)

num_classes = len(train_df["category"].unique())
# print(num_classes)

# 修改模型的最后一层
googlenet_model.fc = nn.Linear(in_features=1024, out_features=num_classes)

# print(googlenet_model.fc)

googlenet_model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(googlenet_model.parameters(), lr=lr)

total_loss_train_plot = []
total_acc_train_plot = []
total_loss_val_plot = []
total_acc_val_plot = []

for epoch in range(epochs):
    total_loss_train = 0
    total_acc_train = 0

    googlenet_model.train()
    for inputs, labels in train_loader:
        # 1. 清空梯度
        optimizer.zero_grad()
        # 2. 前向传播
        outputs = googlenet_model(inputs)
        # 3. 计算损失
        train_loss = criterion(outputs, labels.long())
        total_loss_train += train_loss.item()
        # 4. 反向传播
        train_loss.backward()
        # 5. 更新参数
        optimizer.step()
        # 6. 计算准确率
        train_acc = (torch.argmax(outputs, dim=1) == labels).sum().item()
        total_acc_train += train_acc

    total_loss_train_plot.append(total_loss_train)
    total_acc_train_plot.append(total_acc_train / len(train_dataset))

    print("epoch", epoch + 1)
    print("train loss:", round(total_loss_train_plot[-1], 4))
    print("train acc:", round(total_acc_train_plot[-1], 4))
    print("-------------------------------------")

    total_loss_val = 0
    total_acc_val = 0

    googlenet_model.eval()
    # no_grad() 函数用于停止梯度的计算
    with torch.no_grad():
        for inputs, labels in val_loader:
            # 1. 前向传播
            outputs = googlenet_model(inputs)
            # 2. 计算损失
            val_loss = criterion(outputs, labels.long())
            total_loss_val += val_loss.item()
            # 3. 计算准确率
            val_acc = (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_acc_val += val_acc
    
    total_loss_val_plot.append(total_loss_val)
    total_acc_val_plot.append(total_acc_val / len(val_dataset))

    print("val loss:", round(total_loss_val_plot[-1], 4))
    print("val acc:", round(total_acc_val_plot[-1], 4))
    print("=====================================")

# 评估模型
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].plot(total_loss_train_plot, label="train loss")
axes[0].plot(total_loss_val_plot, label="val loss")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].set_xlabel("epochs")
axes[0].set_ylabel("loss")
axes[0].grid()

axes[1].plot(total_acc_train_plot, label="train acc")
axes[1].plot(total_acc_val_plot, label="val acc")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].set_xlabel("epochs")
axes[1].set_ylabel("accuracy")
axes[1].grid()

plt.tight_layout()
plt.show()

# 保存模型
torch.save(googlenet_model.state_dict(), "model/transfer_learning_model.pth")


