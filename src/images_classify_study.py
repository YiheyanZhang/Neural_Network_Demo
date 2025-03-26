import torch
from torch import nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import os
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

image_path = []
labels = []

# 遍历数据集文件夹，获取图片路径和标签
for i in os.listdir("dataset/afhq"):
    # print(i)
    for label in os.listdir(f"dataset/afhq/{i}"):
        # print(label)
        for image in os.listdir(f"dataset/afhq/{i}/{label}"):
            # print(image)
            image_path.append(f"dataset/afhq/{i}/{label}/{image}")
            labels.append(label)

# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
# DataFrame 是 pandas 库中的一种数据结构，类似于表格，可以存储多种类型的数据
data_df = pd.DataFrame(zip(image_path, labels), columns=["image_path", "label"])
# print(data_df.head())

# 分割数据集
# 70% 训练集，15% 测试集，15% 验证集
train = data_df.sample(frac=0.7, random_state=42)
# drop() 方法用于删除行或列，删除训练集中的数据，剩下的就是测试集
test = data_df.drop(train.index)
val = test.sample(frac=0.5, random_state=42)
test = test.drop(val.index)

# LabelEncoder 是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码
label_encoder = LabelEncoder()
# fit() 方法会计算出每个类别对应的编码
label_encoder.fit(data_df["label"])

# transforms.Compose() 用于将多个数据预处理操作合并为一个操作
transform = transforms.Compose([
    # Resize() 用于调整图片大小
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])

# 定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.Tensor(label_encoder.transform(dataframe["label"])).to(device=device)
    
    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        
        if(self.transform):
            image = self.transform(image).to(device=device)

        return image, label

# 创建数据集
train_dataset = CustomImageDataset(train, transform)
val_dataset = CustomImageDataset(val, transform)
test_dataset = CustomImageDataset(test, transform)

# print(train_dataset.__getitem__(0))

# 显示类别对应的标签
# print(label_encoder.inverse_transform([0,1,2]))

# 图像显示部分
# n_rows = 3
# n_cols = 3

# fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
# for row in range(n_rows):
#     for col in range(n_cols):
#         # open()用于打开一个文件，返回一个file对象
#         # sample(n = 1)["image_path"]表示从数据集中随机抽取一张图片
#         # iloc[] 用于通过行号获取行数据
#         image = Image.open(data_df.sample(n = 1)["image_path"].iloc[0]).convert("RGB")
#         axs[row, col].imshow(image)
#         axs[row, col].axis("off")

# plt.show()

# 设置超参数
lr = 1e-4
batch_size = 64
epochs = 10

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 创建模型
class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # 池化层
        self.pooling = nn.MaxPool2d(2, 2)
        # 激活函数
        self.relu = nn.ReLU()
        # 扁平化层
        self.flatten = nn.Flatten()
        # 全连接层
        self.linear = nn.Linear(128*16*16, 128)
        # 输出层
        self.output = nn.Linear(128, len(data_df['label'].unique()))

    def forward(self, x):
        # 输出尺寸 = (输入尺寸 + 2 * padding - 卷积核大小) / 步长 + 1
        x = self.conv1(x) # 3*128*128 -> 32*128*128
        # 输出尺寸 = (输入尺寸 - 池化核大小) / 步长 + 1
        x = self.pooling(x) # 32*128*128 -> 32*64*64
        x = self.relu(x)

        x = self.conv2(x) # 32*64*64 -> 64*64*64
        x = self.pooling(x) # 64*64*64 -> 64*32*32
        x = self.relu(x)

        x = self.conv3(x) # 64*32*32 -> 128*32*32
        x = self.pooling(x) # 128*32*32 -> 128*16*16
        x = self.relu(x)

        x = self.flatten(x) # 128*16*16 -> 32768
        x = self.linear(x) # 32768 -> 128
        x = self.relu(x)

        x = self.output(x) # 128 -> 3
        return x

model = CNN_Net().to(device=device)
# summary() 用于打印模型的结构
# print(summary(model, input_size=(3, 128, 128)))

# 定义损失函数
# CrossEntropyLoss() 用于多分类问题，计算交叉熵损失
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = Adam(model.parameters(), lr=lr)

# 存储训练过程中的损失和准确率
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []

for epoch in range(epochs):
    total_acc_train = 0
    total_loss_train = 0
    total_acc_validation = 0
    total_loss_validation = 0

    now_progress = 0
    total_progress = len(train_dataloader)

    model.train()
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        # 计算损失
        train_loss = criterion(outputs, labels.long())
        total_loss_train += train_loss.item()
        # 反向传播
        train_loss.backward()
        optimizer.step()
        # 计算准确率
        train_acc = (torch.argmax(outputs, dim = 1) == labels).sum().item()
        total_acc_train += train_acc

        now_progress += 1
        if(now_progress % 100 == 0):
            print("epoch", epoch + 1,"train progress: ",round(now_progress / total_progress * 100, 2), "%")

    total_acc_train = total_acc_train / len(train_dataset)
    total_acc_train_plot.append(total_acc_train)

    total_loss_train_plot.append(total_loss_train)

    now_progress = 0
    total_progress = len(val_dataloader)
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            outputs = model(inputs)
            # 计算损失
            val_loss = criterion(outputs, labels.long())
            total_loss_validation += val_loss.item()
            # 计算准确率
            val_acc = (torch.argmax(outputs, dim = 1) == labels).sum().item()
            total_acc_validation += val_acc

            now_progress += 1
            if(now_progress % 10 == 0):
                print("epoch", epoch + 1,"validation progress: ",round(now_progress / total_progress * 100, 2), "%")

    total_acc_validation = total_acc_validation / len(val_dataset)
    total_acc_validation_plot.append(total_acc_validation)

    total_loss_validation_plot.append(total_loss_validation)

    print("----------")
    print("epoch", epoch + 1)
    print("train_loss : ", round(total_loss_train_plot[-1], 4))
    print("train_acc : ", round(total_acc_train_plot[-1], 4))
    print("validation_loss : ", round(total_loss_validation_plot[-1], 4))
    print("validation_acc : ", round(total_acc_validation_plot[-1], 4))
    print("==========")

# 保存模型
torch.save(model.state_dict(), "model/images_classify_study_model.pth")

def main():
    model = CNN_Net().to(device=device)
    model.load_state_dict(torch.load("model/images_classify_study_model.pth"))

    with torch.no_grad():
        total_loss_test = 0
        total_acc_test = 0
        for inputs, labels in test_dataloader:
            predictions = model(inputs)
            test_loss = criterion(predictions, labels.long())
            test_acc = (torch.argmax(predictions, dim = 1) == labels).sum().item()

            total_loss_test += test_loss.item()
            total_acc_test += test_acc
        
        print("test_loss:", round(total_loss_test, 4))
        print("test_acc:", round(total_acc_test / len(test_dataset), 4))
        print("!!!!!!!!!!")

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    axs[0].plot(total_loss_train_plot, label="train_loss")
    axs[0].plot(total_loss_validation_plot, label="validation_loss")
    axs[0].legend()
    axs[0].set_title("train and validation loss")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")

    axs[1].plot(total_acc_train_plot, label="train_acc")
    axs[1].plot(total_acc_validation_plot, label="validation_acc")
    axs[1].legend()
    axs[1].set_title("train and validation acc")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("acc")
    axs[1].set_ylim(0, 1)

    fig.show()


if __name__ == "__main__":
    main()
