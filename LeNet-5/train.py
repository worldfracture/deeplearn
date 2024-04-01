import torch
from torch import nn
from torchvision.datasets import ImageFolder

from net import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

# 数据预处理 转为tensor 以及 标准化:
transform = transforms.Compose([
     #转为灰度图像:
     transforms.Grayscale(num_output_channels=1),
     #将图片转换为Tensor,归一化至(0,1):
     transforms.ToTensor(),
     #T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

BATCH_SIZE = 100
#数据集要作为一整个文件夹读入：

#构造训练集:
train_data = ImageFolder(root="D:\deeplearn\data\emnist\Train_png", transform=transform)
#shuffle代表是否在构建批次时随机选取数据:
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=BATCH_SIZE, shuffle=True)

#构造数据集:
test_data = ImageFolder(root="D:\deeplearn\data\emnist\Test_png", transform=transform)
#之所以要将test_data转换为loader是因为网络不支持原始的ImageFolder类数据，到时候直接使用批训练，便是tensor类。因此batch_size为全部testdata(test_data.__len__())
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size=test_data.__len__())


# 如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用net里面定义的模型，将模型数据转到GPU
model = MyLeNet5().to(device)

# 定义一个损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

# 定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 学习率每隔10轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 前向转播
        X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)

        cur_acc = torch.sum(y == pred)/output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    print("train_loss" + str(loss/n))
    print("train_acc" + str(current/n))

def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # 前向转播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("val_loss" + str(loss / n))
        print("val_acc" + str(current / n))

        return current/n

# 开始训练
epoch = 2
min_acc = 0
for t in range(epoch):
    print(f'epoch{t+1}\n------------------')
    train(train_loader, model, loss_fn, optimizer)
    a = val(test_loader, model, loss_fn)
    # 保存最好的模型权重
    if a > min_acc:
        folder = 'sava_model'
        if not os.path.exists(folder):
            os.mkdir('sava_model')
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(), 'LeNet-5/sava_model/best_model.pth')
print('Done!')