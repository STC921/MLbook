import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

X_train = np.arange(10, dtype='float32').reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')

# 画图部分
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 数据标准化
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
x_train_norm = torch.from_numpy(X_train_norm)  # 转为 PyTorch 张量
y_train = torch.from_numpy(y_train)  # 确保 y_train 也是 PyTorch 张量

# 创建 TensorDataset
train_ds = TensorDataset(x_train_norm, y_train)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# 设置随机种子，初始化权重和偏置
torch.manual_seed(1)
weight = torch.randn(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)


# 模型和损失函数
def model(xb):
    return xb @ weight + bias


def loss_fn(input, target):
    return (input - target).pow(2).mean()


# 训练超参数
learning_rate = 0.001
num_epochs = 200
log_epochs = 10

# 训练循环
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()

    # 更新权重和偏置
    with torch.no_grad():
        weight -= weight.grad * learning_rate
        bias -= bias.grad * learning_rate
        weight.grad.zero_()
        bias.grad.zero_()

    # 打印损失值
    if epoch % log_epochs == 0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')

path = 'iris_classifier.pt'
model_new = torch.load(path)
model_new.eval()
