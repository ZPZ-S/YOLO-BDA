import torch
import torch.nn as nn
import torch.optim as optim

# 简单的模型
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 112 * 112, 10)  # 假设输入图像大小为 224x224
)

# 随机生成数据
inputs = torch.randn(32, 3, 224, 224).cuda()  # 32个样本，3通道224x224图像
labels = torch.randint(0, 10, (32,)).cuda()

model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 测试一次前向传播
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")

