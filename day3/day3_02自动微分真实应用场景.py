"""

1.先前向传播(正向传播)计算出预测值z
2.基于损失函数，结合预测值z和真实值y来计算梯度
3.结合权重更新公式 w新 = w旧 - 学习率 * 梯度 来更新权重
"""

import torch

# 1.定义x，表示特征(输入数据)，假设是一个三行四列的全1矩阵
x = torch.ones(3, 4)
print(f"x:{x.data}")

# 2.定义y，表示标签(真实值)，假设是一个三行五列的全0矩阵
y = torch.zeros(3, 5)

# 3.初始化可自动微分的权重和初始值(根据特征和真实值的形状来设置权重和偏置值的形状)
# 因为线性回归公式为 wx + b 所以wx的形状要能满足矩阵乘法的要求，乘起来以后每一列都要加上一个b，所以b的形状要和w的列数一致
w = torch.randn(4, 5, requires_grad = True)
b = torch.randn(5, requires_grad = True)

# 4.前向传播(正向传播)
z = x @ w + b
print(f"z:{z.data}")

# 5.定义损失函数
criterion = torch.nn.MSELoss()  # nn:neural network 神经网络
loss = criterion(z, y)  # loss = 损失

# 6.进行自动微分
loss.sum().backward()

# 7.打印w和b的梯度
print(f"w.grad:{w.grad}\nb.grad={b.grad}")

# 后续就是w新 = w旧 - 学习率 * 梯度 来更新权重
