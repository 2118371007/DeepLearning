import torch

# 定义初始权重w
w = torch.tensor(10, requires_grad = True, dtype = torch.float)

# 定义损失函数
loss = w ** 2 + 20
print(f"权重初始值={w},(0.01*w.grad)=空,loss={loss}")

# 循环一百次更新权重
for i in range(1, 101):

    # 正向计算(前向传播)
    loss = w ** 2 + 20

    # 默认梯度会累计，当第一次没有求导(反向传播)的时候不需要梯度清零,后面每一次都需要梯度清零
    if w.grad is not None:
        w.grad.zero_()

    # 反向传播(求导)
    loss.sum().backward()

    # 更新权重
    w.data = w.data - 0.01 * w.grad

    # print(w.grad)
    print(f"第{i}次，权重初始值={w},(0.01 * w.grad)={0.01 * w.grad:.5f},loss={loss:.5f}")
