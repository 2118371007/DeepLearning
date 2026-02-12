"""

一个张量一旦设置了自动微分，这个张量就不能直接转为numpy的nd数组，需要通过detach()函数解决
"""
import torch
import numpy as np

t1 = torch.tensor([1, 2, 3], requires_grad = True, dtype = torch.float)

# 通过detach()拷贝一份张量(共享同一块内存空间，但是拷贝的张量requires_grad = False，可以转换为nd数组)
t2 = t1.detach()
t1.data[0] = 100
print(f"t1:{t1.data}\nt2:{t2.data}")
print("*" * 40)
n1 = t1.detach().numpy()

print(f"t1:{t1.data},t1type:{type(t1)}\nn1:{n1},n1type:{type(n1)}")
