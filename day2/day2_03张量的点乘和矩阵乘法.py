"""
点乘:
    要求两个张量维度一致，对应位置元素直接做相应的运算(+-*/都可以)
    t1 * t2
    t1.mul(t2)

矩阵乘法:
    和线性代数中一样
    t1 @ t2
    t1.matmul(t2)
"""
import torch

# 点乘，对应位置直接做相应的运算
def demo1():
    t1=torch.tensor([[1,2],[3,4]])
    t2=torch.tensor([[5,6],[7,8]])
    t3=t1*t2
    print(f"t1:{t1}\nt2:{t2}\nt3:{t3}")
    print("*"*40)

# 矩阵乘法
def demo2():
    t1=torch.tensor([[1,2],[3,4]])
    t2=torch.tensor([[5,6],[7,8]])
    t3=t1@t2
    print(f"t1:{t1}\nt2:{t2}\nt3:{t3}")
    print("*"*40)

if __name__ == '__main__':
    demo1()
    demo2()