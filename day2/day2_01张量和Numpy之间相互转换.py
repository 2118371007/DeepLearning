"""
掌握:
张量->numpy:  张量对象.numpy()
numpy->张量:  torch.tensor(nd数组)
从标量中提取内容: 标量.item()
"""


import torch
import numpy as np

# 张量转nd数组
def demo1():
    # 创建张量
    t1=torch.tensor([1,2,3])
    print(f"t1:{t1},type:{type(t1)}")

    # 把t1转变为numpy数组(nd数组),共享内存(张量和nd数组共享同一块内存空间)
    n1=t1.numpy()
    print(f"n1:{n1},type:{type(n1)}")
    print("*"*40)


    n1[0]=100#修改了n1，n1和t1都会改变
    print(f"t1:{t1},type:{type(t1)}")

    print(f"n1:{n1},type:{type(n1)}")
    print("*"*40)

    t2 = torch.tensor([1, 2, 3])
    print(f"t2:{t2},type:{type(t2)}")

    # 把t2转变为numpy数组(nd数组),不共享内存
    n2 = t2.numpy().copy()
    print(f"n2:{n2},type:{type(n2)}")

    n2[0] = 100  # 修改了n2，n2和t2都会改变
    print(f"t2:{t2},type:{type(t2)}")

    print(f"n2:{n2},type:{type(n2)}")
    print("*"*40)


# nd数组转张量
def demo2():
    # 创建numpy数组
    n1=np.array([1,2,3])
    print(f"n1:{n1},type:{type(n1)}")

    # 把n1转换为张量(共享内存)
    t1=torch.from_numpy(n1)
    print(f"t1:{t1},type:{type(t1)}")
    print("*"*40)

    t1[0]=100
    print(f"n1:{n1},type:{type(n1)}")

    print(f"t1:{t1},type:{type(t1)}")
    print("*"*40)

    n2 = np.array([1, 2, 3])
    print(f"n2:{n2},type:{type(n2)}")

    # 把n2转换为张量(不共享内存)
    t2 = torch.tensor(n2)
    print(f"t2:{t2},type:{type(t2)}")
    print("*"*40)

    n2[0] = 100
    print(f"n2:{n2},type:{type(n2)}")

    print(f"t2:{t2},type:{type(t2)}")

# 从标量张量(只有一个值的张量)中提取内容
def demo3():
    # 创建标量
    t1=torch.tensor(1)
    # 提取
    a=t1.item()
    print(f"t1:{t1},type:{type(t1)}")
    print(f"a:{a},type:{type(a)}")

if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()