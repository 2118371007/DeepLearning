import torch
import numpy as np

# tensor(值，类型)需要掌握
# 场景一:标量
def demo1():
    # 创建一个标量(一个单独的数，没有维度)
    t1=torch.tensor(10)
    print("t1:",t1.item(),"type:",t1.type())
    print(f"t1:{t1}","type:",t1.type())
    print("*"*40)

# 场景二:二维列表转张量(高维也可以)
    data=[[1,2,3],[4,5,6]]
    t2=torch.tensor(data)
    print("t2:",t2,"type:",t2.type())
    print(f"t2:{t2}","type:",t2.type())
    print("*"*40)

# 场景三:numpy nd数组转张量
    # 生成一个元素范围为[0,100),3*3的np数组
    data=np.random.randint(0,100,size=(3,3))
    # 创建张量，并且指定数据类型位float
    t3=torch.tensor(data,dtype=torch.float)
    print("t3:",t3,"type:",t3.type())
    print("*"*40)


if __name__ == '__main__':
    demo1()
