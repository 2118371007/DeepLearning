import torch

def fun1():
    # 创建一个三乘三的张量
    t=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    # 创建一个三行四列的全0张量    zeros需要掌握
    t1=torch.zeros(3,4,dtype=torch.int)
    # 创建一个形状和t相同的全0张量
    t2=torch.zeros_like(t)
    print(f"t:{t},type:{type(t)}")
    print("*"*40)
    print(f"t1:{t1},type:{type(t1)}")
    print("*"*40)
    print(f"t2:{t2},type{type(t2)}")
    print("*"*40)

def fun2():
    # 创建一个三乘三的张量
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 创建一个三行四列的全1张量
    t1 = torch.ones(3, 4, dtype=torch.int)
    # 创建一个形状和t相同的全1张量
    t2 = torch.ones_like(t)
    print(f"t:{t},type:{type(t)}")
    print("*" * 40)
    print(f"t1:{t1},type:{type(t1)}")
    print("*" * 40)
    print(f"t2:{t2},type{type(t2)}")
    print("*" * 40)

def fun3():
    # 创建一个三乘三的张量
    t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # 创建一个三行四列的全255张量(指定值)
    t1 = torch.full((3,4),255)
    # 创建一个形状和t相同的全255张量
    t2 = torch.full_like(t,255)
    print(f"t:{t},type:{type(t)}")
    print("*" * 40)
    print(f"t1:{t1},type:{type(t1)}")
    print("*" * 40)
    print(f"t2:{t2},type{type(t2)}")
    print("*" * 40)
if __name__ == '__main__':
    fun1()
    fun2()
    fun3()