import torch
# 设计随机数种子(按照当前时间戳生成随机数)
# torch.initial_seed
# 手动指定一个随机数种子  manual_seed需要掌握
torch.manual_seed(6)

# 创建线性张量
def fun1():
    # 创建指定范围的线性张量
    # 参数1:起始值,参数2:结束值,参数3:步长   arange需要掌握
    t1=torch.arange(0,10,2)
    print(f"t1:{t1}")
    print("*"*40)

    # 创建指定范围的线性张量(等差数列)
    # 参数1:起始值,参数2:结束值,参数3:元素个数   linspace需要掌握
    t2=torch.linspace(1,10,4)
    print(f"t2:{t2}")
    print("*"*40)


# 创建随机张量
def fun2():
    # 前面已经设置过了随机数种子，这里就不必再次设置
    # 创建一个三行四列的均匀分布的随机张量,范围(0,1)   rand需要掌握
    t1=torch.rand(3,4)
    print(f"t1:{t1}")
    print("*"*40)
    # 创建一个三行四列的正态分布的随机张量     randn需要掌握
    t2=torch.randn(3,4)
    print(f"t2:{t2}")
    print("*"*40)
    # 创建一个范围(0,10),三行四列的随机整数张量       randint需要掌握
    t3=torch.randint(0,10,(3,4))
    print(f"t3:{t3}")
    print("*"*40)

if __name__ == '__main__':
    fun1()
    fun2()