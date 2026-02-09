"""

涉及到的函数:
    reshape()               在不改变张量内容的前提下，对其形状做改变
    unsqueeze()             在指定的轴上增加一个(1)维度，等价于升维
    squeeze()               删除所有为1的维度，等价于降维
    transpose()             交换指定两个轴的维度
    permute()               交换指定多个轴的维度
    view()                  只能修改连续的张量的形状，连续张量 = 内存中的存储顺序和张量中显示的顺序一致
    contiguous()            把不连续的张量转变为连续的张量，就是基于张量中显示的张量顺序修改内存中的存储顺序
    is_contiguous()         判断张量是否连续
"""
import torch
#设置随机数种子
torch.manual_seed(66)

#定义函数演示reshape()用法
#把张量转换成指定行列的张量(转换前后元素个数必须相等)
def demo1():
    #生成一个三行两列的张量
    t1=torch.randint(1,10,(3,2))
    print(f"t1:{t1}\nshape:{t1.shape},row:{t1.shape[0]},col:{t1.shape[1]}")

    print("*"*40)

    #把t1转换成一个两行三列的张量
    t2=t1.reshape(2,3)
    print(f"t2:{t2}\nshape:{t2.shape},row:{t2.shape[0]},col:{t2.shape[1]}")

#定义函数演示unsqueeze()和squeeze()用法
def demo2():
    #生成一个三行两列的张量
    t1=torch.randint(1,10,(3,2))
    print(f"t1:{t1},shape:{t1.shape}")
    print("*"*40)

    #在t1零轴上增加一个维度
    t2=t1.unsqueeze(0)
    print(f"t2:{t2},shape:{t2.shape}")
    print("*"*40)

    #在t1的一轴上添加一个维度
    t3=t1.unsqueeze(1)
    print(f"t3:{t3},shape:{t3.shape}")
    print("*"*40)

    #在t1的二轴上添加一个维度
    t4=t1.unsqueeze(2)
    print(f"t4:{t4},shape:{t4.shape}")
    print("*"*40)

    t5=torch.randint(1,10,(1,3,2,1,1,2,1))
    print(f"t5:{t5},shape:{t5.shape}")
    print("*"*40)

    #删除t5所有一维的轴
    t6=t5.squeeze()
    print(f"t6:{t6},shape:{t6.shape}")
    print("*"*40)

#定义函数演示transpose()和permute()用法
#transpose()两个参数为要交换的两个轴
#permute()参数为要转变为的轴的顺序
def demo3():
    t1=torch.randint(1,10,(2,3,4))
    print(f"t1:{t1},shape:{t1.shape}")
    print("*"*40)

    #把t1维度(2,3,4)->(4,3,2)
    t2=t1.transpose(0,2)
    print(f"t2:{t2},shape:{t2.shape}")
    print("*"*40)

    #把t1维度(2,3,4)->(4,2,3)
    t3=t1.permute(2,0,1)
    print(f"t3:{t3},shape:{t3.shape}")
    print("*"*40)

#定义函数演示view()和contiguous()和is_contiguous() 用法
def demo4():
    t1=torch.randint(1,10,(2,3))
    print(f"t1:{t1},shape:{t1.shape},is_contiguous:{t1.is_contiguous()}")
    print("*"*40)

    #把t1通过view(转为三行两列的张量)
    t2=t1.view(3,2)
    print(f"t2:{t2},shape:{t2.shape},is_contiguous:{t2.is_contiguous()}")
    print("*"*40)

    #通过transpose()或permute()转换的张量是不连续的
    t3=t1.transpose(0,1)
    print(f"t3:{t3},shape:{t3.shape},is_contiguous:{t3.is_contiguous()}")
    print("*"*40)

    #再次通过view()把t3转变为两行三列
    t4=t3.contiguous().view(2,3)
    print(f"t4:{t4},shape:{t4.shape}，is_contiguous:{t4.is_contiguous()}")
    print("*"*40)


if __name__ == '__main__':
    # demo1()
    # demo2()
    # demo3()
   demo4()