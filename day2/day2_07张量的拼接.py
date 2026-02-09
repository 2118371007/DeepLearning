"""

    cat()       不改变维度数，除了拼接的那个维度外，其他维度必须保持一致
    stack()     会改变维度，所有的维度必须保持一致
"""


import torch

#设置随机数种子
torch.manual_seed(888)
t1=torch.randint(1,10,(2,3))
t2=torch.randint(1,10,(2,3))
print(f"t1:{t1},shape:{t1.shape}\nt2:{t2},shape:{t2.shape}")


#定义函数演示cat()的用法
def demo1():
    #把t1和t2按照零轴拼接
    t3=torch.cat([t1,t2],0)                     #(2,3)+(2,3)=(4,3)
    print(f"t3:{t3},shape:{t3.shape}")

    #按照一轴拼
    t4=torch.cat([t1,t2],1)                     #(2,3)+(2,3)=(2,6)
    print(f"t4:{t4},shape:{t4.shape}")

#定义函数演示stack()的用法
#在零轴上拼接两个张量，结果就是在零轴上变成二，一二轴为原张量的大小
#在一轴上拼接两个张量，结果就是在一轴上变成二，零二轴为原张量的大小
def demo2():
    #按照零轴拼
    t3=torch.stack([t1,t2],0)                   #(2,3)+(2,3)=(2,2,3)
    print(f"t3:{t3},shape:{t3.shape}")

    #按照一轴拼
    t4=torch.stack([t1,t2],1)                   #(2,3)+(2,3)=(2,2,3)
    print(f"t4:{t4},shape:{t4.shape}")

    #按照二轴拼
    t5=torch.stack([t1,t2],2)                   #(2,3)+(2,3)=(2,3,2)
    print(f"t5:{t5},shape:{t5.shape}")


if __name__ == '__main__':
    # demo1()
    demo2()