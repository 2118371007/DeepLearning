"""
    sum(),max(),min(),mean()            都有dim参数,0表示列,1表示行   mean()是求平均值
    pow(),sqrt(),exp(),log(),log2(),log10()     没有dim参数         exp()是求e的n次方

需掌握:
    sum(),max(),min(),mean()
    mean()必须是float
"""
import torch
t1=torch.tensor([
    [1,2,3],
    [4,5,6]
],dtype=torch.float)
print(t1)
print("-"*40)

# 带dim参数的函数
# sum()求和
print(t1.sum(dim=0))    #按 列 求和
print(t1.sum(dim=1))    #按 行 求和
print(t1.sum())         #整 体 求和
print("-"*40)

#  max()求最大值,min()求最小值同理
print(t1.max(dim=0))    #按 列 求最大值
print(t1.max(dim=1))    #按 行 求最大值
print(t1.max())         #整 体 求最大值
print("-"*40)

# mean()求平均值
print(t1.mean(dim=0))    #按 列 求平均值
print(t1.mean(dim=1))    #按 行 求平均值
print(t1.mean())         #整 体 求平均值
print("*"*40)

# 不带dim参数的函数
print(t1.pow(3))        #每个元素的三次方
print(t1**3)            #同上
print("-"*40)

print(t1.sqrt())        #每个元素开方
print("-"*40)

print(t1.log())         #每个元素以e为底的对数
print("-"*40)

print(t1.log2())        #每个元素以2为底的对数
print("-"*40)

print(t1.log10())       #每个元素以10为底的对数
print("-"*40)