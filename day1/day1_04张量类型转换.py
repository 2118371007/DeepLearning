import torch

# 直接使用type(torch.元素类型)转换    type需要掌握
def fun1():
    t1=torch.tensor([1,2,3],dtype=torch.float)
    print(f"t1:{t1},元素类型:{t1.dtype},张量类型:{type(t1)}")
    print("*"*40)
    t2=t1.type(torch.int)
    print(f"t2:{t2},元素类型:{t2.dtype},张量类型:{type(t2)}")

if __name__ == '__main__':
    fun1()