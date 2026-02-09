"""
add()   sub()   mul()   div()   neg()   对应加减乘除,取反(不修改源数据)
add_()  sub_()   mul_()   div_()   neg_()   对应加减乘除,取反(修改源数据)

可以用+-*/替代

如果是一个张量和数值运算，则张量中每个元素和这个数值运算
"""
import torch
def demo1():
    t1=torch.tensor([1,2,3,4,5,6])
    t2=t1+10
    print(f"t1:{t1}")
    print(f"t2:{t2}")
    t1+=t1
    print(f"t1:{t1}")
    # 其他运算同理

if __name__ == '__main__':
    demo1()