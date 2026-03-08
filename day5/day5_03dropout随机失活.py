"""

欠拟合：
	数据集太多，神经网络设计的太简单，让模型学不会正确的特征

过拟合：
	数据集太少，神经网络设计的太复杂，学习过程会记住每一个小细节，只要有一点小细节对不上就不能准确识别(模型不是在学习规律，而是在背诵特征)

正则化的作用：
	缓解模型的过拟合情况

正则化的方式：
	L1正则化：
		权重可以变为0，相当于降维
	L2正则化：
		权重可以无限接近于0
	DropOut：
		随机失活，每批次样本训练时，随机让一部分神经元死亡，防止一些特征对结果的影响较大(防止过拟合)
		公式：
			神经元死亡概率为：p
		未死亡的神经元缩放 1 / (1 - p) 倍	
	BN(批量化归一)：
		
"""
import torch
import torch.nn as nn


def dm01():
	# 手动定义输入特征
	x1 = torch.randint(0, 10, (1, 4), dtype = torch.float)
	print(f"x1:{x1.data}")
	
	# 定义隐藏层1层
	linear1 = nn.Linear(4, 5)
	# 加权求和
	l1 = linear1(x1)
	print(f"l1:{l1.data}")
	# 激活函数
	putout = torch.relu(l1)  # max(0,x)
	print(f"putout:{putout.data}")
	# 随机失活      只有训练阶段有，测试阶段没有
	dropout = nn.Dropout(p = 0.5)  # 隐藏层的神经元死亡概率为0.5，未死亡的神经元缩放 1 / (1 - 0.5)倍
	d1 = dropout(putout)
	print(f"d1:{d1.data}")


if __name__ == '__main__':
	dm01()
