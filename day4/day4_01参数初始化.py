"""
参数初始化的目的:
	1. 防止梯度消失或梯度爆炸
	2. 提高收敛速度
	3. 打破对称性(给不同的神经元初始化不同的权重和偏置值，让他们的计算结果不同)
	
参数初始化的方法:
	无法打破对称性的:
		全0初始化
		全1初始化
		固定值初始化
	
	可以打破对称性的:
		均匀分布随机初始化
		正态分布初始化
		kaiming初始化      kaiming不能用于初始化偏置值
		xavier初始化       xavier不能用于初始化偏置值
		
总结：
	1. 掌握kaiming    xavier  全0(有时用于初始化偏置值)
	2. 关于初始化的选择:
		激活函数 ReLu 及其系列：优先用 kaiming
		激活函数非 ReLu ：优先用 xavier
		如果是浅层网络可以考虑用 均匀分布随机初始化
"""

import torch.nn as nn


# 均匀分布随机初始化 uniform_()
def dm01():
	# 创建一个线性层，输入维度 5 ，输出维度 3
	linear = nn.Linear(5, 3)
	# 对权重(w)进行均匀分布随机初始化，从0-1均匀分布随机产生参数
	nn.init.uniform_(linear.weight)
	# 对偏置值(b)进行均匀分布随机初始化，从0-1均匀分布随机产生参数
	nn.init.uniform_(linear.bias)
	# 打印初始化后得权重和偏置值
	print(linear.weight.data)
	print(linear.bias.data)


# 固定初始化 constant_()
def dm02():
	# 创建一个线性层，输入维度 5 ，输出维度 3
	linear = nn.Linear(5, 3)
	# 对权重(w)进行固定初始化,固定值为 6
	nn.init.constant_(linear.weight, 6)
	# 打印初始化后得权重
	print(linear.weight.data)


# 全 0 初始化 zeros_()
def dm03():
	# 创建一个线性层，输入维度 5 ，输出维度 3
	linear = nn.Linear(5, 3)
	# 对权重(w)进行固定初始化,固定值为 6
	nn.init.zeros_(linear.weight)
	# 打印初始化后得权重
	print(linear.weight.data)


# 全 1 初始化 ones_()
def dm04():
	linear = nn.Linear(5, 3)
	nn.init.ones_(linear.weight)
	print(linear.weight.data)


# 正态分布随机初始化 normal_()
def dm05():
	linear = nn.Linear(5, 3)
	nn.init.normal_(linear.weight)
	print(linear.weight.data)


# kaiming初始化    
def dm06():
	linear = nn.Linear(5, 3)
	# kaiming正态分布初始化
	nn.init.kaiming_normal_(linear.weight)
	print(linear.weight.data)
	
	# kaiming均匀分布初始化
	nn.init.kaiming_uniform_(linear.weight)
	print(linear.weight.data)


# xavier初始化
def dm07():
	linear = nn.Linear(5, 3)
	# xavier正态分布初始化
	nn.init.xavier_normal_(linear.weight)
	print(linear.weight.data)
	
	# xavier均匀分布初始化
	nn.init.xavier_uniform_(linear.weight)
	print(linear.weight.data)


if __name__ == '__main__':
	# dm01()    #均匀分布随机初始化
	# dm02()    #固定值初始化
	# dm03()    #全 0 初始化
	# dm04()    #全 1 初始化
	# dm05()    #正态分布随机初始化
	# dm06()    #kaiming 初始化
	dm07()
