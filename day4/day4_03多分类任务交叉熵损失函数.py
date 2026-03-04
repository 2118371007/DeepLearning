"""

损失函数介绍:
	概述：
		损失函数也叫成本函数、目标函数、代价函数、误差函数 就是用来衡量模型好坏(模型拟合情况)的
	分类：
		分类问题：
			多分类交叉熵损失：CrossEntropyLoss
			二分类交叉熵损失：BECLoss
		回归问题：
			MAE:平均绝对误差
			MSE:均方误差
			Smooth L1：结合上述两个特点做的优化升级

多分类交叉熵损失：CrossEntropyLoss
	设计思路：
		Loss = -∑ylog(S(f(x)))
	简单记忆：
		x:          样本
		f(x):       加权求和
		S(f(x)):	处理后的概率
		y:          样本x属于某一类别的真实概率
	大白话解释：
		损失函数结果 = 正确类别概率的对数的负值
	细节：
		CrossEntropyLoss = Softmax() * 损失计算，后续如果用这个损失函数，输出层就不用额外调用 Softmax()激活函数了
"""

import torch
import torch.nn as nn


def dm01():
	# 1. 手动创建样本的真实值 就是上述公式中的 y
	# 下方真实值的意思是第一个样本属于第一类的概率为百分之百，第二个样本属于第二类的概率为百分之百
	y_true = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype = torch.float, requires_grad = True)
	
	# 2. 手动创建样本的预测值 就是上述公式中的 f(x)
	# 下方预测值的意思是第一个样本属于第一类的概率为0.8，第二类和第三类概率分别为为0.1，第二个样本同理
	y_pred = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype = torch.float)
	
	# 3. 创建多分类交叉熵损失函数
	criterion = nn.CrossEntropyLoss()
	
	# 4. 计算损失值
	loss = criterion(y_pred, y_true)
	print(f"loss:{loss}")


if __name__ == '__main__':
	dm01()
