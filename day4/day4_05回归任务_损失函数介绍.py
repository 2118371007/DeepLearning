"""

回归任务常用损失函数：
	MAE(L1Loss):    平均绝对误差
		公式：
			真实值和预测值的误差的绝对值之和除以样本总数
		权重可以降为0，数据会变得稀疏
		
		弊端：
			在0点不平滑，可能会错过极小值
	
	MSE:    均方误差
		公式：
			(真实值 - 预测值)²之和除以样本总数
			
		弊端:
			真实值和预测值差距过大的时候可能会造成梯度爆炸(差值的平方过大)
	Smooth L1:
		就是基于 MAE 和 MSE  的综合，在[-1 ,1]上是 MSE(L2)，在其他段上是 MAE(L1)
		既解决了 L1 的不平滑问题(0点不可道，可能错过极小值)
		又解决了 L2 的梯度爆炸问题
		
			
"""

import torch
import torch.nn as nn


# 定义函数演示 MAE 损失函数
def dm01():
	# 手动定义真实值
	y_true = torch.tensor([2, 3, 4], dtype = torch.float)
	
	# 手动定义预测值
	y_pred = torch.tensor([0.2, 1.8, 2.2], dtype = torch.float, requires_grad = True)
	"""
		真实值和预测值的误差的绝对值之和除以样本总数
		((2 - 0.2) + (3 - 1.8) + (4 - 2.2)) / 3 = 1.6
	"""
	# 定义 MAE 激活函数对象
	criterion = nn.L1Loss()
	
	# 计算损失
	loss = criterion(y_pred, y_true)
	print(f"loss:{loss}")


# 定义函数演示 MSE 损失函数
def dm02():
	# 手动定义真实值
	y_true = torch.tensor([2, 3, 4], dtype = torch.float)
	
	# 手动定义预测值
	y_pred = torch.tensor([0.2, 1.8, 2.2], dtype = torch.float, requires_grad = True)
	"""
		(真实值 - 预测值)²之和除以样本总数
		((2 - 0.2)² + (3 - 1.8)² + (4 - 2.2)²) / 3 = 2.64
	"""
	# 定义 MSE 激活函数对象
	criterion = nn.MSELoss()
	
	# 计算损失
	loss = criterion(y_pred, y_true)
	print(f"loss:{loss}")


# 定义函数演示 SmoothL1 损失函数
def dm03():
	# 手动定义真实值
	y_true = torch.tensor([2, 3, 4], dtype = torch.float)
	
	# 手动定义预测值
	y_pred = torch.tensor([0.2, 1.8, 2.2], dtype = torch.float, requires_grad = True)
	
	# 定义 SmoothL1 激活函数对象
	criterion = nn.SmoothL1Loss()
	
	# 计算损失
	loss = criterion(y_pred, y_true)
	print(f"loss:{loss}")


if __name__ == '__main__':
	# dm01()
	
	# dm02()
	
	dm03()
