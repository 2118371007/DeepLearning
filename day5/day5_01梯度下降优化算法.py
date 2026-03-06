"""

梯度下降相关介绍：
	概述：
		梯度下降是结合 本次损失函数的导数(作为梯度)，基于学习率来更新权重的
	公式：
		w新 = w旧 - 学习率 * 梯度	
	存在的问题：
		1. 遇到平缓区域，梯度下降(权重更新)可能会变得缓慢
		2. 可能会遇到鞍点(梯度为0)的情况，导致不能正常更新权重
		3. 可能会遇到局部最小值(局部最优解)，错过真正的最优解
	解决思路：
		从上述的梯度或者学习率入手进行优化：动量法Momentum、自适应学习率AdaGrad,RMSProp、综合衡量Adam
	
	动量法:Momentum
		动量法公式：
			St = β * St-1 + (1 - β) * Gt
		解释：
			St：     本次的指数移动加权平均结果
			β：      调节权重系数，越大数据越平缓，历史指数移动加权平均权重越大，本次梯度权重越小
			St-1：   历史的指数移动加权平均结果
			Gt：     本次计算出的梯度(不考虑历史梯度)
		加入动量法后的梯度更新公式：
			w新 = w旧 - 学习率 * St
			
	自适应学习率:AdaGrad
		公式：
			累计平方梯度：
			St = St-1 + Gt * Gt
		解释：
			St：     累计平方梯度
			St-1：   历史累计平方梯度
			Gt：     本次的梯度
		学习率：
			学习率 = 学习率 / (sqrt(St) + 小常数)
			小常数为：1e-10  1乘以十的的负十次方      目的是防止分母变为0
		梯度下降公式：
			w新 = w旧 - 本次的学习率 * Gt
		缺点：
			可能会导致学习率过早或过量的降低，导致模型后期学习率太小，较难找到最优解
			
			
	自适应学习率:RMSProp
		公式：
			指数加权平均  累计历史平方梯度：
			St = β * St-1 + (1 - β) * Gt * Gt
		解释：
			St：     累计平方梯度
			St-1：   历史累计平方梯度
			Gt：     本次的梯度
			β：      调和权重系数
		学习率：
			学习率 = 学习率 / (sqrt(St) + 小常数)
			小常数为：1e-10  1乘以十的的负十次方      目的是防止分母变为0
		梯度下降公式：
			w新 = w旧 - 本次的学习率 * Gt
		优点：
			通过引入 β 调节历史平方梯度的权重
			
	自适应矩估计:Adam
		思路：
			即优化学习率，又有话梯度
		公式：
			一阶矩：算均值
				Mt = β1 * Mt-1 + (1 - β1) * Gt          充当梯度
				St = β2 * St-1 + (1 - β2) * Gt * Gt     充当学习率
			二阶矩：梯度的方差
				Mt^ = Mt / (1 - β1 ^ t)
				St^ = St / (1 - β2 ^ t)
			权重更新公式：
				w新 = w旧 - (学习率 / (sqrt(St^) + 小常数)) * Mt^
			大白话翻译：
				Adam = RMSProp + Momentum
				
总结：如何选择梯度下降优化算法
	简单任务和较小的模型：
		SGD和动量法
	复杂任务，或者有大量数据：
		Adam
	需要处理稀疏数据或者问本数据：
		AdaGrad，RMSProp
"""

import torch
import torch.nn as nn
import torch.optim as optim


# 定义函数演示 动量法
def dm_Momentum():
	# 1. 定义权重
	w = torch.tensor([1], dtype = torch.float, requires_grad = True)
	
	# 2. 定义损失函数
	criterion = (w ** 2) / 2
	
	# 3. 创建优化器 -> 基于SGD随机梯度下降，加入参数Momentum就是动量法
	# 参1：待优化的参数  参2：学习率  参3：动量参数(调节权重系数，公式中的β)
	optimizer = optim.SGD([w], lr = 0.1, momentum = 0.9)
	
	for epoch in range(20):
		criterion = (w ** 2) / 2
		# 4. 计算梯度值：梯度清零 + 反向传播 + 参数更新
		optimizer.zero_grad()
		criterion.sum().backward()
		optimizer.step()
		
		print(f"第{epoch + 1}轮：w:{w}, 梯度:{w.grad}")


# 定义函数演示 自适应学习率AdaGrad
def dm_AdaGrad():
	# 1. 定义权重
	w = torch.tensor([1], dtype = torch.float, requires_grad = True)
	
	# 2. 定义损失函数
	criterion = (w ** 2) / 2
	
	# 3. 创建优化器 -> 基于AdaGrad
	optimizer = optim.Adagrad([w], lr = 0.1)
	
	for epoch in range(20):
		criterion = (w ** 2) / 2
		# 4. 计算梯度值：梯度清零 + 反向传播 + 参数更新
		optimizer.zero_grad()
		criterion.sum().backward()
		optimizer.step()
		
		print(f"第{epoch + 1}轮：w:{w}, 梯度:{w.grad}")


# 定义函数演示 自适应学习率RMSProp
def dm_RMSProp():
	# 1. 定义权重
	w = torch.tensor([1], dtype = torch.float, requires_grad = True)
	
	# 2. 定义损失函数
	criterion = (w ** 2) / 2
	
	# 3. 创建优化器 -> 基于RMSprop
	# 参1：待优化的参数  参2：学习率  参3：调节权重系数，公式中的β
	optimizer = optim.RMSprop([w], lr = 0.1, alpha = 0.9)
	
	for epoch in range(20):
		criterion = (w ** 2) / 2
		# 4. 计算梯度值：梯度清零 + 反向传播 + 参数更新
		optimizer.zero_grad()
		criterion.sum().backward()
		optimizer.step()
		
		print(f"第{epoch + 1}轮：w:{w}, 梯度:{w.grad}")


# 定义函数演示 自适应矩估计Adam
def dm_Adam():
	# 1. 定义权重
	w = torch.tensor([1], dtype = torch.float, requires_grad = True)
	
	# 2. 定义损失函数
	criterion = (w ** 2) / 2
	
	# 3. 创建优化器 -> 基于RMSprop
	# 参1：待优化的参数  参2：学习率  参3：调节权重系数，公式中的β(第一个是梯度用的，第二个是学习率用的)
	optimizer = optim.Adam([w], lr = 0.1, betas = (0.9, 0.999))
	
	for epoch in range(20):
		criterion = (w ** 2) / 2
		# 4. 计算梯度值：梯度清零 + 反向传播 + 参数更新
		optimizer.zero_grad()
		criterion.sum().backward()
		optimizer.step()
		
		print(f"第{epoch + 1}轮：w:{w}, 梯度:{w.grad}")


if __name__ == '__main__':
	# dm_Momentum()
	# dm_AdaGrad()
	# dm_RMSProp()
	dm_Adam()
