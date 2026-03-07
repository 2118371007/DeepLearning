"""

学习率的影响：
	学习率越小，梯度下降越慢
	学习率越大，梯度下降越快，可能会越过最小值，造成震荡，甚至不收敛(梯度爆炸)
	
学习率衰减策略介绍：
	目的：
		较之于AdaGrad  RMSProp Adam  ，可以通过等间隔，指定间隔，指数来手动控制学习率衰减
	
	分类：
		等间隔学习率衰减
		指定间隔学习率衰减
		指数学习率衰减
		
等间隔学习率衰减：
	step_size：  间隔的轮数，即多少轮调整一次学习率
	gama：       学习率衰减系数，即 lr新 = lr旧 * gama
	
指定间隔学习率衰减：
	milestones = [50,125,165]       要在那几轮调整学习率
	gama：       学习率衰减系数，即 lr新 = lr旧 * gama
	
指数间隔学习率衰减：
	前期学习率衰减更快，中期慢，后期更慢，更符合梯度下降规律
	公式：
		lr新 = lr旧 * gama ** epoch
	

总结：
	等间隔学习率衰减：
		优点：
			直观，易于调试，适用于大批量数据
		缺点：
			学习率变化较大，可能跳过最优解
		使用场景：
			大型数据集，较为简单的任务
	
	指定间隔学习率衰减：
		优点：
			易于调试，稳定训练过程
		缺点：
			在某些情况下可能衰减过快，导致优化提前停滞
		使用场景：
			对训练平稳性要求较高的场景
			
	指数学习率衰减：
		优点：
			平滑，考虑历史更新，收敛稳定性较强
		缺点：
			超参调节较复杂，可能需要更多的资源
		使用场景：
			高精度训练，避免过快的收敛
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# 定义函数，演示等间隔学习率衰减
def dm01():
	# 定义学习率、训练轮数、每轮训练的批次数
	lr, epochs, iteration = 0.1, 200, 10
	
	# 定义真实值
	y_true = torch.tensor([0], dtype = torch.float)
	
	# 定义输入特征
	x = torch.tensor([1], dtype = torch.float)
	
	# 定义权重
	w = torch.tensor([1], dtype = torch.float, requires_grad = True)
	
	# 创建优化器对象->动量法，加速收敛，减少震荡
	optimizer = optim.SGD([w], lr = lr, momentum = 0.9)
	
	# 创建等间隔学习率衰减对象
	# 参1：优化器对象  参2：间隔的轮数(多少轮调整一次学习率)  参3：学习率衰减系数
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.5)
	
	# 创建两个列表，分别表示训练轮数和每轮训练用的学习率
	lr_list, epoch_list = [], []
	
	# 循环遍历训练轮数，进行具体的训练
	for epoch in range(epochs):
		# 获取当前训练轮数和学习率，并存到列表中
		epoch_list.append(epoch)
		# get_last_lr()为获取最后的学习率
		lr_list.append(scheduler.get_last_lr())
		
		# 循环遍历每轮每批次进行训练
		for i in range(iteration):
			# 计算预测值，然后基于损失函数计算损失
			y_pred = w * x
			# 计算损失，最小二乘法
			loss = (y_pred - y_true) ** 2
			# 梯度清零，反向传播，更新权重
			optimizer.zero_grad()
			loss.sum().backward()
			optimizer.step()
		
		# 更新学习率
		scheduler.step()
	
	# 打印结果、可视化
	print(f"lr_list:{lr_list}")
	plt.plot(epoch_list, lr_list)
	plt.xlabel("epochs")
	plt.ylabel("lr")
	plt.show()


# 定义函数，演示指定间隔学习率衰减
def dm02():
	# 定义学习率、训练轮数、每轮训练的批次数
	lr, epochs, iteration = 0.1, 200, 10
	
	# 定义真实值
	y_true = torch.tensor([0], dtype = torch.float)
	
	# 定义输入特征
	x = torch.tensor([1], dtype = torch.float)
	
	# 定义权重
	w = torch.tensor([1], dtype = torch.float, requires_grad = True)
	
	# 创建优化器对象->动量法，加速收敛，减少震荡
	optimizer = optim.SGD([w], lr = lr, momentum = 0.9)
	
	# 创建等间隔学习率衰减对象
	# 参1：优化器对象  参2：第几轮的时候衰减  参3：学习率衰减系数
	scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50, 125, 165], gamma = 0.5)
	
	# 创建两个列表，分别表示训练轮数和每轮训练用的学习率
	lr_list, epoch_list = [], []
	
	# 循环遍历训练轮数，进行具体的训练
	for epoch in range(epochs):
		# 获取当前训练轮数和学习率，并存到列表中
		epoch_list.append(epoch)
		# get_last_lr()为获取最后的学习率
		lr_list.append(scheduler.get_last_lr())
		
		# 循环遍历每轮每批次进行训练
		for i in range(iteration):
			# 计算预测值，然后基于损失函数计算损失
			y_pred = w * x
			# 计算损失，最小二乘法
			loss = (y_pred - y_true) ** 2
			# 梯度清零，反向传播，更新权重
			optimizer.zero_grad()
			loss.sum().backward()
			optimizer.step()
		
		# 更新学习率
		scheduler.step()
	
	# 打印结果、可视化
	print(f"lr_list:{lr_list}")
	plt.plot(epoch_list, lr_list)
	plt.xlabel("epochs")
	plt.ylabel("lr")
	plt.show()


# 定义函数，演示指数学习率衰减
def dm03():
	# 定义学习率、训练轮数、每轮训练的批次数
	lr, epochs, iteration = 0.1, 200, 10
	
	# 定义真实值
	y_true = torch.tensor([0], dtype = torch.float)
	
	# 定义输入特征
	x = torch.tensor([1], dtype = torch.float)
	
	# 定义权重
	w = torch.tensor([1], dtype = torch.float, requires_grad = True)
	
	# 创建优化器对象->动量法，加速收敛，减少震荡
	optimizer = optim.SGD([w], lr = lr, momentum = 0.9)
	
	# 创建等间隔学习率衰减对象
	# 参1：优化器对象  参2：学习率衰减系数
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
	
	# 创建两个列表，分别表示训练轮数和每轮训练用的学习率
	lr_list, epoch_list = [], []
	
	# 循环遍历训练轮数，进行具体的训练
	for epoch in range(epochs):
		# 获取当前训练轮数和学习率，并存到列表中
		epoch_list.append(epoch)
		# get_last_lr()为获取最后的学习率
		lr_list.append(scheduler.get_last_lr())
		
		# 循环遍历每轮每批次进行训练
		for i in range(iteration):
			# 计算预测值，然后基于损失函数计算损失
			y_pred = w * x
			# 计算损失，最小二乘法
			loss = (y_pred - y_true) ** 2
			# 梯度清零，反向传播，更新权重
			optimizer.zero_grad()
			loss.sum().backward()
			optimizer.step()
		
		# 更新学习率
		scheduler.step()
	
	# 打印结果、可视化
	print(f"lr_list:{lr_list}")
	plt.plot(epoch_list, lr_list)
	plt.xlabel("epochs")
	plt.ylabel("lr")
	plt.show()


if __name__ == '__main__':
	# dm01()
	# dm02()
	dm03()
