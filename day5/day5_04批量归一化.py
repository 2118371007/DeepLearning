"""
批量归一化，也属于正则化的一种，也是用来解决过拟合情况的

批量归一化：
	思路：
		先对数据做标准化(会丢失一些信息)，然后再对数据做缩放(λ，理解为权重w)和平移(β，立即为偏置b)找补回一些信息
		缩小数据差异，更好的训练模型
	应用场景：
		计算机视觉使用较多
	
	BatchNorm1d：主要用于全连接层或处理一维数据的网络，例如文本数据，他接受的形状为(N ,num_features)的张量作为输入
	BatchNorm2d：主要用于卷积神经网络，处理二维图像数据或特征图，他接受的形状为(N ,C ,H ,W)的张量作为输入
	BatchNorm3d：主要用于三维卷积神经网络(3D CNN)，处理三维数据，例如视频或医学图像，他接受的形状为(N ,C ,D ,H ,W)的张量作为输入
"""

import torch
import torch.nn as nn


# 定义函数，处理二维数据
def dm01():
	# 创建一张图像样本数据
	# 1张图片，2个通道，3行4列(像素点)
	input_2d = torch.randn(size = (1, 2, 3, 4))
	print(f"input_2d:{input_2d}")
	
	# 创建批量归一化层(BN层)
	# 参1：输入特征数 = 通道数     参2：噪声(小常数)      参3：动量值，用于计算移动平均统计量      参4：表示使用可学习的变换参数(λ和β)
	bn2d = nn.BatchNorm2d(num_features = 2, eps = 1e-5, momentum = 0.1, affine = True)
	
	# 对数据进行批量归一化处理
	output_2d = bn2d(input_2d)
	print(f"output_2d:{output_2d}")


# 定义函数，处理一维数据
def dm02():
	# 创建样本数据
	# 2条样本，每条样本有2个特征
	input_1d = torch.randn(size = (2, 2))
	print(f"input_1d:{input_1d}")
	
	# 创建线性层
	linear = nn.Linear(2, 4)
	# 对数据进行线性变化
	l1 = linear(input_1d)
	
	# 创建批量归一化层(BN层)
	# 输入特征数
	bn1d = nn.BatchNorm1d(num_features = 4)
	
	# 对线性结果l1进行归一化处理
	output_1d = bn1d(l1)
	
	print(f"output_1d:{output_1d}")


if __name__ == '__main__':
	# dm01()
	dm02()
