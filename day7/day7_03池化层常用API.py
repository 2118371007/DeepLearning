"""

池化层：
	目的：
		降维
	思路：
		最大池化    和卷积一样的算法，只不过是用池化核去算对应位置的最大值
		平均池化    用池化核去算对应位置的平均值
	特点：
		池化不会改变数据的通道数
"""
import torch
import torch.nn as nn


# 定义函数，演示单通道池化
def dm01():
	# 定义单通道的图片数据(1,3,3)1通道，宽高都是3
	input1 = torch.tensor([
		[
			[0, 1, 2],
			[3, 4, 5],
			[6, 7, 8]
		]
	])
	print(f"input1:{input1},shape:{input1.shape}")
	# 创建池化层对象
	#                 池化核(窗口)大小  步长  填充
	pool1 = nn.MaxPool2d(2, 1, 0)
	output1 = pool1(input1)
	print(f"output1:{output1},shape:{output1.shape}")
	pool2 = nn.AvgPool2d(2, 1, 0)
	output2 = pool2(input1)
	print(f"output2:{output2},shape:{output2.shape}")


def dm02():
	# 创建多通道的图片数据(3,3,3)，3通道，宽高都是3
	input1 = torch.tensor([
		[
			[0, 1, 2],
			[3, 4, 5],
			[6, 7, 8]
		],
		[
			[0, 1, 2],
			[3, 4, 5],
			[6, 7, 8]
		],
		[
			[0, 1, 2],
			[3, 4, 5],
			[6, 7, 8]
		]
	])
	print(f"input1:{input1},shape:{input1.shape}")
	pool1 = nn.MaxPool2d(2, 1, 0)
	output1 = pool1(input1)
	print(f"output1:{output1},shape:{output1.shape}")
	pool2 = nn.AvgPool2d(2, 1, 0)
	output2 = pool2(input1)
	print(f"output2:{output2},shape:{output2.shape}")


if __name__ == '__main__':
	dm01()
	dm02()
