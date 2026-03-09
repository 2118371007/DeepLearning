"""

卷积神经网络：
	包含卷积层的神经网络
	组成：
		卷积层：
			用于提取图像的局部特征，结合卷积核(一个卷积核就是一个神经元)实现，处理后的结果叫 特征图
		池化层：
			用于降维，降采样
		全连接层：
			用于预测结果，并输出结果
	
	特征图大小计算方式：
		N = (W - F + 2P) / S + 1
		W:输入图像的宽/高
		F:卷积核大小
		P:填充
		S:步长
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def dm01():
	img1 = plt.imread("./data/img.jpg")
	# HWC
	# print(f"img1:{img1},img1shape:{img1.shape}")
	# 把图像形状 HWC -> CHW   思路：img -> 张量 ->CHW
	img2 = torch.tensor(img1, dtype = torch.float)
	# permute交换通道(2,0,1)表示，把原本的顺序(H,W,C)变为(C,H,W)
	img3 = torch.permute(img2, (2, 0, 1))
	# print(f"img3:{img3},img3shape:{img3.shape}")
	
	# 增加一个维度，表示图像数量  (C,H,W) -> (1,C,H,W)
	img4 = img3.unsqueeze(0)
	# print(f"img4:{img4},img4shape:{img4.shape}")
	# 创建卷积层对象，提取特征图
	#              输入图片的通道数  输出图片的通道数(卷积核个数/特征图个数)  卷积核大小  步长  填充
	conv1 = nn.Conv2d(3, 4, 3, 1, 0)
	
	# 卷积层处理图
	conv_out = conv1(img4)
	# print(f"conv_out:{conv_out},conv_outshape:{conv_out.shape}")
	
	# 查看提取到的4个特征图
	img5 = conv_out[0]
	# 把提取到的特征图(C,H,W) -> (H,W,C)
	img6 = img5.permute(1, 2, 0)
	feature1 = img6[:, :, 0].detach().numpy()
	plt.imshow(feature1)
	plt.show()


if __name__ == '__main__':
	dm01()
