"""

深度学习项目的步骤：
	1. 准备数据集
		这里使用计算机视觉模块 torchvision 自带的 CIFAE10 数据集
	2. 搭建神经网络
	3. 训练模型
	4. 测试模型
	
卷积层：
	提取局部的图像特征 -> 特征图
	每个卷积核都是一个神经元
	卷积层参数计算公式：
		参数个数 = 输入通道数 * 卷积核尺寸 * 卷积核个数 + 卷积核个数
池化层：
	降维，有最大池化和平均池化
	池化只在HW上做调整，不调整通道数
	池化层步长为 2 就表示图像尺寸缩小一半
	
	
优化思路：
	1. 添加卷积核的输出参数(增加卷积核的个数)
	2. 增加全连接层的数量
	3. 降低学习率
	4. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary


# 1. 准备数据集
def create_dataset():
	# 训练集
	# 参1：数据集存放的位置    参2：是否是训练集   参3：数据转换(转换为张量)  参4：是否下载数据集(如果数据集不存在)
	train_dataset = CIFAR10("./data", train = True, transform = ToTensor(), download = True)
	# 测试集
	test_dataset = CIFAR10("./data", train = False, transform = ToTensor(), download = True)
	return train_dataset, test_dataset


# 2. 搭建神经网络
class imgmodel(nn.Module):
	def __init__(self):
		super().__init__()
		# 卷积层1  输入通道数   输出通道数(卷积核个数)   卷积核尺寸   步长   填充
		# CIFAR-10 输入是 3*32*32，经过卷积层的特征图大小为(32-3+2*0)/1+1=30
		self.conv1 = nn.Conv2d(3, 16, 3, 1, 0)
		# 池化层1  卷积核尺寸   步长   填充
		# 大小为30的特征图经过步长为2的池化层，特征图尺寸缩小一半，所以特征图大小为15
		self.pool1 = nn.MaxPool2d(2, 2, 0)
		# 卷积层2
		# 经过卷积层2，特征图大小为(15-3+2*0)/1+1=13
		self.conv2 = nn.Conv2d(16, 32, 3, 1, 0)
		# 池化层2
		# 经过池化层2，特征图尺寸缩小一半，所以特征图大小为6
		self.pool2 = nn.MaxPool2d(2, 2, 0)
		# 卷积层3
		# 经过卷积层3，特征图大小为(6-3+2*0)/1+1=4
		self.conv3 = nn.Conv2d(32, 64, 3, 1, 0)
		# 池化层3
		# 大小为4，特征图尺寸缩小一半，所以特征图大小为2
		self.pool3 = nn.MaxPool2d(2, 2, 0)
		# 全连接层1，输入通道数   输出通道数
		# 16 * 6 * 6是因为上一层卷积层有64个特征图，然后特征图尺寸为2 * 2，总的特征数为64 * 2 * 2
		self.linear1 = nn.Linear(64 * 2 * 2, 256)
		self.linear2 = nn.Linear(256, 64)
		# 输出层
		# 输出特征数为10是因为CIFAR-10数据集有10类
		self.output = nn.Linear(64, 10)
	
	def forward(self, x):
		x = self.pool1(torch.relu(self.conv1(x)))
		x = self.pool2(torch.relu(self.conv2(x)))
		x = self.pool3(torch.relu(self.conv3(x)))
		# print(f"x.shape:{x.shape}")
		# 细节，全连接层只能处理二维数据，要把数据展平
		# 经过卷积层后的数据是四维的，分别表示图片数量，通道数，高，宽
		# x.size(0)是取数据的0轴上的数(图片数量) -1表示自动计算
		x = x.reshape(x.size(0), -1)
		x = torch.relu(self.linear1(x))
		x = torch.relu(self.linear2(x))
		# 后面要用多分类交叉熵，这里输出层就不使用softmax()激活函数
		return self.output(x)


# 3. 训练模型
def train(train_dataset):
	# 创建数据加载器
	dataloader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
	# 创建模型
	model = imgmodel()
	# 定义损失函数
	criterion = nn.CrossEntropyLoss()
	# 定义优化器
	optimizer = optim.Adam(model.parameters(), lr = 0.001)
	# 定义变量记录训练的总轮数
	epochs = 100
	# 开始训练
	for epoch in range(epochs):
		# 定义变量记录总损失和总部样本和预测正确的样本数以及本轮开始训练时的时间
		total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
		# 开始本轮的各个批次训练
		for x, y in dataloader:
			model.train()
			# 前向传播，获得预测值
			y_pred = model(x)
			# 计算损失
			loss = criterion(y_pred, y)
			# 梯度清零，反向传播，优化器更新参数
			optimizer.zero_grad()
			loss.sum().backward()
			optimizer.step()
			# 累加损失和正确数
			total_loss += loss.item() * len(y)
			total_correct += (torch.argmax(y_pred, -1) == y).sum().item()
			# 累加样本数
			total_samples += len(y)
		print(
			f"epoch:{epoch + 1},loss:{total_loss / total_samples:.4f},acc:{total_correct / total_samples},time:{time.time() - start:.2f}s")
	# 保存模型参数
	torch.save(model.state_dict(), "./model/CNN_MODEL.pth")


def evaluate(test_dataset):
	model = imgmodel()
	model.load_state_dict(torch.load("./model/CNN_MODEL.pth"))
	# 创建数据加载器
	dataloader = DataLoader(test_dataset, batch_size = 16, shuffle = False)
	total_correct = 0
	total_samples = 0
	for x, y in dataloader:
		model.eval()
		y_pred = model(x)
		y_pred = torch.argmax(y_pred, 1)
		total_correct += (y_pred == y).sum().item()
		total_samples += len(y)
	print(f"acc:{total_correct / total_samples:.4f}")


# 4. 测试模型
if __name__ == '__main__':
	# 获取数据集和测试集
	train_dataset, test_dataset = create_dataset()
	# print(f"shape:{train_dataset.data.shape}")
	# {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
	# print(f"数据集类别:{train_dataset.class_to_idx}")
	# 展示下标为1111的图片
	# img = train_dataset.data[1111]
	# plt.imshow(img)
	# 把1111图片的标签作为标题
	# plt.title(train_dataset.targets[1111])
	# plt.show()
	# print(1)
	# model = imgmodel()
	# summary(model, (3, 32, 32))
	# train(train_dataset)
	evaluate(test_dataset)
