"""

基于手机的20列特征预测手机的价格区间(四个区间)

ANN的实现步骤：
	1. 构建数据集
	2. 搭建神经网络
	3. 模型训练
	4. 模型测试
"""

import torch
from torch.utils.data import TensorDataset  # 数据集对象  数据 -> tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader  # 数据加载器
import torch.nn as nn
import torch.optim as optim  # 优化器
from sklearn.model_selection import train_test_split  # 训练集合测试集的划分
import matplotlib.pyplot as plt  # 可视化
import numpy as np  # 数组(矩阵)操作
import pandas as pd  # 数据处理
import time  # 时间模块

from torchsummary import summary  # 模型结构可视化


# todo: 1. 构建数据集
def create_dataset():
	# 1.1加载csv文件数据集
	data = pd.read_csv("./data/手机价格预测.csv")
	
	# 1.2获取x特征列和y标签
	# 选取所有行，除了最后一列的所有列
	x = data.iloc[:, :-1]
	
	# 选取所有行，最后一列
	y = data.iloc[:, -1]
	
	# 1.3把特征列转为浮点型，方便后面自动微分
	x = x.astype(np.float32)
	
	# 1.4切分训练集合测试集
	# 参1：特征 参2：标签 参3：测试集比例(百分二十作为测试集，剩下的都是训练集)
	# 参4：随机种子(保证没次切分的结果都一致) 
	# 参5：样本的分布(即参考y的类别进行数据抽取)，确保训练集和测试集中各个价格的比例和原数据一致，防止训练时全部都是贵手机而影响模型训练
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3, stratify = y)
	
	# 1.5把数据转为张量数据集     数据->张量->数据集->数据加载器
	train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
	test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
	
	# 1.6返回结果
	# 		训练集         测试集         输入特征数(有多少列就有多少个特征)           输出标签数
	return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# todo: 2. 搭建神经网络
class PhonePriceModel(nn.Module):
	def __init__(self, input_dim, output_dim):
		# 初始化父类成员
		super().__init__()
		# 搭建神经网络
		# 隐藏层1
		self.linear1 = nn.Linear(input_dim, 128)
		# 隐藏层2
		self.linear2 = nn.Linear(128, 256)
		# 输出层
		self.output = nn.Linear(256, output_dim)
	
	# 前向传播
	def forward(self, x):
		# 隐藏层1：加权求和 + 激活函数(relu)
		x = torch.relu(self.linear1(x))
		# 隐藏层2：加权求和 + 激活函数(relu)
		x = torch.relu(self.linear2(x))
		# 输出层：加权求和，本来应该用softmax()激活函数做激活函数
		# 但是后面直接用多分类交叉熵CrossEntropyLoss()
		# CrossEntropyLoss() = softmax() + 损失计算
		x = self.output(x)
		return x



# todo: 3. 模型训练
def train(train_dataset, input_dim, output_dim):
	# 创建数据加载器    数据 -> 张量 -> 数据集 -> 数据加载器
	# 参1：数据集对象 参2：批次大小 参3：是否打乱数据(训练集打乱，测试集不打乱)
	train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
	# 创建模型
	model = PhonePriceModel(input_dim, output_dim)
	# 定义损失函数
	criterion = nn.CrossEntropyLoss()
	# 定义优化器
	optimizer = optim.SGD(model.parameters(), lr = 0.001)
	# 开始训练
	# 定义变量存储训练轮数
	epochs = 50
	for epoch in range(epochs):
		# 定义变量，记录每轮训练的损失值和训练批次
		total_loss = 0.0
		batch_num = 0
		# 定义变量，表示训练开始的时间
		start_time = time.time()
		# 开始本轮的各个批次的训练
		for x, y in train_loader:
			# 切换模型状态
			# model.train()为训练模式 model.eval()为测试模式
			model.train()
			# 预测模型
			y_pred = model(x)
			# 计算损失值
			loss = criterion(y_pred, y)
			# 梯度清零，反向传播，优化器更新参数
			optimizer.zero_grad()
			loss.sum().backward()
			optimizer.step()
			# 累计损失值
			# 把本轮每批次的平均损失加起来
			total_loss += loss.item()
			batch_num += 1
		# 每轮训练结束，打印训练信息
		print(f"epoch:{epoch + 1},loss:{total_loss / batch_num:.4f},time:{time.time() - start_time:.2f}s")
	# 多轮训练结束，保存模型参数
	# 参1：模型对象的参数(权重矩阵，偏置矩阵)    参2：保存的路径(后缀可以用pth pkl pickle)
	torch.save(model.state_dict(), "./model/phone_price_model.pth")


# todo: 4. 模型测试
def evaluate(test_dataset, input_dim, output_dim):
	# 创建神经网络分类对象
	model = PhonePriceModel(input_dim, output_dim)
	# 加载模型参数
	model.load_state_dict(torch.load("./model/phone_price_model.pth"))
	# 创建测试集的数据加载器
	# 参1：数据集对象 参2：批次大小 参3：是否打乱数据(训练集打乱，测试集不打乱)
	test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)
	# 定义变量，记录预测正确的样本数
	correct = 0
	# 从数据加载器中获取每批次的数据
	for x, y in test_loader:
		# 切换模型状态
		model.eval()
		# 模型预测
		y_pred = model(x)
		# print(f"y_pred:{y_pred}")
		# 根据预测结果得到类别 argmax()获取最大值的下标
		# dim = 1表示逐行处理
		y_pred = torch.argmax(y_pred, dim = 1)
		# print(f"y_pred:{y_pred}")
		
		# 统计预测正确的样本数
		# y_pred == y将预测结果和真实答案比对，产生一个布尔张量（如 [True, False, True...]）
		# 用预测值和真实值做对比，如果预测正确，结果就是true，值为1，再求和就能得到预测正确的样本数
		correct += (y_pred == y).sum()
	# 模型预测结束，计算并打印模型准确率
	print(f"准确率:{correct / len(test_dataset):.4f}")


if __name__ == '__main__':
	# 准备数据集
	train_dataset, test_dataset, input_dim, output_dim = create_dataset()
	# train(train_dataset, input_dim, output_dim)
	# 构建网络模型
	# model = PhonePriceModel(input_dim, output_dim)
	# 计算模型参数    
	# 参1：模型对象   参2：输入数据形状(批次大小，输入特征数)
	# summary(model, (16, input_dim))
	# print(f"summary:{summary}")
	evaluate(test_dataset, input_dim, output_dim)
