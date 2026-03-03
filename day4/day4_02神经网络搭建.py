"""
深度学习的四个步骤:
	1. 准备数据
	2. 搭建神经网络
	3. 模型训练         反向传播在这一步
	4. 模型测试
	
神经网络搭建流程:
	1. 定义一个类，继承nn.Module
	2. 在_init_()方法中，搭建神经网络
	3. 在forward()方法中，完成前向传播
"""

import torch
import torch.nn as nn
from torchsummary import summary  # 计算模型参数，查看模型结构


# todo:1. 搭建神经网络，自定义继承nn.Module
class ModuleDemo(nn.Module):
	# todo:1.1 在init魔法方法中，完成初始化：父类成员以及神经网络搭建
	def __init__(self):
		# 1.1 初始化父类成员
		super().__init__()
		# 1.2 搭建神经网络：隐藏层 + 输出层
		# 隐藏层1：输入特征数3，输出特征数3
		self.linear1 = nn.Linear(3, 3)
		# 隐藏层2：输入特征数3，输出特征数2
		self.linear2 = nn.Linear(3, 2)
		# 输出层：输入特征数2，输出特征数2
		self.output = nn.Linear(2, 2)
		
		# 1.3 对隐藏层进行参数初始化
		# 隐藏层1
		nn.init.xavier_normal_(self.linear1.weight)
		nn.init.zeros_(self.linear1.bias)
		
		# 隐藏层2
		nn.init.kaiming_normal_(self.linear2.weight)
		nn.init.zeros_(self.linear2.bias)
	
	# todo:1.2 前向传播：输入层 -> 隐藏层 -> 输出层
	# forward方法，在实例化模型的时候，底层会自动调用该函数，该函数中为初始化定义的层layer传入数据，进行前向传播
	# 该方法固定名字为forward
	def forward(self, x):
		# 1.1 第一层隐藏层计算：加权求和 + 激活函数(sigmoid)
		
		# 分解版写法
		# x1=self.linear1(x)      #加权求和
		# x2=torch.sigmoid(x1)    #激活函数
		
		# 合并版写法
		x = torch.sigmoid(self.linear1(x))  # 先加权求和，再激活函数
		
		# 1.2 第二层隐藏层计算：加权求和 + 激活函数(relu)
		x = torch.relu(self.linear2(x))
		
		# 1.3 第三层输出层计算：加权求和 + 激活函数(softmax)
		# dim=-1表示按行计算，一条样本一条样本的处理
		x = torch.softmax(self.output(x), dim = -1)
		
		# 1.4 返回预测值
		return x


# todo ：2. 模型训练
def train():
	# 1. 创建模型对象
	my_module = ModuleDemo()
	print(my_module)
	
	# 2. 创建数据集样本
	# 这里必须是 3 列，因为在搭建神经网络的时候就明确规定了第一层隐藏层只有三个特征输入
	# 此处的 6 行 3 列可以理解为有六组数据，一组有三个特征
	data = torch.randn(6, 3)
	print(f"data:{data}")  # (5,3)
	
	# 3. 调用神经网络模型进行训练
	output = my_module(data)
	print(f"output:{output}")  # (5,2)
	
	# 4. 计算和查看模型参数
	print("=======计算模型参数======")
	# 参1：神经网络模型对象    参2：输入数据维度
	summary(my_module, (5, 3))
	
	print("=======查看模型参数======")
	for name, param in my_module.named_parameters():
		print(f"name:{name}")
		print(f"param:{param}")



if __name__ == '__main__':
	train()
