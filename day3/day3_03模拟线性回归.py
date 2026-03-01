"""

数据转换路径:
    numpy对象(nd数组)->张量(Tensor)->数据集对象(TensorDataset)->数据加载器(DataLoader)
"""

# 导入相关模块
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn模块中有平方损失函数和假设函数
from torch import optim  # optim中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集
import matplotlib.pyplot as plt  # 可视化

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 1.定义函数，创建线性回归样本数据
def create_dataset():
	# 1.1创建数据集对象
	# x是特征，y是真实值，coef是系数，就是反向传播过程中最后得到的权重w，如果提前知道这个w，就能很容易的算出y
	x, y, coef = make_regression(
		n_samples = 100,  # 100条样本
		n_features = 1,  # 1个特征
		noise = 10,  # 噪声，噪声越大，样本点越散，噪声越小，样本点越集中
		coef = True,  # 是否返回系数(反向传播最后得到的权重w)
		random_state = 6,  # 随机种子
		bias = 6.6  # 偏置值
	)
	# make_regression生成的数据是nd数组，必须转换成tensor才能方便后面的操作
	x = torch.tensor(x, dtype = torch.float)
	y = torch.tensor(y, dtype = torch.float)
	
	return x, y, coef


# 2.定义函数，表示训练模型
def train(x, y, coef):
	# 1.创建数据集对象  把tensor->数据集对象->数据加载器
	dataset = TensorDataset(x, y)
	
	# 2.创建数据加载器对象
	# 参1：数据集对象  参2：批次大小 参3：是否打乱数据(训练集打乱，测试集不打乱)
	# 总的数据100条，16条一批，每批(16,16,16,16,16,16,4)
	dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)
	
	# 3.创建初试的线性回归模型
	# 参1：输入特征维度 参2：输出特征维度
	model = nn.Linear(1, 1)
	
	# 4.创建损失函数对象
	criterion = nn.MSELoss()
	
	# 5.创建优化器对象
	# 参1：模型参数   参2：学习率
	optimizer = optim.SGD(model.parameters(), lr = 0.01)
	
	# 6.具体的训练过程
	# 6.1定义变量，分别表示：训练轮数，每轮的平均损失值，训练总损失值，训练的样本数
	epochs = 100
	loss_list = []
	total_loss = 0.0
	total_sample = 0
	
	# 6.2开始训练，按轮训练
	for epoch in range(epochs):  # epoch的值：0,1,2.....99
		# 6.3每轮是分批次训练的，所以从数据加载器中获取批次数据
		for train_x, train_y in dataloader:
			# 6.4模型预测
			y_pred = model(train_x)
			
			# 6.5计算损失值
			loss = criterion(y_pred, train_y.reshape(-1, 1))  # 把y转换为n行1列的数据
			
			# 6.6计算总损失和样本批次数
			total_loss += loss.item()
			total_sample += 1
			
			# 6.7梯度清零 + 反向传播 + 梯度更新
			optimizer.zero_grad()  # 梯度清零
			loss.backward()  # 反向传播，计算梯度
			optimizer.step()  # 梯度更新
		
		# 6.8把本轮的平均损失值添加到列表中
		loss_list.append(total_loss / total_sample)
		print(f"轮数:{epoch + 1},平均损失值:{total_loss / total_sample}")
	
	# 7.打印最终训练结果
	print(f"{epochs}轮的平均损失分别为:{loss_list}")
	print(f"模型参数,权重:{model.weight},偏置:{model.bias}")
	
	# 8.绘制损失曲线
	#               轮数              每轮的平均损失值
	plt.plot(range(epochs), loss_list)
	plt.title("损失值曲线变化图")
	plt.grid()  # 绘制网格线
	plt.show()
	
	# 9.绘制预测值和真实值的关系
	# 9.1绘制样本点分布情况
	plt.scatter(x, y)
	# 9.2绘制训练模型的预测值
	y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x])
	# 9.3计算真实值
	y_true = torch.tensor(data = [v * coef + 6.6 for v in x])
	# 9.4绘制预测值和真实值的折线图
	plt.plot(x, y_pred, color = "red", label = "预测值")
	plt.plot(x, y_true, color = "blue", label = "真实值")
	# 9.5绘制图例、网格
	plt.legend()
	plt.grid()
	plt.show()








if __name__ == '__main__':
	x, y, coef = create_dataset()
	print(f"x:{x}, y:{y}, coef:{coef}")
	train(x, y, coef)
