"""

二分类任务的损失函数：BCELoss
	公式：
		Loss = -ylog(预测值) - (1 - y)log(1 - 预测值)
	细节：
		公式中没有包含sigmoid激活函数，所以使用BCELoss的时候还要手动指定sigmoid激活函数	
"""

import torch
import torch.nn as nn


def dm01():
	# 1. 手动创建样本的真实值 就是上述公式中的 y
	# 下方真实值的意思是在第一个样本是b类，第二个样本是b类，第三个样本是a类
	y_true = torch.tensor([1, 1, 0], dtype = torch.float)
	
	# 2. 手动创建样本的预测值 就是上述公式中的 概率
	# 二分类中的预测值的意思为1类的概率，即b类的概率
	# 下方预测值的意思是第一个样本是b类的概率为0.837，第二个样本是b类的概率为0.673，第三个样本是b类为0.387(第三个样本为a类的概率大一些)
	y_pred = torch.tensor([0.837, 0.673, 0.387])
	
	# 3. 创建二分类交叉熵损失函数
	criterion = nn.BCELoss()
	
	# 4. 计算损失值
	loss = criterion(y_pred, y_true)
	print(f"loss:{loss}")


if __name__ == '__main__':
	dm01()
