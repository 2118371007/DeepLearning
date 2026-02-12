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
        random_state = 6  # 随机种子
    )
    # make_regression生成的数据是nd数组，必须转换成tensor才能方便后面的操作
    x = torch.tensor(x, dtype = torch.float)
    y = torch.tensor(y, dtype = torch.float)
    
    return x, y, coef


if __name__ == '__main__':
    x, y, coef = create_dataset()
    print(f"x:{x}, y:{y}, coef:{coef}")
