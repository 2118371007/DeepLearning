"""

图像分类：
	二值图：    1通道，每个像素点由 0 或者 1组成
	灰度图：    1通道，每个像素点范围  [0,255]
	索引图：    1通道，每个像素点范围  [0,255]，像素点表示颜色表的索引
	RGB真彩图： 3通道，每个像素点范围  [0,255]
	
HWC：高，宽，通道	
常用API:
	imshow()：基于HWC，展示图像
	imread()：读取图像，获取HWC
	imsave()：基于HWC，保存图像
"""
import torch
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 定义函数，绘制一个全黑和全白的图像
def dm01():
	# HWC，高宽全都是200XP，三通道
	# 全0矩阵，表示rgb三通道都是0，表示全黑
	img1 = torch.zeros((200, 200, 3))
	plt.imshow(img1)
	plt.title("全黑")
	plt.show()
	
	img2 = torch.full((200, 200, 3), 255)
	plt.imshow(img2)
	plt.title("全白")
	plt.show()


# 定义函数，读取图片并且另存为
def dm02():
	img = plt.imread("./data/img.jpg")
	plt.imshow(img)
	# 关闭坐标轴
	plt.axis("off")
	plt.show()
	plt.imsave("./data/img_copy.jpg", img)


if __name__ == '__main__':
	dm01()
	dm02()
