import sys
import numpy as np
from sklearn.externals import joblib
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
def iris_type(s):
	it = {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
	return it[s]
def draw(clf, x):
	iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'
	# 开始画图
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()               #第0列的范围
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()               #第1列的范围
	x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]   #生成网格采样点
	grid_test = np.stack((x1.flat, x2.flat), axis=1)            #stack():沿着新的轴加入一系列数组
	print('grid_test:\n', grid_test)
	# 输出样本到决策面的距离
	z = clf.decision_function(grid_test)
	print('the distance to decision plane:\n', z)

	grid_hat = clf.predict(grid_test)                           # 预测分类值 得到【0,0.。。。2,2,2】
	print('grid_hat:\n', grid_hat)
	grid_hat = grid_hat.reshape(x1.shape)                       # reshape grid_hat和x1形状一致
																#若3*3矩阵e，则e.shape()为3*3,表示3行3列

	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

	plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)             # pcolormesh(x,y,z,cmap)这里参数代入
																# x1，x2，grid_hat，cmap=cm_light绘制的是背景。
	plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark) # 样本点
	plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolor='none', zorder=10)       # 测试点
	plt.xlabel(iris_feature[0], fontsize=20)
	plt.ylabel(iris_feature[1], fontsize=20)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.title('svm in iris data classification', fontsize=30)
	plt.grid()
	plt.show()
try:
	corpus_file = sys.argv[1]
	model_file  = sys.argv[2]
except:
	sys.stderr.write("\tpython "+sys.argv[0]+" corpus_file model_file\n")
	sys.exit(-1)
#读取数据
data = np.loadtxt(corpus_file, dtype=float, delimiter=',', converters={4:iris_type})
#分离特征向量与标签
x, y = np.split(data, (4,), axis=1)
#取前两维特征
x = x[:, 0:2]
#划分训练集与测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(
	x, y, random_state=1, test_size=0.3)
#载入模型
clf = joblib.load(model_file)
#画图
draw(clf, x)
