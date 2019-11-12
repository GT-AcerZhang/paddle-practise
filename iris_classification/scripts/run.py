import sys
import numpy as np
from matplotlib import colors
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl
def iris_type(s):
	it = {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
	return it[s]
def classifier():
	clf = svm.SVC(C=0.5, kernel="linear", decision_function_shape="ovr")
	return clf
def train(clf, x_train, y_train):
	clf.fit(x_train, y_train.ravel())
def show_accuracy(a, b, tip):
	acc = a.ravel() == b.ravel()
	print("%s Accuracy:%.3f" % (tip, np.mean(acc)))
def print_accuracy(clf, x_train, y_train, x_test, y_test):
	print("training prediction : %.3f" % (clf.score(x_train, y_train)))
	print("test data prediction: %.3f" % (clf.score(x_test, y_test)))
	show_accuracy(clf.predict(x_train), y_train, "train data")
	show_accuracy(clf.predict(x_test), y_test, "testing data")
	print("decision_function:\n", clf.decision_function(x_train))
def draw(clf, x):
	iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'
	x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
	x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
	x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
	grid_test = np.stack((x1.flat, x2.flat), axis=1)
	print('grid_test:\n', grid_test)
	z = clf.decision_function(grid_test)
	print('the distance to decision plane:\n', z)
	grid_hat = clf.predict(grid_test)
	print('grid_hat:\n', grid_hat)
	grid_hat = grid_hat.reshape(x1.shape)
	cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
	plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
	plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)
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
except:
	sys.stderr.write("python "+sys.argv[0]+" corpus_file\n")
	sys.exit(-1)
# step 1 : 数据预处理
data = np.loadtxt(corpus_file, 
		dtype=float, 
		delimiter=",",
		converters={4:iris_type})
#print(data.shape)
#print(data[0:20, :])
#sys.exit(0)
x, y = np.split(data, (4,), axis=1)
x = x[:, 0:2]
#print(x.shape)
#print(y.shape)
x_train, x_test, y_train, y_test = model_selection.train_test_split(
		x, y, random_state=1, test_size=0.3)

# step 2 : 定义模型
clf = classifier()
# step 3 : 训练模型
train(clf, x_train, y_train)
# step 4 : 评估模型
print_accuracy(clf, x_train, y_train, x_test, y_test)
# step 5 : 模型使用
draw(clf, x)
