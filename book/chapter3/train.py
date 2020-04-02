import sys
import numpy as np
from matplotlib import colors
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.externals import joblib

def iris_type(s):
	it = {b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2}
	return it[s]
def classifier():
	#clf = svm.SVC(C=0.8, kernel='rbf', gamma=50, decision_function_shape='ovr')
	clf = svm.SVC(C=0.5, kernel='linear', decision_function_shape='ovr')
	return clf
def train(clf, x_train, y_train):
	clf.fit(x_train, y_train.ravel())
def show_accuracy(a, b, tip):
	acc = (a.ravel() == b.ravel())
	print("%s Accuracy:%.3f" % (tip, np.mean(acc)))
def print_accuracy(clf, x_train, y_train, x_test, y_test):
	print("training prediction:%.3f" % (clf.score(x_train, y_train)))
	print("test data prediction:%.3f" % (clf.score(x_test, y_test)))
	show_accuracy(clf.predict(x_train), y_train, 'training data')
	show_accuracy(clf.predict(x_test), y_test, "testing data")
	print("decision_function:\n", clf.decision_function(x_train))
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
#训练
clf = classifier()
train(clf, x_train, y_train)
#评估
print_accuracy(clf, x_train, y_train, x_test, y_test)
#保存
joblib.dump(clf, model_file)
