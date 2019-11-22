import paddle
import paddle.fluid as fluid
import numpy as np
import math
import random
import sys

# 训练数据：随机a, [b, c], 要训练y=a+b+c的函数
corpus = []
for i in range(2000):
	a = random.random()*2000
	b = random.random()*2000
	c = random.random()*2000
	corpus.append([a, [b,c], a+b+c])
def reader():
	for a,b,c in corpus:
		yield a, b, c
# 数据读取器
BATCH_SIZE = 20
train_reader = paddle.batch(
	reader, batch_size=BATCH_SIZE)
# 前向传播
x1 = fluid.data(name='x1', shape=[-1, 1], dtype='float32')
x2 = fluid.data(name='x2', shape=[-1, 2], dtype='float32')
y = fluid.data(name='y', shape=[-1, 1], dtype='float32')
y_predict = fluid.layers.fc(input=[x1, x2], size=1, act=None,
	param_attr=[fluid.ParamAttr(name='fc0_w'), fluid.ParamAttr(name='fc1_w')],
	bias_attr=fluid.ParamAttr(name='fc0_b'))

# 代价函数与优化算法
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_loss = fluid.layers.mean(cost)
sgd_optimizer = fluid.optimizer.Adam()
sgd_optimizer.minimize(avg_loss)

# 克隆预测程序
main_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()
test_program = main_program.clone(for_test=True)

# 执行引擎
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(startup_program)

# 喂数据
feeder = fluid.DataFeeder(place=place, feed_list=[x1, x2, y])
# 训练
for pass_id in range(200):
	avg_los_value = None
	for data_train in train_reader():
		avg_los_value = exe.run(
			main_program,
			feed=feeder.feed(data_train),
			fetch_list=[avg_loss])
	print(np.array(avg_los_value)[0][0])

# 显示参数
fc0_w = fluid.global_scope().find_var('fc0_w').get_tensor()
print("fc0_w.shape =", np.array(fc0_w).shape)
print(np.array(fc0_w))
fc1_w = fluid.global_scope().find_var('fc1_w').get_tensor()
print("fc1_w.shape =", np.array(fc1_w).shape)
print(np.array(fc1_w))
fc_b = fluid.global_scope().find_var('fc0_b').get_tensor()
print("fc_b.shape =", np.array(fc_b).shape)
print(np.array(fc_b))
# 测试训练所得函数
y = exe.run(test_program,
	feed=feeder.feed([[100.0, [200.0, 300.0], 600.0]]),
	fetch_list=[y_predict], return_numpy=True)
print("100.0 + 200.0 + 300.0 = %f" % (y[0][0]))
