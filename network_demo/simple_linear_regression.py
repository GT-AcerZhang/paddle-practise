import paddle.fluid as fluid
import numpy as np

train_data = np.array([[1.0], [2.0], [3.0], [4.0]]).astype('float32')
y_true = np.array([[2.0], [4.0], [6.0], [8.0]]).astype('float32')

x = fluid.data(name='x', shape=[None, 1], dtype='float32')
y = fluid.data(name='y', shape=[None, 1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.01)
sgd_optimizer.minimize(avg_cost)

cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

for i in range(200):
	outs = exe.run(feed={'x':train_data, 'y':y_true},
		fetch_list=[y_predict, avg_cost])
print(outs)
