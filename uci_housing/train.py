from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy
import math
import sys
# 获取训练过程中测试误差
def train_test(executor, program, reader, feeder, fetch_list):
	accumulated = 1*[0]
	count = 0
	for data_test in reader():
		outs = executor.run(program=program,
			feed=feeder.feed(data_test),
			fetch_list=fetch_list)
		accumulated = [x_c[0]+x_c[1][0] for x_c in zip(accumulated, outs)]
		count += 1
	return [x_d/count for x_d in accumulated]

BATCH_SIZE = 20
train_reader = paddle.batch(
	paddle.reader.shuffle(
		paddle.dataset.uci_housing.train(), buf_size=500),
		batch_size=BATCH_SIZE)
test_reader = paddle.batch(
	paddle.reader.shuffle(
		paddle.dataset.uci_housing.test(), buf_size=500),
		batch_size=BATCH_SIZE)
# 神经网络定义
x = fluid.data(name='x', shape=[None, 13], dtype='float32')
y = fluid.data(name='y', shape=[None, 1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_loss = fluid.layers.mean(cost)

# 程序配置
main_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()
test_program = main_program.clone(for_test=True)

# 优化器配置
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

# 运算场所配置
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

num_epochs = 100
params_dirname = "fit_a_line.inference.model"
feeder = fluid.DataFeeder(place=place, feed_list=[x,y])
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"
#plot_prompt = Ploter(train_prompt, test_prompt)
step = 0
exe_test = fluid.Executor(place)

for pass_id in range(num_epochs):
	for data_train in train_reader():
		avg_loss_value = exe.run(main_program,
			feed=feeder.feed(data_train),
			fetch_list=[avg_loss])
		if step%10 == 0:
			#plot_prompt.append(train_prompt, step, avg_loss_value[0])
			#plot_prompt.plot()
			print("%s, Step %d, Cost %f" % (train_prompt, step, avg_loss_value[0]))
		if step%100 == 0:
			test_metics = train_test(executor=exe_test,
				program=test_program,
				reader=test_reader,
				fetch_list=[avg_loss.name],
				feeder=feeder)
			#plot_prompt.append(test_prompt, step, test_metics[0])
			#plot_prompt.plot()
			print("%s, Step %d, Cost %f" % (
				test_prompt, step, test_metics[0]))
			if test_metics[0] < 10.0:
				break
		step += 1
		if math.isnan(float(avg_loss_value[0])):
			sys.exit("got NaN loss, training failed.")
if params_dirname is not None:
	fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)
