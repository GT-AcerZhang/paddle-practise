import sys
import time
import paddle
import paddle.fluid as fluid
import numpy as np
from MNIST import *

def checkData():
	print("checkData......................................")
	with fluid.dygraph.guard():
		mnist = MNIST()
		train_reader = paddle.batch(
			paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
		_, data = list(enumerate(train_reader()))[0]
		dy_x_data = np.array(
			[x[0].reshape(1, 28, 28) for x in data]).astype("float32")
		img = fluid.dygraph.to_variable(dy_x_data)
		#print("img = %s" % (str(img.numpy())))
		print("img.shape = %s" % (str(img.numpy().shape)))
		print("MNIST(img).shape = %s" % (str(mnist(img).numpy().shape)))
def train1():
	print("train1........................................")
	print("\t训练代码demo")
	with fluid.dygraph.guard():
		epoch_num = 5
		BATCH_SIZE = 64
		train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
		mnist = MNIST()
		adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
		for epoch in range(epoch_num):
			for batch_id, data in enumerate(train_reader()):
				dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
				y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)
				img = fluid.dygraph.to_variable(dy_x_data)
				label = fluid.dygraph.to_variable(y_data)

				cost = mnist(img)

				loss = fluid.layers.cross_entropy(cost, label)
				avg_loss = fluid.layers.mean(loss)
				
				if batch_id%100 == 0 and batch_id is not 0:
					print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
				avg_loss.backward()
				adam.minimize(avg_loss)
				mnist.clear_gradients()
def train2():
	print("train2........................................")
	print("\t获取神经网络中的参数")
	with fluid.dygraph.guard():
		epoch_num = 5
		BATCH_SIZE = 64

		mnist = MNIST()
		adam = fluid.optimizer.AdamOptimizer(learning_rate=0.0001, parameter_list=mnist.parameters())
		train_reader = paddle.batch(
			paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)

		np.set_printoptions(precision=3, suppress=True)
		for epoch in range(epoch_num):
			for batch_id, data in enumerate(train_reader()):
				dy_x_data = np.array(
					[x[0].reshape(1, 28, 28) for x in data]).astype("float32")
				y_data = np.array(
					[x[1] for x in data]).astype("int64").reshape(BATCH_SIZE, 1)
				img = fluid.dygraph.to_variable(dy_x_data)
				label = fluid.dygraph.to_variable(y_data)
				label.stop_gradient = True

				cost = mnist(img)
				loss = fluid.layers.cross_entropy(cost, label)
				avg_loss = fluid.layers.mean(loss)

				dy_out = avg_loss.numpy()

				avg_loss.backward()
				adam.minimize(avg_loss)
				mnist.clear_gradients()

				dy_param_value = {}
				for param in mnist.parameters():
					print("%s = %s" % (param.name, str(param.numpy())))
					dy_param_value[param.name] = param.numpy()

				if batch_id%20 == 0:
					print("loss at step {}: {}".format(batch_id, avg_loss.numpy()))
				break
			break
		print("Final loss : {}".format(avg_loss.numpy()))
		#print("_simple_img_conv_pool_1_conv2d W's mean is: {}".format(mnist._simple_img_conv_pool_1._conv2d._filter_param.numpy().mean()))
		#print("_simple_img_conv_pool_1_conv2d Bias's mean is: {}".format(mnist._simple_img_conv_pool_1._conv2d._bias_param.numpy().mean()))
def train3(use_cudnn, model_file):
	print("train3........................................")
	print("\t多卡训练（paddle有bug，没有调试通）")
	place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
	with fluid.dygraph.guard(place):
		strategy = fluid.dygraph.parallel.prepare_context()
		epoch_num = 5
		BATCH_SIZE = 64
		mnist = MNIST(use_cudnn=use_cudnn)
		adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
		mnist = fluid.dygraph.parallel.DataParallel(mnist, strategy)

		train_reader = paddle.batch(
			paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
		train_reader = fluid.contrib.reader.distributed_batch_reader(train_reader)

		start_time = time.time()
		for epoch in range(epoch_num):
			for batch_id, data in enumerate(train_reader()):
				dy_x_data = np.array(
					[x[0].reshape(1, 28, 28) for x in data]).astype('float32')
				y_data = np.array(
					[x[1] for x in data]).astype('int64').reshape(-1, 1)
				img = fluid.dygraph.to_variable(dy_x_data)
				label = fluid.dygraph.to_variable(y_data)
				label.stop_gradient = True

				cost, acc = mnist(img, label)

				loss = fluid.layers.cross_entropy(cost, label)
				avg_loss = fluid.layers.mean(loss)

				avg_loss = mnist.scale_loss(avg_loss)
				avg_loss.backward()
				mnist.apply_collective_grads()

				adam.minimize(avg_loss)
				mnist.clear_gradients()
				if batch_id%100 == 0 and batch_id is not 0:
					print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
		fluid.dygraph.save_dygraph(mnist.state_dict(), model_file)
		end_time = time.time()
		print("training model has finished! time=%.2fs" % (end_time - start_time))
def test(reader, model, batch_size):
	acc_set = []
	avg_loss_set = []
	for batch_id, data in enumerate(reader()):
		dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
		y_data = np.array([x[1] for x in data]).astype('int64').reshape(batch_size, 1)
		img = fluid.dygraph.to_variable(dy_x_data)
		label = fluid.dygraph.to_variable(y_data)
		label.stop_gradient = True

		prediction, acc = model(img, label)
		
		loss = fluid.layers.cross_entropy(input=prediction, label=label)
		avg_loss = fluid.layers.mean(loss)
		avg_loss_set.append(float(avg_loss.numpy()))
		acc_set.append(float(acc.numpy()))
	acc_val_mean = np.array(acc_set).mean()
	avg_loss_val_mean = np.array(avg_loss_set).mean()

	return avg_loss_val_mean, acc_val_mean
def train4(use_cudnn, model_file):
	with fluid.dygraph.guard():
		epoch_num = 5
		BATCH_SIZE = 64

		mnist = MNIST(use_cudnn=use_cudnn)
		adam = fluid.optimizer.Adam(learning_rate=0.001, parameter_list=mnist.parameters())
		train_reader = paddle.batch(
			paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
		test_reader = paddle.batch(
			paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)

		np.set_printoptions(precision=3, suppress=True)
		dy_param_init_value = {}
		start_time = time.time()
		for epoch in range(epoch_num):
			for batch_id, data in enumerate(train_reader()):
				# step 1 : 处理输入
				dy_x_data = np.array(
					[x[0].reshape(1, 28, 28) for x in data]).astype('float32')
				y_data = np.array(
					[x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)
				img = fluid.dygraph.to_variable(dy_x_data)
				label = fluid.dygraph.to_variable(y_data)
				label.stop_gradient = True

				# step 2 : 前向传播&&损失函数
				cost = mnist(img)
				loss = fluid.layers.cross_entropy(cost, label)
				avg_loss = fluid.layers.mean(loss)
				dy_out = avg_loss.numpy()

				# step 3 : 反向传播&&最优化
				avg_loss.backward()
				adam.minimize(avg_loss)
				mnist.clear_gradients()
				# step 4 : 测试模型
				if batch_id%100 == 0 and batch_id is not 0:
					mnist.eval()
					test_cost, test_acc = test(test_reader, mnist, BATCH_SIZE)
					mnist.train()
					print("epoch {}, batch_id {}, train loss is {}, test cost is {}, test acc is {}".format(
						epoch, batch_id, avg_loss.numpy(), test_cost, test_acc))
		fluid.dygraph.save_dygraph(mnist.state_dict(), model_file)
		end_time = time.time()
		print("training model has finished! time=%.2fs" % (end_time - start_time))
if __name__ == "__main__":
	try:
		ctype      = sys.argv[1]
		model_file = sys.argv[2]
	except:
		sys.stderr.write("\tpython "+sys.argv[0]+" cpu|onegpu|multigpu model_file\n")
		sys.exit(-1)
	#checkData()
	#train1()
	#train2()
	#train3(True, model_file)
	if ctype == "cpu":
		train4(False, model_file)
	elif ctype == "onegpu":
		train4(True, model_file)
	elif ctype == "multigpu":
		train3(True, model_file)
	else:
		sys.stderr.write("\tpython "+sys.argv[0]+" cpu|onegpu|twogpu model_file\n")
		sys.exit(-1)
