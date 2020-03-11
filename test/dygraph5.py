import paddle
import paddle.fluid as fluid
import numpy as np

class SimpleImgConvPool(fluid.dygraph.Layer):
	def __init__(self,
			num_channels,
			num_filters,
			filter_size,
			pool_size,
			pool_stride,
			pool_padding=0,
			pool_type='max',
			global_pooling=False,
			conv_stride=1,
			conv_padding=0,
			conv_dilation=1,
			conv_groups=1,
			act=None,
			use_cudnn=False,
			param_attr=None,
			bias_attr=None):
		super(SimpleImgConvPool, self).__init__()
		self._conv2d = fluid.dygraph.Conv2D(
			num_channels=num_channels,
			num_filters=num_filters,
			filter_size=filter_size,
			stride=conv_stride,
			padding=conv_padding,
			dilation=conv_dilation,
			groups=conv_groups,
			param_attr=param_attr,
			bias_attr=bias_attr,
			act=act,
			use_cudnn=use_cudnn)
		self._pool2d = fluid.dygraph.Pool2D(
			pool_size=pool_size,
			pool_type=pool_type,
			pool_stride=pool_stride,
			pool_padding=pool_padding,
			global_pooling=global_pooling,
			use_cudnn=use_cudnn)
	def forward(self, inputs):
		x = self._conv2d(inputs)
		x = self._pool2d(x)
		return x
class MNIST(fluid.dygraph.Layer):
	def __init__(self):
		super(MNIST, self).__init__()
		self._simple_img_conv_pool_1 = SimpleImgConvPool(
			1, 20, 5, 2, 2, act="relu")
		self._simple_img_conv_pool_2 = SimpleImgConvPool(
			20, 50, 5, 2, 2, act="relu")
		self.pool_2_shape = 50*4*4
		SIZE = 10
		scale = (2.0/(self.pool_2_shape**2 * SIZE))**0.5
		self._fc = fluid.dygraph.Linear(
			self.pool_2_shape,
			10,
			param_attr=fluid.param_attr.ParamAttr(
				initializer=fluid.initializer.NormalInitializer(
					loc=0.0, scale=scale)),
			act="softmax"
		)
	def forward(self, inputs, label=None):
		x = self._simple_img_conv_pool_1(inputs)
		x = self._simple_img_conv_pool_2(x)
		x = fluid.layers.reshape(x, shape=[-1, self.pool_2_shape])
		x = self._fc(x)
		if label is not None:
			acc = fluid.layers.accuracy(input=x, label=label)
			return x, acc
		else:
			return x
if __name__ == "__main__":
	# 未训练的情况下，直接打印前向传播的计算结果
	'''
	with fluid.dygraph.guard():
		mnist = MNIST()
		train_reader = paddle.batch(
			paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
		_, data = list(enumerate(train_reader()))[0]
		dy_x_data = np.array(
			[x[0].reshape(1, 28, 28) for x in data]).astype('float32')
		img = fluid.dygraph.to_variable(dy_x_data)
		result = mnist(img).numpy()
		print("result is: {}".format(result))
		print("result.shape is: {}".format(result.shape))
		print("dy_x_data.shape=%s" % (str(dy_x_data.shape)))
	'''
	# 训练
	'''
	with fluid.dygraph.guard():
		epoch_num = 5
		BATCH_SIZE = 64
		train_reader = paddle.batch(
			paddle.dataset.mnist.train(), batch_size=32, drop_last=True)
		mnist = MNIST()
		adam = fluid.optimizer.AdamOptimizer(learning_rate=0.001, parameter_list=mnist.parameters())
		for epoch in range(epoch_num):
			for batch_id, data in enumerate(train_reader()):
				dy_x_data = np.array([x[0].reshape(1,28,28) for x in data]).astype('float32')
				y_data = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

				img = fluid.dygraph.to_variable(dy_x_data)
				label = fluid.dygraph.to_variable(y_data)
				
				cost  = mnist(img)
				loss = fluid.layers.cross_entropy(cost, label)
				avg_loss = fluid.layers.mean(loss)
				if batch_id%100 == 0 and batch_id is not 0:
					print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
				avg_loss.backward()
				adam.minimize(avg_loss)
				mnist.clear_gradients()
	'''
	# 训练的同时，获取参数
	'''
	with fluid.dygraph.guard():
		epoch_num = 5
		BATCH_SIZE = 64

		mnist = MNIST()
		adam = fluid.optimizer.AdamOptimizer(
			learning_rate=0.01, 
			parameter_list=mnist.parameters())
		train_reader = paddle.batch(
			paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
		np.set_printoptions(precision=3, suppress=True)
		for epoch in range(epoch_num):
			for batch_id, data in enumerate(train_reader()):
				dy_x_data = np.array([x[0].reshape(1,28,28) for x in data]).astype('float32')
				y_data = np.array([x[1] for x in data]).astype("int64").reshape(BATCH_SIZE, 1)

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
					dy_param_value[param.name] = param.numpy()
					print(param.name)
				if batch_id%20 == 0:
					print("loss at step {}: {}".format(batch_id, avg_loss.numpy()))
				break
			break
		print("Final loss: {}".format(avg_loss.numpy()))
		#print("_simple_img_conv_pool_1_conv2d W's mean is: {}".format(mnist._simple_img_conv_pool_1._conv2d._filter_param.numpy().mean()))
		#print("_simple_img_conv_pool_1_conv2d Bias's mean is: {}".format(mnist._simple_img_conv_pool_1._conv2d._bias_param.numpy().mean()))
	'''
	# 保存模型
	#'''
	with fluid.dygraph.guard():
		epoch_num = 5
		BATCH_SIZE = 64
		mnist = MNIST()
		adam = fluid.optimizer.Adam(learning_rate=0.001, parameter_list=mnist.parameters())
		train_reader = paddle.batch(
			paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
		
		np.set_printoptions(precision=3, suppress=True)
		dy_param_init_value={}
		for epoch in range(epoch_num):
			for batch_id, data in enumerate(train_reader()):
				dy_x_data = np.array([x[0].reshape(1,28,28) for x in data]).astype('float32')
				y_data = np.array([x[1] for x in data]).astype('int64').reshape(BATCH_SIZE, 1)

				img = fluid.dygraph.to_variable(dy_x_data)
				label = fluid.dygraph.to_variable(y_data)
				label.stop_gradient = True

				cost = mnist(img)
				loss = fluid.layers.cross_entropy(cost, label)
				avg_loss = fluid.layers.mean(loss)

				dy_out = avg_loss.numpy()

				avg_loss.backward()
				adam.minimize(avg_loss)
				if batch_id == 20:
					fluid.dygraph.save_dygraph(mnist.state_dict(), "paddle_dy")
				mnist.clear_gradients()
				if batch_id == 20:
					for param in mnist.parameters():
						dy_param_init_value[param.name] = param.numpy()
					model, _ = fluid.dygraph.load_dygraph("paddle_dy")
					mnist.set_dict(model)
					break
			if epoch == 0:
				break
		restore = mnist.parameters()

		success = True
		for value in restore:
			if (not np.array_equal(value.numpy(), dy_param_init_value[value.name])) or (not np.isfinite(value.numpy().all())) or (np.isnan(value.numpy().any())):

				success = False
		print("model save and load success? {}".format(success))
	#'''
