import paddle.fluid as fluid
import numpy as np

print("step 1------------------------------")
# step 1 : Variable与numpy.array的转换
x = np.ones([2,2], np.float32)
with fluid.dygraph.guard():
	inputs = []
	for _ in range(10):
		inputs.append(fluid.dygraph.to_variable(x))
	ret = fluid.layers.sums(inputs)
	print("ret = %s" % (str(ret.numpy())))
	print("ret.shape = %s" % (str(ret.numpy().shape)))
	ret.stop_gradient = False
	loss = fluid.layers.reduce_sum(ret)
	print("loss = %s" % (str(loss.numpy())))
	loss.backward()
	print("loss.gradient = %s" % (str(loss.gradient())))
# step 2 : 面向对象的神经网络
print("step 2------------------------------")
class MyLayer(fluid.dygraph.Layer):
	def __init__(self, input_size):
		super(MyLayer, self).__init__()
		self.linear = fluid.dygraph.nn.Linear(input_size, 12)
	def forward(self, inputs):
		print("inputs = %s" % (str(inputs.numpy())))
		print("inputs.shape = %s" % (str(inputs.numpy().shape)))
		x = self.linear(inputs)
		print("after linear = %s" % (str(x.numpy())))
		print("after linear.shape = %s" % (str(x.numpy().shape)))
		x = fluid.layers.relu(x)
		print("after relu = %s" % (str(x.numpy())))
		print("after relu.shape = %s" % (str(x.numpy().shape)))
		self._x_for_debug = x
		x = fluid.layers.elementwise_mul(x, x)
		print("after elementwise_mul = %s" % (str(x.numpy())))
		print("after elementwise_mul.shape = %s" % (str(x.numpy().shape)))
		x = fluid.layers.reduce_sum(x)
		print("after reduce_sum = %s" % (str(x.numpy())))
		print("after reduce_sum.shape = %s" % (str(x.numpy().shape)))
		return [x]
np_inp = np.array([[1.0, 2.0, -1.0]], dtype=np.float32)
with fluid.dygraph.guard():
	var_inp = fluid.dygraph.to_variable(np_inp)
	my_layer = MyLayer(np_inp.shape[-1])
	x = my_layer(var_inp)[0]
	dy_out = x.numpy()
	x.backward()
	dy_grad = my_layer._x_for_debug.gradient()
	my_layer.clear_gradients()
# step 3 : 自动减枝1
print("step 3------------------------------")
with fluid.dygraph.guard():
	x = fluid.dygraph.to_variable(np.random.randn(5, 5))
	y = fluid.dygraph.to_variable(np.random.randn(5, 5))
	z = fluid.dygraph.to_variable(np.random.randn(5, 5))
	z.stop_gradient = False
	a = x + y
	print("a.stop_gradient=%s" % (str(a.stop_gradient)))
	b = a + z
	print("b.stop_gradient=%s" % (str(b.stop_gradient)))
# step 4 : 自动减枝2
print("step 4------------------------------")
with fluid.dygraph.guard():
	value0 = np.arange(26).reshape(2,13).astype("float32")
	value1 = np.arange(6).reshape(2,3).astype("float32")
	value2 = np.arange(10).reshape(2,5).astype("float32")
	fc = fluid.Linear(13, 5, dtype="float32")
	fc2 = fluid.Linear(3, 3, dtype="float32")
	a = fluid.dygraph.to_variable(value0)
	b = fluid.dygraph.to_variable(value1)
	c = fluid.dygraph.to_variable(value2)
	out1 = fc(a)
	out2 = fc2(b)
	out1.stop_gradient = True
	out = fluid.layers.concat(input=[out1, out2, c], axis=1)
	print("out = %s" % (str(out.numpy())))
	out.backward()
	print("out = %s" % (str(out.numpy())))
	print("out.gradient = %s" % (str(out.gradient())))
	print("out.shape = %s" % (str(out.numpy().shape)))
	assert(fc.weight.gradient() == 0).all()
	assert(out1.gradient() == 0).all()
