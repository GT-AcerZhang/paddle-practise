import paddle.fluid as fluid
import numpy as np

class MyLayer(fluid.dygraph.Layer):
	def __init__(self, input_size):
		super(MyLayer, self).__init__()
		self.linear = fluid.dygraph.nn.Linear(input_size, 12)
	def forward(self, inputs):
		print("input = %s" % (str(inputs.numpy())))
		x = self.linear(inputs)
		print("after linear : x = %s" % (str(x.numpy())))
		x = fluid.layers.relu(x)
		print("after relu : x = %s" % (str(x.numpy())))
		self._x_for_debug = x
		x = fluid.layers.elementwise_mul(x, x)
		print("after elementwise mul : x = %s" % (str(x.numpy())))
		x = fluid.layers.reduce_sum(x)
		print("after reduce_sum : x = %s" % (str(x.numpy())))
		return [x]
if __name__ == "__main__":
	np_inp = np.array([[1.0, 2.0, -1.0]], dtype=np.float32)
	print("np_inp.shape=%s" % (str(np_inp.shape)))
	print("np_inp.shape[-1]=%d" % (np_inp.shape[-1]))
	with fluid.dygraph.guard():
		var_inp = fluid.dygraph.to_variable(np_inp)
		my_layer = MyLayer(np_inp.shape[-1])
		x = my_layer(var_inp)[0]
		dy_out = x.numpy()
		print("dy_out = %s" %(str(dy_out)))
		x.backward()
		dy_grad = my_layer._x_for_debug.gradient()
		my_layer.clear_gradients()
