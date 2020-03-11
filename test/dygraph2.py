import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
	x = fluid.dygraph.to_variable(np.random.randn(5, 5))
	y = fluid.dygraph.to_variable(np.random.randn(5, 5))
	z = fluid.dygraph.to_variable(np.random.randn(5, 5))
	z.stop_gradient = False
	a = x + y
	print(a.stop_gradient)
	b = a + z
	print(b.stop_gradient)
