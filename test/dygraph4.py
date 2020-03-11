import paddle.fluid as fluid
import numpy as np

with fluid.dygraph.guard():
	value0 = np.arange(26).reshape(2,13).astype("float32")
	print("value0.shape = %s" % (str(value0.shape)))
	value1 = np.arange(6).reshape(2,3).astype("float32")
	print("value1.shape = %s" % (str(value1.shape)))
	value2 = np.arange(10).reshape(2,5).astype("float32")
	print("value2.shape = %s" % (str(value2.shape)))
	fc = fluid.Linear(13,5,dtype="float32")
	fc2 = fluid.Linear(3, 3, dtype="float32")
	a = fluid.dygraph.to_variable(value0)
	print("a.shape = %s" % (str(a.numpy().shape)))
	b = fluid.dygraph.to_variable(value1)
	print("b.shape = %s" % (str(b.numpy().shape)))
	c = fluid.dygraph.to_variable(value2)
	print("c.shape = %s" % (str(c.numpy().shape)))
	out1 = fc(a)
	print("out1.shape = %s" % (str(out1.numpy().shape)))
	out2 = fc2(b)
	print("out2.shape = %s" % (str(out2.numpy().shape)))
	out1.stop_gradient = True
	out = fluid.layers.concat(input=[out1, out2, c], axis=1)
	print("out.shape = %s" % (str(out.numpy().shape)))
	out.backward()
	assert(fc.weight.gradient() == 0).all()
	assert(out1.gradient() == 0).all()
