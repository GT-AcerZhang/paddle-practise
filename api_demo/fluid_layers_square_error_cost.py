import paddle.fluid as fluid
import numpy as np

with fluid.program_guard(fluid.default_main_program(), fluid.default_startup_program()):
	x = fluid.layers.data(name='x', shape=[None, 10, 2])
	feed_x = np.ones(shape=[3, 10, 2], dtype=np.float32)
	y = fluid.layers.data(name='y', shape=[None, 10, 2])
	feed_y = np.ones(shape=[3, 10, 2], dtype=np.float32)*3
	cost = fluid.layers.square_error_cost(input=x, label=y)
	avg_cost = fluid.layers.mean(cost)
	exe = fluid.Executor(fluid.CPUPlace())
	#exe.run(fluid.default_startup_program())
	out1, out2 = exe.run(fluid.default_main_program(),
		feed={"x":feed_x, "y":feed_y}, fetch_list=[cost.name, avg_cost.name])
	print("cost = {}".format(out1))
	print("avg_cost = {}".format(out2))
