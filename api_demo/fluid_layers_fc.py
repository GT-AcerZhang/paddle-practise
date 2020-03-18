import paddle.fluid as fluid
import numpy as np

# 单张量
data = fluid.layers.data(name="data", shape=[None, 32], dtype="float32")
feed_data = np.ones(shape=[3,32], dtype=np.float32)
fc1 = fluid.layers.fc(input=data, size=2, act="tanh")

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
out1 = exe.run(fluid.default_main_program(),
	feed={"data":feed_data}, fetch_list=[fc1.name])
print(out1)
# 多张量
'''
data_1 = fluid.layers.data(name="data_1", shape=[None, 32], dtype="float32")
feed_data_1 = np.ones(shape=[3,32], dtype=np.float32)
data_2 = fluid.layers.data(name="data_2", shape=[None, 36], dtype="float32")
feed_data_2 = np.ones(shape=[3,36], dtype=np.float32)
fc2 = fluid.layers.fc(input=[data_1, data_2], size=2, act="tanh")

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())
out2 = exe.run(fluid.default_main_program(),
	feed={"data_1":feed_data_1, "data_2":feed_data_2},
	fetch_list=[fc2.name])
print(out2)
'''
