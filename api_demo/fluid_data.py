import paddle.fluid as fluid
import numpy as np

x = fluid.data(name='x', shape=[3,2,1], dtype='float32')
y = fluid.data(name='y', shape=[-1,2,1], dtype='float32')
z = x + y

feed_data = np.ones(shape=[3,2,1], dtype=np.float32)
exe = fluid.Executor(fluid.CPUPlace())
out = exe.run(fluid.default_main_program(),
	feed={'x':feed_data, 'y':feed_data},
	fetch_list=[z.name])
print(out)
print(type(out))
