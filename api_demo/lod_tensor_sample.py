#加载库
import paddle
import paddle.fluid as fluid
import numpy as np
#定义前向计算
x = fluid.layers.data(name='x', shape=[1], dtype='float32', lod_level=1)
y = fluid.layers.data(name='y', shape=[1], dtype='float32', lod_level=2)
out = fluid.layers.sequence_expand(x=x, y=y, ref_level=0)
#定义运算场所
place = fluid.CPUPlace()
#创建执行器
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
#创建LoDTensor
x_d = fluid.create_lod_tensor(np.array([[1.1], [2.2],[3.3],[4.4]]).astype('float32'), [[1,3]], place)
y_d = fluid.create_lod_tensor(np.array([[1.1],[1.1],[1.1],[1.1],[1.1],[1.1]]).astype('float32'), [[1,3], [1,2,1,2]], place)
#开始计算
results = exe.run(fluid.default_main_program(),
	feed={'x':x_d, 'y': y_d },
	fetch_list=[out],return_numpy=False)
#输出执行结果
print("The data of the result: {}.".format(np.array(results[0])))
#输出 result 的序列长度
print("The recursive sequence lengths of the result: {}.".format(results[0].recursive_sequence_lengths()))
#输出 result 的 LoD
print("The LoD of the result: {}.".format(results[0].lod()))
