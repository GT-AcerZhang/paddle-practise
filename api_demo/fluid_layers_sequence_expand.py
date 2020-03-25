import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
    
x = fluid.data(name='x', shape=[4, 1], dtype='float32')
y = fluid.data(name='y', shape=[8, 1],dtype='float32', lod_level=1)
out = layers.sequence_expand(x=x, y=y, ref_level=0)
    
exe = fluid.Executor(fluid.CPUPlace())
place = fluid.CPUPlace()
    
np_data = np.array([[9], [10], [11], [12]]).astype('float32')
x_lod_tensor = fluid.create_lod_tensor(np_data, [[3, 1]], place)
print(x_lod_tensor)

np_data = np.array([[1], [2], [3], [4], [5], [6], [7], [8]]).astype('float32')
y_lod_tensor = fluid.create_lod_tensor(np_data, [[2, 2], [3,3,1,1]], place)
print(y_lod_tensor)

out_main = exe.run(fluid.default_main_program(),feed={'x': x_lod_tensor, 'y': y_lod_tensor},fetch_list=[out], return_numpy=False)
print(out_main[0])
