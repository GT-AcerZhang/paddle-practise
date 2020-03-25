import paddle.fluid as fluid
import numpy as np

def LodTensor_to_Tensor(lod_tensor):
	lod = lod_tensor.lod()
	print("lod :", lod)
	array = np.array(lod_tensor)
	print("array :", array)
	new_array = []
	for i in range(len(lod[0])-1):
		new_array.append(array[lod[0][i]:lod[0][i+1]])
	return new_array
a = fluid.create_lod_tensor(np.array([[1.1], [2.2],[3.3],[4.4]]).astype('float32'),
	[[1, 3]], fluid.CPUPlace())
new_array = LodTensor_to_Tensor(a)
print(new_array)
