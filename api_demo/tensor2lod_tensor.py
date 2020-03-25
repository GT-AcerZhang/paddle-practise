import paddle.fluid as fluid
import numpy as np

def to_lodtensor(data, place):
	seq_lens = [len(seq) for seq in data]
	cur_len = 0
	lod = [cur_len]
	for l in seq_lens:
		cur_len += l
		lod.append(cur_len)
	flattened_data = np.concatenate(data, axis=0).astype("float32")
	flattened_data = flattened_data.reshape([len(flattened_data), 1])
	res = fluid.LoDTensor()
	res.set(flattened_data, place)
	res.set_lod([lod])
	return res
new_array = [np.array([[1.1]]).astype('float32'), 
	np.array([[2.2], [3.3], [4.4]]).astype('float32')]
lod_tensor = to_lodtensor(new_array, fluid.CPUPlace())
print("The LoD of the result : {}.".format(lod_tensor.lod()))
print("The array : {}.".format(np.array(lod_tensor)))
