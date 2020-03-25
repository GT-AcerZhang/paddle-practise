import paddle.fluid as fluid 
import numpy as np

a = fluid.create_lod_tensor(
	np.array([
		[1],[1],[1],
		[1],[1],
		[1],[1],[1],[1],
		[1],
		[1],[1],
		[1],[1],[1]]).astype('int64'),
	[[3,1,2], [3,2,4,1,2,3]], fluid.CPUPlace())
print(a.recursive_sequence_lengths()) # 子张量长度
print(sum(a.recursive_sequence_lengths()[-1]))
print("--------------------------------------------")

def LodTensor_to_Tensor(lod_tensor):
	lod = lod_tensor.lod()
	print(lod)
	array = np.array(lod_tensor)
	print(array)
	new_array = []
	for i in range(len(lod[0])-1):
		new_array.append(array[lod[0][i]:lod[0][i+1]])
	return new_array
a = fluid.create_lod_tensor(np.array([[1.1], [2.2], [3.3], [4.4]]).astype('float32'), [[1,3]], fluid.CPUPlace())
new_array = LodTensor_to_Tensor(a)
print(new_array)
print("--------------------------------------------")

def to_lodtensor(data, place):
	seq_lens = [len(seq) for seq in data]
	cur_len = 0
	lod = [cur_len]
	for l in seq_lens:
		cur_len += l
		lod.append(cur_len)
	flattened_data = np.concatenate(data, axis=0).astype("float32")
	print("data : {}".format(data))
	print("flattened_data : {}".format(flattened_data))
	print("flattened_data.shape : {}".format(flattened_data.shape))
	flattened_data = flattened_data.reshape([len(flattened_data), 1])
	print("flattened_data : {}".format(flattened_data))
	print("flattened_data.shape : {}".format(flattened_data.shape))
	res = fluid.LoDTensor()
	res.set(flattened_data, place)
	res.set_lod([lod])
	return res
lod_tensor = to_lodtensor(new_array, fluid.CPUPlace())
print("The LoD of the result: {}.".format(lod_tensor.lod()))
print("The array : {}.".format(np.array(lod_tensor)))
