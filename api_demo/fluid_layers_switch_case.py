import paddle.fluid as fluid
import paddle.fluid.layers as layers

def fn_1():
	return layers.fill_constant(shape=[1,2], dtype='float32', value=1)
def fn_2():
	return layers.fill_constant(shape=[2,2], dtype='int32', value=2)
def fn_3():
	return layers.fill_constant(shape=[3], dtype='int32', value=3)
main_program = fluid.default_main_program()
startup_program = fluid.default_main_program()
with fluid.program_guard(main_program, startup_program):
	index_1 = layers.fill_constant(shape=[1], dtype='int32', value=1)
	index_2 = layers.fill_constant(shape=[1], dtype='int32', value=2)
	out_1 = layers.switch_case(branch_index=index_1,
		branch_fns={1:fn_1, 2:fn_2},
		default=fn_3)
	out_2 = layers.switch_case(branch_index=index_2,
		branch_fns=[(1, fn_1), (2, fn_2)],
		default=fn_3)
	out_3 = layers.switch_case(branch_index=index_2,
		branch_fns=[(0, fn_1), (4, fn_2), (7, fn_3)])
	exe = fluid.Executor(fluid.CPUPlace())
	res_1, res_2, res_3 = exe.run(main_program, fetch_list=[out_1, out_2, out_3])
	print(res_1)
	print(res_2)
	print(res_3)
