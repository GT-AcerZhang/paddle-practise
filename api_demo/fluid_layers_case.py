import paddle.fluid as fluid
import paddle.fluid.layers as layers
def fn_1():
	return layers.fill_constant(shape=[1,2], dtype='float32', value=1)
def fn_2():
	return layers.fill_constant(shape=[2,2], dtype='int32', value=2)
def fn_3():
	return layers.fill_constant(shape=[3], dtype='int32', value=3)
main_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()
with fluid.program_guard(main_program, startup_program):
	x = layers.fill_constant(shape=[1], dtype='float32', value=0.3)
	y = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
	z = layers.fill_constant(shape=[1], dtype='float32', value=0.2)
	pred_1 = layers.less_than(z, x)
	pred_2 = layers.less_than(x, y)
	pred_3 = layers.equal(x, y)
	out_1 = layers.case(pred_fn_pairs=[(pred_1, fn_1), (pred_2, fn_2)], default=fn_3)
	out_2 = layers.case(pred_fn_pairs=[(pred_2, fn_2), (pred_3, fn_3)])
	exe = fluid.Executor(fluid.CPUPlace())
	res_1, res_2 = exe.run(main_program, fetch_list=[out_1, out_2])
	print(res_1)
	print(res_2)
