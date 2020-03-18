import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.executor import Executor
from paddle.fluid.framework import Program, program_guard
# 疑问：如果true_func or false_func执行时需要输入参数，怎么办？
def true_func():
	return layers.fill_constant(shape=[1,2], dtype='int32', value=1), \
		layers.fill_constant(shape=[2,3], dtype='bool', value=True)
def false_func():
	return layers.fill_constant(shape=[3,4], dtype='float32', value=3), \
		layers.fill_constant(shape=[4,5], dtype='int64', value=2)
main_program = Program()
startup_program = Program()
with program_guard(main_program, startup_program):
	x = layers.fill_constant(shape=[1], dtype='float32', value=0.1)
	y = layers.fill_constant(shape=[1], dtype='float32', value=0.23)
	pred = layers.less_than(x, y)
	out = layers.cond(pred, true_func, false_func)

	place = fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
	exe = fluid.Executor(place)
	ret = exe.run(main_program, fetch_list=out)
	print(ret)
