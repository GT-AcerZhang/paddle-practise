import paddle.fluid as fluid
import paddle.fluid.layers as layers

def cond(i, ten):
	return i < ten
def body(i, dummy):
	i = i+1
	return i, dummy
i = layers.fill_constant(shape=[1], dtype='int64', value=0)
ten = layers.fill_constant(shape=[1], dtype='int64', value=10)
out,ten=layers.while_loop(cond=cond, body=body, loop_vars=[i, ten])

exe = fluid.Executor(fluid.CPUPlace())
res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[out, i])
print(res)
