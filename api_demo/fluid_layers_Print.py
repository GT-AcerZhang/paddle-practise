import paddle.fluid as fluid

data = fluid.layers.fill_constant(shape=[3,4], value=16, dtype='int64')
data = fluid.layers.Print(data, message="Print data:")

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

ret = exe.run()

