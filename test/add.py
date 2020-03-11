import paddle.fluid as fluid
# 定义变量
a = fluid.data(name='a', shape=[None, 1], dtype='int64')
b = fluid.data(name='b', shape=[None, 1], dtype='int64')
print(a)
print(b)
# 组建网络
result = fluid.layers.elementwise_add(a,b, name='result')
print(result)
# 准备运行网络
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())
# 读取输入数据
import numpy
data_1 = int(input("Please enter an integer: a="))
data_2 = int(input("Please enter an integer: b="))
print("data_1 : %d" % data_1)
print("data_2 : %d" % data_2)
x      = numpy.array([[data_1]])
y      = numpy.array([[data_2]])
print("x = %s, x.shape=%s" % (str(x), str(x.shape)))
print("y = %s, y.shape=%s" % (str(y), str(y.shape)))
#运行网络
outs = exe.run(feed={'a':x, 'b':y}, fetch_list=[a,b,result])
print(a)
print(b)
#输出计算结果
print("%d+%d=%d" % (data_1, data_2, outs[2][0][0]))
