import paddle
import paddle.fluid as fluid
import numpy as np
import math
import random
import sys
# 测试数据：随机a, [b, c], 要测试y=a+b+c的函数
corpus = []
for i in range(2000):
	a = random.random()*2000
	b = random.random()*2000
	c = random.random()*2000
	corpus.append([a, [b,c]])
def reader():
	for a,b in corpus:
		yield a, b
try:
	model_dir = sys.argv[1]
except:
	sys.stderr.write("\tpython "+sys.argv[0]+" model_dir\n")
	sys.exit(-1)
# load模型
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
infer_program, feed_names, fetch_names = fluid.io.load_inference_model(model_dir, exe)
print(feed_names)
# 准备数据
x3 = fluid.data(name='x3', shape=[-1, 1], dtype='float32')
x2 = fluid.data(name='x2', shape=[-1, 2], dtype='float32')
feeder = fluid.DataFeeder(place=place, feed_list=feed_names)
test_reader = paddle.batch(reader, batch_size=20)
for data_test in test_reader():
	results = exe.run(infer_program, 
		feed=feeder.feed(data_test),
		fetch_list=fetch_names)
	indx = -1
	for x3, x2 in data_test:
		indx += 1
		print("%.4f+%.4f+%.4f=%.4f" % (x3, x2[0], x2[1], results[0][indx]))
