from __future__ import print_function

import sys
import argparse
import math
import numpy
import paddle
import paddle.fluid as fluid

def parse_args():
	parser = argparse.ArgumentParser("fit_1_line")
	parser.add_argument(
		'--use_gpu',
		type=bool,
		default=False,
		help="Whether to use GPU or not.")
	parser.add_argument(
		'--model', 
		type=str, 
		default='./fit_a_line.inference.model',
		help='the model file.')
	args = parser.parse_args()
	return args
def save_result(points1, points2):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	x1 = [idx for idx in range(len(points1))]
	y1 = points1
	y2 = points2
	l1 = plt.plot(x1, y1, 'r--', label='predictions')
	l2 = plt.plot(x1, y2, 'g--', label='GT')
	plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
	plt.title('predictions VS GT')
	plt.legend()
	plt.savefig('./image/prediction_gt.png')
def main(args):
	use_cuda = args.use_gpu
	place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
	infer_exe = fluid.Executor(place)

	[infer_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(args.model, infer_exe)
	batch_size = 10

	infer_reader = paddle.batch(
		paddle.dataset.uci_housing.test(), batch_size=batch_size)
	infer_data = next(infer_reader())
	infer_feat = numpy.array([data[0] for data in infer_data]).astype('float32')
	infer_label = numpy.array([data[1] for data in infer_data]).astype('float32')

	assert feed_target_names[0] == 'x'
	results = infer_exe.run(infer_program,
		feed={feed_target_names[0]:numpy.array(infer_feat)},
		fetch_list=fetch_targets)
	print("infer results: (House Price)")
	for idx, val in enumerate(results[0]):
		print("%d: %.2f" % (idx, val))
	print("\nground truth:")
	for idx, val in enumerate(infer_label):
		print("%d: %.2f" % (idx, val))
	print("feed_target_names : {}".format(feed_target_names))
	print("fetch_targets : {}".format(fetch_targets))
	save_result(results[0], infer_label)
if __name__ == "__main__":
	args = parse_args()
	main(args)
