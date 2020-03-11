import paddle
import paddle.fluid as fluid
import numpy as np

class SimpleImgConvPool(fluid.dygraph.Layer):
	def __init__(self,
			num_channels,
			num_filters,
			filter_size,
			pool_size,
			pool_stride,
			pool_padding=0,
			pool_type='max',
			global_pooling=False,
			conv_stride=1,
			conv_padding=0,
			conv_dilation=1,
			conv_groups=1,
			act=None,
			use_cudnn=False,
			param_attr=None,
			bias_attr=None):
		super(SimpleImgConvPool, self).__init__()
		self._conv2d = fluid.dygraph.Conv2D
