import sys
from MNIST import *
def test(reader, model, batch_size):
	acc_set = []
	avg_loss_set = []
	for batch_id, data in enumerate(reader()):
		dy_x_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
		y_data = np.array([x[1] for x in data]).astype('int64').reshape(batch_size, 1)
		img = fluid.dygraph.to_variable(dy_x_data)
		label = fluid.dygraph.to_variable(y_data)
		label.stop_gradient = True

		prediction, acc = model(img, label)
		
		loss = fluid.layers.cross_entropy(input=prediction, label=label)
		avg_loss = fluid.layers.mean(loss)
		avg_loss_set.append(float(avg_loss.numpy()))
		acc_set.append(float(acc.numpy()))
	acc_val_mean = np.array(acc_set).mean()
	avg_loss_val_mean = np.array(avg_loss_set).mean()

	return avg_loss_val_mean, acc_val_mean
try:
	model_file = sys.argv[1]
except:
	sys.stderr.write("\tpython "+sys.argv[0]+" model_file\n")
	sys.exit(-1)
with fluid.dygraph.guard():
	mnist = MNIST()
	model_dict, _ = fluid.dygraph.load_dygraph(model_file)
	mnist.load_dict(model_dict)
	print("checkpoint loaded!")
	mnist.eval()
	
	BATCH_SIZE = 64
	reader = paddle.batch(
		paddle.dataset.mnist.test(), batch_size=BATCH_SIZE, drop_last=True)
	loss, acc = test(reader, mnist, BATCH_SIZE)
	print("test loss is {}, test acc is {}".format(loss, acc))
