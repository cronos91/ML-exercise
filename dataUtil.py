import numpy as np
import pickle

path = '/Users/seokinj/tensorflow/assignments2016/assignment1/cs231n/datasets/cifar-10-batches-py/'
def load_CIFAR_batch(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		X = dict[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
		Y = np.array(dict[b'labels'])
		return X,Y

def load_CIFAR_data():
	xs = []
	ys = []
	for x in range(1,6):
		X,Y = load_CIFAR_batch(path+'data_batch_'+str(x))
		xs.append(X)
		ys.append(Y)
	x_trains = np.concatenate(xs)
	y_trains = np.concatenate(ys)
	x_tests, y_tests = load_CIFAR_batch(path+'test_batch')
	return x_trains, y_trains, x_tests, y_tests	

X_trains, Y_trains, X_tests, Y_tests = load_CIFAR_data()
'''
i    "X_train = np.reshape(X_train, (X_train.shape[0], -1))\n",
    "X_val = np.reshape(X_val, (X_val.shape[0], -1))\n",
    "X_test = np.reshape(X_test, (X_test.shape[0], -1))\n",
    "X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))\n",
'''
