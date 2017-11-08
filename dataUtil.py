import numpy as np
import pickle

path = '/Users/seokinj/tensorflow/assignments2016/assignment1/cs231n/datasets/cifar-10-batches-py/data_batch_'
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
		X,Y = load_CIFAR_batch(path+str(x))
		xs.append(X)
		ys.append(Y)
	print(len(xs))
	print(len(xs[0]))
	return np.concatenate(xs).shape

print(load_CIFAR_data())
