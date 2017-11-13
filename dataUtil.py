import numpy as np
import pickle

path = '/Users/seokinj/tensorflow/assignments2016/assignment1/cs231n/datasets/cifar-10-batches-py/'
def multiple_trick(w,x,b):
        nw = np.hstack([w,b])
        x = np.vstack([x,[1]]) # hstack ; horizontal stack / vstack ; vertical stack /
                               # w = np.append(w, b, axis=1) #axis 1 : column / axis 0 : row
        return np.dot(nw,x) # np.dot(w,x) == w.dot(x)

def load_CIFAR_batch(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		# data format : 1024(R channel) / 1024(B channel) / 1024(G channel) = reshape(10000,3,32,32)
		# image data : [R1,B1,G1], [R2,B2,G2] = transpose(0,2,3,1)
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
