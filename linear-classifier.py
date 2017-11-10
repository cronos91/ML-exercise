import numpy as np
#from cs231n.data_utils import load_CIFAR10

train_data = '/Users/seokinj/tensorflow/assignments2016/assignment1/cs231n/datasets/cifar-10-batches-py/data_batch_'
#x = np.array([[56], [231], [24], [2]]) # input 
#w = np.array([[0.2, -0.5, 0.1, 2.0], [1.5, 1.3, 2.1, 0.0], [0, 0.25, 0.2, -0.3]]) # each row is one kind of template 
#b = np.array([[1.1], [3.2], [-1.2]])

#print(x.T)     #x is vector. if x is [56, 231, 24, 2], x is interpreted 'vector' and 'matrix'

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def multiple_trick(w,x,b):
	nw = np.hstack([w,b])
	x = np.vstack([x,[1]]) # hstack ; horizontal stack / vstack ; vertical stack / 
			       # w = np.append(w, b, axis=1) #axis 1 : column / axis 0 : row
	return np.dot(nw,x) # np.dot(w,x) == w.dot(x)

"""
def loss_svm(r,):
	delta = 1.0
"""
#for i in range(0,5):
#	unpickle(train_data+str(i))
	

a = unpickle(train_data+'1')
#print(a[b'labels'])
#print(a[b'data'].shape) # (10000, 3072)
print(a[b'data'].reshape(10000,3,32,32))
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(a[b'data'].reshape(10000,3,32,32).transpose(0,2,3,1))
print('###############################')
print(a[b'data'].reshape(10000,32,32,3))
#X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")

#r = multiple_trick(w,x,b)
#print(r)
#print(str(np.argmax(r))+"st : "+str(np.max(r)))
