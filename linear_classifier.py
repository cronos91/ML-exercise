import numpy as np
import dataUtil

#x = np.array([[56], [231], [24], [2]]) # input 
#w = np.array([[0.2, -0.5, 0.1, 2.0], [1.5, 1.3, 2.1, 0.0], [0, 0.25, 0.2, -0.3]]) # each row is one kind of template 
#b = np.array([[1.1], [3.2], [-1.2]])
#print(x.T)     #x is vector. if x is [56, 231, 24, 2], x is interpreted 'vector' and 'matrix

'''
class Optimizer():
'''	

w = np.random.randn(3073, 10) * 0.0001
x_train, y_train, x_test, y_test = dataUtil.load_CIFAR_data()
x_train = np.reshape(x_train, (x_train.shape[0],-1))
x_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))])

loss = 0.0

# w : (3073, 10) / x_train : (50000, 3073) / x1 : (1,3073)
# x1w, scores : (1, 10) ... scores of a row of X
# y_train : (50000,) ... # of correct class
# correct score : scores[# class]

for i in range(0,x_train.shape[0]):
	scores = np.dot(x_train[i],w)
	correct = scores[y_train[i]]
	for j in range(0,10):
		if j != y_train[i]:
			margin = scores[j] - correct + 1 # delta = 1
		else:
			continue
		if margin > 0:
			loss += margin 

loss /= x_train.shape[0]
loss += 0.5*1*np.sum(w*w)

print(loss)
