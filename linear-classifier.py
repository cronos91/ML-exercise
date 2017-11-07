import numpy as np

x = np.array([[56], [231], [24], [2]]) # input 
w = np.array([[0.2, -0.5, 0.1, 2.0], [1.5, 1.3, 2.1, 0.0], [0, 0.25, 0.2, -0.3]]) # each row is one kind of template 
b = np.array([[1.1], [3.2], [-1.2]])

#print(x.T)     #x is vector. if x is [56, 231, 24, 2], x is interpreted 'vector' and 'matrix'

def multiple_trick(w,x,b):
	nw = np.hstack([w,b])
	x = np.vstack([x,[1]]) # hstack ; horizontal stack / vstack ; vertical stack / 
			       # w = np.append(w, b, axis=1) #axis 1 : column / axis 0 : row
	return np.dot(nw,x) # np.dot(w,x) == w.dot(x)

r = multiple_trick(w,x,b)
print(str(np.argmax(r))+"st : "+str(np.max(r)))
'''
def svm()
'''
