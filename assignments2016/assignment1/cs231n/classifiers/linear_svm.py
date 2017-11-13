import numpy as np
from random import shuffle

def L_i(W, X, y):
  delta = 1.0
  scores = X.dot(W) # socre = x1 * W = (1,10)
  correct_class_scores = scores[y] # y = class number of correct answer
  D = W.shape[0]
  loss_i = 0.0
  for j in xrange(D):
    if j == y:
      continue
    loss_i += max(0, scores[j] - correct_class_score + delta 
  return loss_i

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (3073, 10) containing weights.
  - X: A numpy array of shape (50000, 3073) containing a minibatch of data.
  - y: A numpy array of shape (50000,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  - dW : A numpy array of shape (3073, 10) 

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  h = 0.00001

  for i in xrange(num_train):
    loss += L_i(W, X[i], y[i])
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
	ix = it.multi_index
	old_value = W[ix]
	W[ix] = old_value + h
	fxh = L_i(W, X[i], y[i]) 
	W[ix] = old_value 		# ?????
    	dW[ix] = (fxh - loss) / h
	it.iternext()
	
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
