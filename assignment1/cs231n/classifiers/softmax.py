import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  #print(X.shape)
  #print(X[1].shape)
  for i in range(num_train):
      scores = X[i].dot(W)
      shift_scores = scores - max(scores)
      loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
      loss += loss_i
      for j in range(num_class):
         softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
         if j == y[i]:
             dW[:,j] += (-1 + softmax_output) *X[i] 
         else: 
             dW[:,j] += softmax_output *X[i] 
  loss /= num_train
  loss += reg * np.sum(W*W)
  dW = dW/num_train + 2*reg* W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis=1).reshape(-1,1)
  loss_is = -shift_scores[range(num_train),y] + np.log(np.sum(np.exp(shift_scores),axis=1))
  loss = np.sum(loss_is)/num_train + reg * np.sum(W*W)
  
  dscores = np.zeros([num_train,num_class])
  dscores[range(num_train),y] = -1
  dscores += np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape(-1,1) 
  dscores /= num_train
  dW = X.T.dot(dscores) + 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

