import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  # compute the loss and the gradient
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
  margins = np.zeros([num_train,num_classes])
  scoress = np.zeros([num_train,num_classes])
  loss = 0.0
  data_loss = 0.0
  
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dmargins = np.ones([num_train,num_classes])
  dscores = np.zeros([num_train,num_classes])
  
  for i in range(num_train):
    scores = X[i].dot(W)
    scoress[i,:] = scores
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        data_loss += margin
        margins[i,j] = margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  data_loss /= num_train
  
  # Add regularization to the loss.
  reg_loss = reg * np.sum(W * W)
  loss = data_loss + reg_loss
  '''
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  #print('reg loss= ', reg_loss)
  #print('data loss = ', data_loss)
  dmargins *= 1/num_train
  
  for i in range(num_train):
      for j in range(num_classes):
          #if margins[i,j] == 0:
          #    dmargins[i,j] = 0
          if j == y[i]:
              dscores[i,j] -= dmargins[i,j] 
          else:
              dscores[i,j] += dmargins[i,j] 
  #print(margins[100])
  #print(dscores[100])
  dW = X.T.dot(dscores)
  #print(dW[100])
  dW += 2*reg*W
  '''
  
  
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T
        dW[:,y[i]] += -X[i].T 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  
  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[1]
  num_train = X.shape[0]
  margins = np.zeros([num_train,num_classes])
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  scoress = X.dot(W)
  for i in range(num_train):
    correct_class_score = scoress[i,y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margins[i,j] = scoress[i,j] - correct_class_score + 1 # note delta = 1
      if margins[i,j] < 0:
        margins[i,j] = 0
  data_loss = np.sum(margins) / num_train
  reg_loss = reg * np.sum(W * W)
  loss = data_loss + reg_loss
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
  dmargins = np.ones([num_train,num_classes])
  dscores = np.zeros([num_train,num_classes])
  dmargins *= 1/num_train
  for i in range(num_train):
      for j in range(num_classes):
          #if margins[i,j] == 0:
          #    dmargins[i,j] = 0
          if j == y[i]:
              dscores[i,j] -= dmargins[i,j] 
          else:
              dscores[i,j] += dmargins[i,j] 
  dW = X.T.dot(dscores) + 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
