import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = np.random.normal(scale=weight_scale,
                                        size=(input_dim, hidden_dim))
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['W2'] = np.random.normal(scale=weight_scale,
                                         size=(hidden_dim, num_classes))
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    scores1, cache1 = affine_relu_forward(x=X,
                                     w=self.params['W1'],
                                     b=self.params['b1'])
    scores2, cache2 = affine_relu_forward(x=scores1,
                                        w=self.params['W2'],
                                        b=self.params['b2'])
    scores = scores2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dout= softmax_loss(scores, y)


    dx2, dw2, db2 = affine_relu_backward(dout=dout, cache=cache2)
    dx1, dw1, db1 = affine_relu_backward(dout=dx2, cache=cache1)

    loss += 0.5 * self.reg * (np.sum(self.params['W1']**2) +
                              np.sum(self.params['W2']**2))

    dw1 += self.reg * self.params['W1']
    dw2 += self.reg * self.params['W2']

    grads = {
      'W1': dw1,
      'W2': dw2,
      'b1': db1,
      'b2': db2
    }
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    for i in xrange(self.num_layers):
      index = str(i + 1)
      if i == 0:
        previous_layer_neurons = input_dim
      else:
        previous_layer_neurons = hidden_dims[i - 1]

      if i == self.num_layers - 1:
        current_layer_neurons = num_classes
      else:
        current_layer_neurons = hidden_dims[i]

      self.params['W' + index] = np.random.normal(scale=weight_scale,
                                              size=(previous_layer_neurons,
                                                    current_layer_neurons))
      self.params['b' + index] = np.zeros(current_layer_neurons)

      if self.use_batchnorm and i < len(hidden_dims):
        self.params['gamma' + index] = np.ones(hidden_dims[i])
        self.params['beta' + index] = np.zeros(hidden_dims[i])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    # Forward through all hidden layers
    input_vals = X
    cache = {}
    for i in xrange(self.num_layers - 1):
      index = repr(i + 1)
      w = self.params['W' + index]
      b = self.params['b' + index]
      if self.use_batchnorm:
        tag = 'abr' + index
        gamma = self.params['gamma' + index]
        beta = self.params['beta' + index]
        bn_param = self.bn_params[i]
        out, cache[tag] = affine_backnorm_relu_forward(x=input_vals, w=w, b=b,
                                                  gamma=gamma, beta=beta,
                                                bn_param=bn_param)
      else:
        tag = 'ar' + index
        out, cache[tag] = affine_relu_forward(x=input_vals, w=w, b=b)

      if self.use_dropout:
        tag = 'dropout' + index
        out, cache[tag] = dropout_forward(out, self.dropout_param)

      input_vals = out

    # Forward through output layer
    last_layer = str(self.num_layers)
    w = self.params['W' + last_layer]
    b = self.params['b' + last_layer]
    tag = 'affine' + last_layer
    scores, cache[tag] = affine_forward(input_vals, w, b)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    # Compute regulation loss
    reg_loss = 0
    for i in xrange(self.num_layers):
      reg_loss += np.sum(self.params['W' + str(i + 1)] ** 2)
    reg_loss *= 0.5 * self.reg

    # Compute network loss
    loss, dout = softmax_loss(scores, y)

    # take sum
    loss += reg_loss

    tag = 'affine' + last_layer
    gradient = affine_backward(dout, cache[tag])
    dout, grads['W' + last_layer], grads['b' + last_layer] = gradient
    grads['W' + last_layer] += self.reg * self.params['W' + last_layer]
    for i in xrange(self.num_layers - 1, 0, -1):
      index = repr(i)

      if self.use_dropout:
        tag = 'dropout' + index
        dout = dropout_backward(dout, cache[tag])

      if self.use_batchnorm:
        tag = 'abr' + index
        gradient = affine_backnorm_relu_backward(dout, cache[tag])
        dout, grads['W' + index], grads['b' + index], _ = gradient
        grads['gamma' + index], grads['beta' + index] = _
      else:
        tag = 'ar' + index
        gradient = affine_relu_backward(dout, cache[tag])
        dout, grads['W' + index], grads['b' + index] = gradient


      grads['W' + index] += self.reg * self.params['W' + index]


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
