"""
Functions for building tensorflow computational graph models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope as vs

# So this will run without safekit installed
cyberpath = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, cyberpath)

# TODO: Look at fixing magic number in full covariance loss


def fan_scale(initrange, activation, tensor_in):
    """
    Creates a scaling factor for weight initialization according to best practices.

    :param initrange: Scaling in addition to fan_in scale.
    :param activation: A tensorflow non-linear activation function
    :param tensor_in: Input tensor to layer of network to scale weights for.
    :return: (float) scaling factor for weight initialization.
    """
    if activation == tf.nn.relu:
        initrange *= np.sqrt(2.0/float(tensor_in.get_shape().as_list()[1]))
    else:
        initrange *= (1.0/np.sqrt(float(tensor_in.get_shape().as_list()[1])))
    return initrange


def ident(tensor_in):
    """
    The identity function

    :param tensor_in: Input to operation.
    :return: tensor_in
    """
    return tensor_in


def weights(distribution, shape, dtype=tf.float32, initrange=1e-5,
            seed=None, l2=0.0, name='weights'):
    """
    Wrapper parameterizing common constructions of tf.Variables.

    :param distribution: A string identifying distribution 'tnorm' for truncated normal, 'rnorm' for random normal, 'constant' for constant, 'uniform' for uniform.
    :param shape: Shape of weight tensor.
    :param dtype: dtype for weights
    :param initrange: Scales standard normal and trunctated normal, value of constant dist., and range of uniform dist. [-initrange, initrange].
    :param seed: For reproducible results.
    :param l2: Floating point number determining degree of of l2 regularization for these weights in gradient descent update.
    :param name: For variable scope.
    :return: A tf.Variable.
    """
    with tf.variable_scope(name):
        if distribution == 'norm':
            wghts = tf.Variable(initrange * tf.random_normal(shape, 0, 1, dtype, seed))
        elif distribution == 'tnorm':
            wghts = tf.Variable(initrange * tf.truncated_normal(shape, 0, 1, dtype, seed))
        elif distribution == 'uniform':
            wghts = tf.Variable(tf.random_uniform(shape, -initrange, initrange, dtype, seed))
        elif distribution == 'constant':
            wghts = tf.Variable(tf.constant(initrange, dtype=dtype, shape=shape))
        else:
            raise ValueError("Argument 'distribution takes values 'norm', 'tnorm', 'uniform', 'constant', "
                             "Received %s" % distribution)
        if l2 != 0.0:
            tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(wghts), l2, name=name + 'weight_loss'))
        return wghts


def batch_normalize(tensor_in, epsilon=1e-5, decay=0.999):
    """
    Batch Normalization:
    `Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift`_

    An exponential moving average of means and variances in calculated to estimate sample mean
    and sample variance for evaluations. For testing pair placeholder is_training
    with [0] in feed_dict. For training pair placeholder is_training
    with [1] in feed_dict. Example:

    Let **train = 1** for training and **train = 0** for evaluation

    .. code-block:: python

        bn_deciders = {decider:[train] for decider in tf.get_collection('bn_deciders')}
        feed_dict.update(bn_deciders)

    During training the running statistics are updated, and batch statistics are used for normalization.
    During testing the running statistics are not updated, and running statistics are used for normalization.

    :param tensor_in: (tf.Tensor) Input Tensor.
    :param epsilon: (float) A float number to avoid being divided by 0.
    :param decay: (float) For exponential decay estimate of running mean and variance.
    :return: (tf.Tensor) Tensor with variance bounded by a unit and mean of zero according to the batch.
    """

    is_training = tf.placeholder(tf.int32, shape=[None]) # [1] or [0], Using a placeholder to decide which
                                          # statistics to use for normalization allows
                                          # either the running stats or the batch stats to
                                          # be used without rebuilding the graph.
    tf.add_to_collection('bn_deciders', is_training)

    pop_mean = tf.Variable(tf.zeros([tensor_in.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([tensor_in.get_shape()[-1]]), trainable=False)

    # calculate batch mean/var and running mean/var
    batch_mean, batch_variance = tf.nn.moments(tensor_in, [0])

    # The running mean/variance is updated when is_training == 1.
    running_mean = tf.assign(pop_mean,
                             pop_mean * (decay + (1.0 - decay)*(1.0 - tf.to_float(is_training))) +
                             batch_mean * (1.0 - decay) * tf.to_float(is_training))
    running_var = tf.assign(pop_var,
                            pop_var * (decay + (1.0 - decay)*(1.0 - tf.to_float(is_training))) +
                            batch_variance * (1.0 - decay) * tf.to_float(is_training))

    # Choose statistic
    mean = tf.nn.embedding_lookup(tf.stack([running_mean, batch_mean]), is_training)
    variance = tf.nn.embedding_lookup(tf.stack([running_var, batch_variance]), is_training)

    shape = tensor_in.get_shape().as_list()
    gamma = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[shape[1]], name='gamma'))
    beta = tf.Variable(tf.constant(1.0, dtype=tf.float32, shape=[shape[1]], name='beta'))

    # Batch Norm Transform
    inv = tf.rsqrt(epsilon + variance)
    tensor_in = beta * (tensor_in - mean) * inv + gamma

    return tensor_in


def dropout(tensor_in, prob):
    """
    Adds dropout node.
    `Dropout A Simple Way to Prevent Neural Networks from Overfitting`_

    :param tensor_in: Input tensor.
    :param prob: The percent of units to keep.
    :return: Tensor of the same shape of *tensor_in*.
    """
    if isinstance(prob, float):
        keep_prob = tf.placeholder(tf.float32)
        tf.add_to_collection('dropout_prob', (keep_prob, prob))
    return tf.nn.dropout(tensor_in, keep_prob)


def layer_norm(h):
    """

    :param h: (tensor) Hidden layer of neural network
    :return: (tensor) Hidden layer after layer_norm transform
    """
    dim = h.get_shape().as_list()
    bias = tf.Variable(tf.zeros([1, dim[1]], dtype=tf.float32))
    gain = tf.Variable(tf.ones([1, dim[1]], dtype=tf.float32))
    mu, variance = tf.nn.moments(h, [1], keep_dims=True)
    return (gain/tf.sqrt(variance))*(h - mu) + bias


def dnn(x, layers=[100, 408], act=tf.nn.relu, scale_range=1.0, norm=None, keep_prob=None, name='nnet'):
    """
    An arbitrarily deep neural network. Output has non-linear activation.

    :param x: (tf.tensor) Input to the network.
    :param layers: List of integer sizes of network layers.
    :param act: Activation function to produce hidden layers of neural network.
    :param scale_range: (float) Scaling factor for initial range of weights (Set to 1/sqrt(fan_in) for tanh, sqrt(2/fan_in) for relu.
    :param norm: Normalization function. Could be layer_norm or other function that retains shape of tensor.
    :param keep_prob: (float) The percent of nodes to keep in dropout layers.
    :param name: (str) For naming and variable scope.
    :return: (tf.Tensor) Output of neural net. This will be just following a non linear transform, so that final activation has not been applied.
    """
    if type(scale_range) is not list:
        scale_range = [scale_range] * len(layers)
    assert len(layers) == len(scale_range)

    for ind, hidden_size in enumerate(layers):
        with tf.variable_scope('layer_%s' % ind):

            fan_in = x.get_shape().as_list()[1]
            W = tf.Variable(fan_scale(scale_range[ind], act, x) * tf.truncated_normal([fan_in, hidden_size],
                                                                                      mean=0.0, stddev=1.0,
                                                                                      dtype=tf.float32, seed=None,
                                                                                      name='W'))
            tf.add_to_collection(name + '_weights', W)
            b = tf.Variable(tf.zeros([hidden_size])) + 0.1*(float(act == tf.nn.relu))
            tf.add_to_collection(name + '_bias', b)
            x = tf.matmul(x, W) + b
            if norm is not None:
                x = norm(x)
            x = act(x, name='h' + str(ind))  # The hidden layer
            tf.add_to_collection(name + '_activation', x)
            if keep_prob:
                x = dropout(x, keep_prob)
    return x


def bidir_lm_rnn(x, t, token_embed, layers, seq_len=None, context_vector=None, cell=tf.nn.rnn_cell.BasicLSTMCell):
    """
    Token level bidirectional LSTM language model that uses a sentence level context vector.

    :param x: Input to rnn
    :param t: Targets for language model predictions (typically next token in sequence)
    :param token_embed: (tensor) MB X ALPHABET_SIZE.
    :param layers: A list of hidden layer sizes for stacked lstm
    :param seq_len: A 1D tensor of mini-batch size for variable length sequences
    :param context_vector: (tensor) MB X 2*CONTEXT_LSTM_OUTPUT_DIM. Optional context to append to each token embedding
    :param cell: (class) A tensorflow RNNCell sub-class
    :return: (tensor) tuple-token_losses , (list of tensors) hidden_states, (tensor) final_hidden
    """

    token_set_size = token_embed.get_shape().as_list()[0]
    with tf.variable_scope('forward'):
        fw_cells = [cell(num_units) for num_units in layers]
        fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cells, state_is_tuple=True)
    with tf.variable_scope('backward'):
        bw_cells = [cell(num_units) for num_units in layers]
        bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cells, state_is_tuple=True)
    x_lookup = tf.nn.embedding_lookup(token_embed, x)

    # List of mb X embedding_size tensors
    input_features = tf.unstack(x_lookup, axis=1)

    if context_vector is not None:
        input_features = [tf.concat([embedding, context_vector], 1) for embedding in input_features]
    # input_features: list of sentence long tensors (mb X embedding_size)
    hidden_states, fw_cell_state, bw_cell_state = tf.nn.static_bidirectional_rnn(fw_cell, bw_cell, input_features,
                                                                          dtype=tf.float32,
                                                                          sequence_length=seq_len,
                                                                          scope='language_model')
    final_hidden = tf.concat((fw_cell_state[-1].h, bw_cell_state[-1].h), 1)
    f_hidden_states, b_hidden_states = tf.split(tf.stack(hidden_states), 2, axis=2) # 2  sen_len X num_users X hidden_size tensors
    # truncate forward and backward output to align for prediction
    f_hidden_states = tf.stack(tf.unstack(f_hidden_states)[:-2]) # sen_len-2 X num_users X hidden_size tensor
    b_hidden_states = tf.stack(tf.unstack(b_hidden_states)[2:]) # sen_len-2 X num_users X hidden_size tensor
    # concatenate forward and backward output for prediction
    prediction_states = tf.unstack(tf.concat((f_hidden_states, b_hidden_states), 2))  # sen_len-2 long list of num_users X 2*hidden_size tensors
    token_losses = batch_softmax_dist_loss(t, prediction_states, token_set_size)

    return token_losses, hidden_states, final_hidden


def lm_rnn(x, t, token_embed, layers, seq_len=None, context_vector=None, cell=tf.nn.rnn_cell.BasicLSTMCell):
    """
    Token level LSTM language model that uses a sentence level context vector.

    :param x: (tensor) Input to rnn
    :param t: (tensor) Targets for language model predictions (typically next token in sequence)
    :param token_embed: (tensor) MB X ALPHABET_SIZE.
    :param layers: A list of hidden layer sizes for stacked lstm
    :param seq_len: A 1D tensor of mini-batch size for variable length sequences
    :param context_vector: (tensor) MB X 2*CONTEXT_LSTM_OUTPUT_DIM. Optional context to append to each token embedding
    :param cell: (class) A tensorflow RNNCell sub-class
    :return: (tuple) token_losses (tensor), hidden_states (list of tensors), final_hidden (tensor)
    """

    token_set_size = token_embed.get_shape().as_list()[0]
    cells = [cell(num_units) for num_units in layers]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    # mb X sentence_length X embedding_size
    x_lookup = tf.nn.embedding_lookup(token_embed, x)

    # List of mb X embedding_size tensors
    input_features = tf.unstack(x_lookup, axis=1)

    # input_features: list max_length of sentence long tensors (mb X embedding_size+context_size)
    if context_vector is not None:
        input_features = [tf.concat([embedding, context_vector], 1) for embedding in input_features]

    # hidden_states: sentence length long list of tensors (mb X final_layer_size)
    # cell_state: data structure that contains the cell state for each hidden layer for a mini-batch (complicated)
    hidden_states, cell_state = tf.nn.static_rnn(cell, input_features,
                                          initial_state=None,
                                          dtype=tf.float32,
                                          sequence_length=seq_len,
                                          scope='language_model')
    # batch_size X sequence_length (see tf_ops for def)
    token_losses = batch_softmax_dist_loss(t, hidden_states, token_set_size)
    final_hidden = cell_state[-1].h
    return token_losses, hidden_states, final_hidden


def join_multivariate_inputs(feature_spec, specs, embedding_ratio, max_embedding, min_embedding):
    """
    Makes placeholders for all input data, performs a lookup on an embedding matrix for each categorical feature,
    and concatenates the resulting real-valued vectors from individual features into a single vector for each data point in the batch.

    :param feature_spec: A dict {categorical: [c1, c2, ..., cp], continuous:[f1, f2, ...,fk]
                        which lists which features to use as categorical and continuous inputs to the model.
                        c1, ..., cp, f1, ...,fk should match a key in specs.
    :param specs: A python dict containing information about which indices in the incoming data point correspond to which features.
                  Entries for continuous features list the indices for the feature, while entries for categorical features
                  contain a dictionary- {'index': i, 'num_classes': c}, where i and c are the index into the datapoint, and number of distinct
                  categories for the category in question.
    :param embedding_ratio: Determines size of embedding vectors for each categorical feature: num_classes*embedding_ratio (within limits below)
    :param max_embedding: A limit on how large an embedding vector can be.
    :param min_embedding: A limit on how small an embedding vector can be.
    :return: A tuple (x, placeholderdict):
            (tensor with shape [None, Sum_of_lengths_of_all_continuous_feature_vecs_and_embedding_vecs],
            dict to store tf placeholders to pair with data, )
    """

    placeholderdict, embeddings, continuous_features, targets = {}, {}, {}, {}

    # Make placeholders for all input data and select embeddings for categorical data
    for dataname in feature_spec['categorical']:
        embedding_size = math.ceil(embedding_ratio * specs[dataname]['num_classes'])
        embedding_size = int(max(min(max_embedding, embedding_size), min_embedding))
        with tf.variable_scope(dataname):
            placeholderdict[dataname] = tf.placeholder(tf.int32, [None])
            embedding_matrix = tf.Variable(1e-5*tf.truncated_normal((specs[dataname]['num_classes'], embedding_size), dtype=tf.float32))
            embeddings[dataname] = tf.nn.embedding_lookup(embedding_matrix, placeholderdict[dataname])

    for dataname in feature_spec['continuous']:
        placeholderdict[dataname] = tf.placeholder(tf.float32, [None, len(specs[dataname]['index'])])
        continuous_features[dataname] = placeholderdict[dataname]

    # concatenate all features
    return tf.concat(continuous_features.values() + embeddings.values(), 1, name='features'), placeholderdict


# ============================================================
# ================ LOSS FUNCTIONS ============================
# ============================================================

def softmax_dist_loss(truth, h, dimension, scale_range=1.0, U=None):
    """
    This function paired with a tensorflow optimizer is multinomial logistic regression.
    It is designed for cotegorical predictions.

    :param truth: A tensorflow vector tensor of integer class labels.
    :param h: A placeholder if doing simple multinomial logistic regression, or the output of some neural network.
    :param dimension: Number of classes in output distribution.
    :param scale_range: For scaling the weight matrices (by default weights are initialized two 1/sqrt(fan_in)) for tanh activation and sqrt(2/fan_in) for relu activation.
    :param U: Optional weight tensor (If you is not provided a new weight tensor is made)
    :return: (Tensor[MB X 1]) Cross-entropy of true distribution vs. predicted distribution.
    """
    fan_in = h.get_shape().as_list()[1]
    if U is None:
        U = tf.Variable(fan_scale(scale_range, tf.tanh, h) * tf.truncated_normal([fan_in, dimension],
                                                                             dtype=tf.float32,
                                                                             name='W'))
    b = tf.Variable(tf.zeros([dimension]))
    y = tf.matmul(h, U) + b
    loss_column = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=truth)
    loss_column = tf.reshape(loss_column, [-1, 1])

    return loss_column


def batch_softmax_dist_loss(truth, h, dimension, scale_range=1.0):
    """
    This function paired with a tensorflow optimizer is multinomial logistic regression.
    It is designed for cotegorical predictions.

    :param truth: (tf.Tensor) A tensorflow vector tensor of integer class labels.
    :param h: (tf.Tensor) A placeholder if doing simple multinomial logistic regression, or the output of some neural network.
    :param dimension: (int) Number of classes in output distribution.
    :param scale_range: (float) For scaling the weight matrices (by default weights are initialized two 1/sqrt(fan_in)) for tanh activation and sqrt(2/fan_in) for relu activation.
    :return: (tf.Tensor, shape = [MB, Sequence_length]) Cross-entropy of true distribution vs. predicted distribution.
    """
    fan_in = h[0].get_shape().as_list()[1]
    initializer = fan_scale(scale_range, tf.tanh, h[0]) * tf.truncated_normal([fan_in, dimension],
                                                                               dtype=tf.float32,
                                                                               name='W')
    U = tf.get_variable('softmax_weights', initializer=initializer)

    hidden_tensor = tf.stack(h) # sequence_length X batch_size X final_hidden_size
    tf.add_to_collection('logit_weights', U)
    b = tf.get_variable('softmax_bias', initializer=tf.zeros([dimension]))
    ustack = tf.stack([U]*len(h)) #sequence_length X final_hidden_size X dimension
    logits = tf.matmul(hidden_tensor, ustack) + b # sequence_length X batch_size X dimension
    logits = tf.transpose(logits, perm=[1, 0, 2]) # batch_size X sequence_length X dimension
    tf.add_to_collection("true_probabilities", tf.nn.softmax(logits)) # added to store probabilities of true logline
    loss_matrix = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=truth) # batch_size X sequence_length
    return loss_matrix


def eyed_mvn_loss(truth, h, scale_range=1.0):
    """
    This function takes the output of a neural network after it's last activation, performs an affine transform,
    and returns the squared error of this result and the target.

    :param truth: A tensor of target vectors.
    :param h: The output of a neural network post activation.
    :param scale_range: For scaling the weight matrices (by default weights are initialized two 1/sqrt(fan_in)) for
    tanh activation and sqrt(2/fan_in) for relu activation.
    :return: (tf.Tensor[MB X D], None) squared_error, None
    """
    fan_in = h.get_shape().as_list()[1]
    dim = truth.get_shape().as_list()[1]
    U = tf.Variable(fan_scale(scale_range, tf.tanh, h) * tf.truncated_normal([fan_in, dim],
                                                                             dtype=tf.float32, name='U'))
    b = tf.Variable(tf.zeros([dim]))
    y = tf.matmul(h, U) + b
    loss_columns = tf.square(y-truth)
    return loss_columns, None


def diag_mvn_loss(truth, h, scale_range=1.0, variance_floor=0.1):
    """
    Takes the output of a neural network after it's last activation, performs an affine transform.
    It returns the mahalonobis distances between the targets and the result of the affine transformation, according
    to a parametrized Normal distribution with diagonal covariance. The log of the determinant of the parametrized
    covariance matrix is meant to be minimized to avoid a trivial optimization.

    :param truth: (tf.Tensor) The targets for this minibatch.
    :param h: (tf.Tensor) The output of dnn. (Here the output of dnn , h, is assumed to be the same dimension as truth)
    :param scale_range: (float) For scaling the weight matrices (by default weights are initialized two 1/sqrt(fan_in)) for tanh activation and sqrt(2/fan_in) for relu activation.
    :param variance_floor: (float, positive) To ensure model doesn't find trivial optimization.
    :return: (tf.Tensor shape=[MB X D], tf.Tensor shape=[MB X 1]) Loss matrix, log_of_determinants of covariance matrices.
    """
    fan_in = h.get_shape().as_list()[1]
    dim = truth.get_shape().as_list()[1]
    U = tf.Variable(
        fan_scale(scale_range, tf.tanh, h) * tf.truncated_normal([fan_in, 2 * dim],
                                                                 dtype=tf.float32,
                                                                 name='U'))
    b = tf.Variable(tf.zeros([2 * dim]))
    y = tf.matmul(h, U) + b
    mu, var = tf.split(y, 2, axis=1)  # split y into two even sized matrices, each with half the columns
    var = tf.maximum(tf.exp(var), variance_floor) # make the variance non-negative
                     #tf.constant(variance_floor, shape=[dim], dtype=tf.float32))
    logdet = tf.reduce_sum(tf.log(var), 1)  # MB x 1
    loss_columns = tf.square(truth - mu) / var  # MB x D
    return loss_columns, tf.reshape(logdet, [-1, 1])


# TODO: Look at fixing magic number in full covariance loss
def full_mvn_loss(truth, h):
    """
    Takes the output of a neural network after it's last activation, performs an affine transform.
    It returns the mahalonobis distances between the targets and the result of the affine transformation, according
    to a parametrized Normal distribution. The log of the determinant of the parametrized
    covariance matrix is meant to be minimized to avoid a trivial optimization.

    :param truth: Actual datapoints to compare against learned distribution
    :param h: output of neural network (after last non-linear transform)
    :return: (tf.Tensor[MB X D], tf.Tensor[MB X 1]) Loss matrix, log_of_determinants of covariance matrices.
    """
    fan_in = h.get_shape().as_list()[1]
    dimension = truth.get_shape().as_list()[1]
    U = 100*tf.Variable(tf.truncated_normal([fan_in, dimension + dimension**2],
                                                                 dtype=tf.float32,
                                                                 name='U'))
    b = tf.Variable(tf.zeros([dimension + dimension**2]))
    y = tf.matmul(h, U) + b
    mu = tf.slice(y, [0, 0], [-1, dimension])  # is MB x dimension
    var = tf.slice(y, [0, dimension], [-1, -1])*0.0001  # is MB x dimension^2 # WARNING WARNING TODO FIX THIS MAGIC NUMBER
    var = tf.reshape(var, [-1, dimension, dimension])  # make it a MB x D x D tensor (var is a superset of the lower triangular part of a Cholesky decomp)
    var_diag = tf.exp(tf.matrix_diag_part(var)) + 1 # WARNING: FIX THIS MAGIC NUMBER
    var = tf.matrix_set_diag(var,var_diag)
    var = tf.matrix_band_part(var, -1, 0)
    z = tf.squeeze(tf.matrix_triangular_solve(var, tf.reshape(truth - mu, [-1, dimension, 1]), lower=True, adjoint=False))  # z should be MB x D
    inner_prods = tf.reduce_sum(tf.square(z), 1)  # take row-wise inner products of z, leaving MB x 1 vector
    logdet = tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(var))), 1) # diag_part converts MB x D x D to MB x D, square and log preserve, then sum makes MB x 1
    loss_column = inner_prods  # is MB x 1 ... hard to track of individual features' contributions due to correlations
    tf.add_to_collection('full', var_diag)
    tf.add_to_collection('full', var)
    return tf.reshape(loss_column, [-1, 1]), tf.reshape(logdet, [-1, 1])


def multivariate_loss(h, loss_spec, placeholder_dict, variance_floor=0.01):
    """
    Computes a multivariate loss according to loss_spec.

    :param h: Final hidden layer of dnn or rnn. (Post-activation)
    :param loss_spec: A tuple of 3-tuples of the form (input_name, loss_function, dimension) where
                        input_name is the same as a target in datadict,
                         loss_function takes two parameters, a target and prediction,
                         and dimension is the dimension of the target.
    :param placeholder_dict: A dictionary to store placeholder tensors for target values.
    :param variance_floor: (float) Parameter for diag_mvn_loss.
    :return loss_matrix: (MB X concatenated_feature_size Tensor) Contains loss for all contributors for each data point.
    """

    log_det_list, log_det_names, loss_list, loss_names = [], [], [], []
    for i, (input_name, loss_func, dimension) in enumerate(loss_spec):
        with tf.variable_scope(input_name):
            # this input will be a (classification or regression) target - need to define a placeholder for it
            if loss_func == softmax_dist_loss:
                x = tf.placeholder(tf.int32, [None])
            else:
                x = tf.placeholder(tf.float32, [None, dimension])
            placeholder_dict["target_%s" % input_name] = x

            # predict this input from the current hidden state
            if loss_func == softmax_dist_loss: # discrete
                component_wise_point_loss = loss_func(x, h, dimension)# MB X 1
            elif loss_func == diag_mvn_loss: # continuous
                component_wise_point_loss, logdet = loss_func(x, h, variance_floor=variance_floor)# MB X DIM_MULTIVARIATE, MB X 1
                if logdet is not None:
                    log_det_list.append(logdet)
            else: # continuous
                component_wise_point_loss, logdet = loss_func(x, h)# MB X DIM_MULTIVARIATE, MB X 1
                if logdet is not None:
                    log_det_list.append(logdet)
            loss_list.append(component_wise_point_loss)

    loss_list.extend(log_det_list)
    loss_matrix = tf.concat(loss_list, 1)  # is MB x (total num contributors)

    return loss_matrix


def layer_norm_rnn(inputs,
                   initial_state=None,
                   layers=(10,),
                   sequence_lengths=None,
                   state_index=-1):
    """
    :param inputs: A list with length the number of time steps of longest sequence in the batch. inputs contains
                    matrices of shape=[num_sequences X feature_dimension]
    :param initial_state: Initialized first hidden states. A  tuple of len(layers) tuples of cell and hidden state tensors
    :param layers: list of number of nodes in each of stacked lstm layers
    :param sequence_lengths: A vector of sequence lengths of size batch_size
    :param state_index: If -1, last state is returned, if None all states are returned, if 1, second state is returned.
    :return: hidden_states, current_state
    """

    layer_norm_stack = [nn.rnn_cell.BasicLSTMCell(layers[0], state_is_tuple=True)]
    for i in range(1, len(layers)):
        layer_norm_stack.append(tf.contrib.rnn.LayerNormBasicLSTMCell(layers[i]))

    cell = nn.rnn_cell.MultiRNNCell(layer_norm_stack, state_is_tuple=True)
    hidden_states, current_state = true_bptt_rnn(cell,
                                                 inputs,
                                                 initial_state=initial_state,
                                                 dtype=tf.float32,
                                                 sequence_length=sequence_lengths,
                                                 state_index=state_index)
    return hidden_states, current_state


def swapping_rnn(inputs,
                 initial_state=None,
                 layers=(10,),
                 sequence_lengths=None,
                 state_index=-1):
    """
    :param inputs: A list with length the number of time steps of longest sequence in the batch. inputs contains
                    matrices of shape=[num_sequences X feature_dimension]
    :param initial_state: Initialized first hidden states. A  tuple of len(layers) tuples of cell and hidden state tensors
    :param layers: list of number of nodes in each of stacked lstm layers
    :param sequence_lengths: A vector of sequence lengths of size batch_size
    :param state_index: If -1, last state is returned, if None all states are returned, if 1, second state is returned.
    :return:
    """

    cells = [nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True) for num_units in layers]
    cell = nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    hidden_states, current_state = true_bptt_rnn(cell,
                                                 inputs,
                                                 initial_state=initial_state,
                                                 dtype=tf.float32,
                                                 sequence_length=sequence_lengths,
                                                 state_index=state_index)
    return hidden_states, current_state


# ==================================================================================
# =======================Adapted From Tensorflow====================================
# ==================================================================================
def _state_size_with_prefix(state_size, prefix=None):
    """Helper function that enables int or TensorShape shape specification.
    This function takes a size specification, which can be an integer or a
    TensorShape, and converts it into a list of integers. One may specify any
    additional dimensions that precede the final state size specification.

    :param state_size: TensorShape or int that specifies the size of a tensor.
      prefix: optional additional list of dimensions to prepend.
    :return: result_state_size: list of dimensions the resulting tensor size.
    """
    result_state_size = tensor_shape.as_shape(state_size).as_list()
    if prefix is not None:
        if not isinstance(prefix, list):
            raise TypeError("prefix of _state_size_with_prefix should be a list.")
        result_state_size = prefix + result_state_size
    return result_state_size


def true_bptt_rnn(cell, inputs, initial_state=None, dtype=None,
                  sequence_length=None, scope=None, state_index=1):  ### Adapted From Tensorflow
    """
    Creates a recurrent neural network specified by RNNCell `cell`.
    The simplest form of RNN network generated is:

    .. code:: python

        state = cell.zero_state(...)
        outputs = []
        for input_ in inputs:
            output, state = cell(input_, state)
            outputs.append(output)
      return (outputs, state)

    However, a few other options are available:
    An initial state can be provided.
    If the sequence_length vector is provided, dynamic calculation is performed.
    This method of calculation does not compute the RNN steps past the maximum
    sequence length of the minibatch (thus saving computational time),
    and properly propagates the state at an example's sequence length
    to the final state output.
    The dynamic calculation performed is, at time t for batch row b,

    .. code ::

        (output, state)(b, t) = (t >= sequence_length(b)) ? (zeros(cell.output_size), states(b, sequence_length(b) - 1)) : cell(input(b, t), state(b, t - 1))

    :param cell: An instance of RNNCell.
    :param inputs: A length T list of inputs, each a tensor of shape
                   [batch_size, input_size].
    :param initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
    :param dtype: (optional) The data type for the initial state.  Required if
        initial_state is not provided.
    :param sequence_length: Specifies the length of each sequence in inputs.
        An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
    :param scope: VariableScope for the created subgraph; defaults to "RNN".
    :param state_index: (int) If -1 final state is returned, if 1 state after first rnn step is returned. If anything else
                        all states are returned
    :return: A pair (outputs, state) where:

        - outputs is a length T list of outputs (one for each input)
        - state is the final state or a a length T list of cell states
    :raise: TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If `inputs` is `None` or an empty list, or if the input depth
        (column size) cannot be inferred from inputs via shape inference.
    """

    if not isinstance(cell, rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    states = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with vs.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        if inputs[0].get_shape().ndims != 1:
            input_shape = inputs[0].get_shape().with_rank_at_least(2)
            input_shape[1:].assert_is_fully_defined()
            (fixed_batch_size, input_size) = input_shape[0], input_shape[1:]
            if input_size[0].value is None:
                raise ValueError(
                    "Input size (second dimension of inputs[0]) must be accessible via "
                    "shape inference, but saw value None.")
        else:
            fixed_batch_size = inputs[0].get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = array_ops.shape(inputs[0])[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:  # Prepare variables
            sequence_length = math_ops.to_int32(sequence_length)
            # convert int to TensorShape if necessary
            output_size = _state_size_with_prefix(cell.output_size,
                                                  prefix=[batch_size])
            zero_output = array_ops.zeros(
                array_ops.pack(output_size),
                inputs[0].dtype)
            zero_output_shape = _state_size_with_prefix(
                cell.output_size, prefix=[fixed_batch_size.value])
            zero_output.set_shape(tensor_shape.TensorShape(zero_output_shape))
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length is not None:
                (output, state) = rnn._rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=cell.state_size)
            else:
                (output, state) = call_cell()
            states.append(state)
            outputs.append(output)
        if state_index == 1:
            next_state = states[1]
        elif state_index == -1:
            next_state = states[-1]
        else:
            next_state = states
    return outputs, next_state

