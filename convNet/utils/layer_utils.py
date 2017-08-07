"""
    Utilities for building layers in the conv net graph.
"""

import tensorflow as tf
import data_utils as du
import config
import math

def input_placeholders(batch_size=None):
     """
        Generates placeholders for input tensors.

        Input
        =====
        - batch_size: the size of the batch, optional.

        Output
        ======
        - features: features placeholder.
        - labels: labels placeholder.
    """
    features = tf.placeholder(tf.float32, shape=(batch_size, config.sampling_rate, config.num_channels))
    labels = tf.placeholder(tf.float32, shape=(batch_size, config.num_activities))
    
    return features, labels

def weight_variable(shape):
    """
        Creates and initializes a weight variable with the input shape.
        
        The weights are randomly initialized from the normal distribution.
        
        Input
        =====
        - shape: the shape of the weight variable.
        
        Output
        ======
        - tf variable for the weight.
    """
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init, name='weights')

def bias_variable(shape):
    """
        Creates and initializes a bias variable with the input shape.
        
        The biases are initialized to small positive numbers to prevent dead neurons.
        
        Input
        =====
        - shape: the shape of the bias variable.
        
        Output
        ======
        - tf variable for the bias.
    """
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init, name='biases')

def affine_layer_variables(matrix_dim0, matrix_dim1):
    """
        Creates affine layer variables with the input dimensions.
        
        The affine layer has the formula: w * x + b.
        
        The weight matrix (w) and the biases (b) are the returned values of this function.
        
        
        Input
        =====
        - matrix_dim0: the first dimension of the matrix
        - matrix_dim1: the second dimension of the matrix and the dimension of the biases
        
        Output
        ======
        - weights: tf variable representing the weight matrix
        - biases: tf variable representing the bias vector
    """
    weights = weight_variable((matrix_dim0, matrix_dim1))
    biases = bias_variable((matrix_dim1,))
    
    return weights, biases

def fully_connected_layer(input_tensor, output_size, activation_fn=None):
    """
        Creates a fully connected layer: input -> [affine + activiation + batch normalization] -> output.
        
        Input
        =====
        - input_tensor: the layer input
        - output_size: the size of the output
        - activation_fn: the activiation function.
        
        Output
        ======
        - output tensor for the fully connected layer
    """
    input_size = input_tensor.shape.as_list()[1]
    
    w, b = affine_layer_variables(input_size, output_size)
    layer = tf.matmul(input_tensor, w) + b
    
    if(activation_fn):
        layer = activation_fn(layer)
    layer = batch_normalization(layer)    
    return layer

def batch_normalization(input):
    """
        Creates a batch normalization layer.
        
        Input
        =====
        - input: input tensor for the bn layer
        
        Output
        ======
        - bn_layer: output tensor for the bn layer
    """
    mu, var = tf.nn.moments(input, axes=[0])
    alpha = tf.Variable(1.0)
    beta = tf.Variable(0.0)
    bn_layer = tf.nn.batch_normalization(input, mu, var, beta, alpha, 1e-10)
    return bn_layer

def convolutional_layer(input_tensor, filter_tensor, activation_fn=None):
    """
        Creates a 1D convolutional layer.
        
        Input
        =====
        - input_tensor: the input tensor with shape (N, M, C) where:
            + N is the number of datapoints (batch size)
            + M is the input width
            + C is the number of input channels
        
        
        - filter_tensor: the filter tensor with shape (W, C, F) where:
            + W is the filter width
            + C is the number ff input channels
            + F is the number of filters (output channels)
        
        - activation_fn: the activation function that follows the convolution
        
        Output
        ======
        - A tensor with shape (N, M, F).
    """
    output_tensor = tf.nn.conv1d(input_tensor, filter_tensor, stride=1, padding='SAME')
    if(activation_fn):
        output_tensor = activation_fn(output_tensor)
    output_tensor = batch_normalization(output_tensor)
    return output_tensor
    