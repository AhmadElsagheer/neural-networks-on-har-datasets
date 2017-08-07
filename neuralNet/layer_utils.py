"""
    Utilities for building layers of the model graph.
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
    features = tf.placeholder(tf.float32, shape=(batch_size, config.INPUT_DIM))
    labels = tf.placeholder(tf.float32, shape=(batch_size, 6))

    return features, labels


def affine_layer_variables(matrix_dim0, matrix_dim1):
    """
        Create variables for an affine layer with the given dimensions.

        The affine layer has the formula: w * x + b. The weight matrix
        (w) and the biases (b) are the returned values of this funcion.

        The weights are randomly initialized from normal distribution and
        the biases are initialized to zero.

        Input
        =====
        - matrix_dim0:  the first dimension of the matrix.
        - matrix_dim1:  the second dimension of the matrix and the dimension
                        of the biases.

        Output
        ======
        - weights: tf variable representing the weight matrix.
        - biases: tf variable representing the bias vector.
    """
    w_init = tf.truncated_normal([matrix_dim0, matrix_dim1], stddev=0.1)
    b_init = tf.zeros([matrix_dim1])

    
    weights = tf.Variable(w_init, name='weights')
    biases = tf.Variable(b_init, name='biases')
    
    return weights, biases

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
    bn_layer tf.nn.batch_normalization(input, mu, var, beta, alpha, 1e-10)
    
    return bn_layer

def add_fully_connected_layer(input, input_size, output_size, activation_fn=None):
    """
        Creates a fully connected layer: input -> [affine + activiation + batch normalization] -> output.
        
        Input
        =====
        - input: the input tensor.
        - input_size: the number of features in the input tensor.
        - output_size: the size of the output
        - activation_fn: the activiation function.
        
        Output
        ======
        - output tensor for the fully connected layer
    """
    w, b = affine_layer_variables(input_size, output_size)
    layer = tf.matmul(input, w) + b

    if(activation_fn):
        layer = activation_fn(layer)

    # batch normalization
    layer = batch_normalization(layer)
    return layer

def get_histogram_summary(num_hidden):
    """
        Creates summarizers for the histograms of weight distribution in different layers.
        
        Input
        =====
        - num_hidden: the number of hidden layers
        
        Output
        ======
        - tf summary
    """
    summary = tf.summary.histogram('softmax_linear/weights', get_variable('softmax_linear/weights:0'))
    for i in range(num_hidden):
        var_name = 'hidden%d/weights:0' % (i + 1)
        cur_summary = tf.summary.histogram(var_name[:-2], get_variable(var_name))
        summary = tf.summary.merge([summary, cur_summary])
    return summary

def get_variable(var_name):
    """
        Returns tf variable given its name.
        
        Input
        =====
        - var_name: the name of the variable.
        
        Output
        ======
        - tf variable
    """
    return tf.get_default_graph().get_tensor_by_name(var_name)
