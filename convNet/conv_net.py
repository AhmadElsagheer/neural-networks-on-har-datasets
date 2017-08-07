"""
    Convolutional Neural Network Model.
"""
import tensorflow as tf
from utils import data_utils as du
from utils import layer_utils as lu
import config
import time

class ConvNetClassifier():
    
    def __init__(self, conv_layers_dims=config.conv_layers_dims, full_layers_dims=config.full_layers_dims, 
                 optimizer=config.optimizer, learning_rate=config.learning_rate, num_steps=config.num_steps,
                 batch_size=config.batch_size):
         """
            Creates the cnn classifier with the configuration in the input.
            
            All configuration parameters are optional. Default values are found in config.py
            
            Input
            =====
            - con_layers_dims: a pair of dimensions for the two convolutional layers.
            - fully_layers_dims: a pair of dimensions for the two hidden fully connected layer.
            - optimizer: the name of the optimizer used for training.
            - learning_rate: the learning rate.
            - num_steps: the number of training steps.
            - batch_size: the size of the batch used for per training step.
        
        """
        exec("opt = tf.train." + optimizer)
        
        self.conv_layers_dims = conv_layers_dims
        self.full_layers_dims = full_layers_dims
        
        self.optimizer = opt
        self.learning_rate = learning_rate
        
        self.num_steps = num_steps
        self.batch_size = batch_size
        
        self.features = None
        self.labels = None
        self.sess = None
        self.acc = None
        self.conf_matrix = None

    def fit(self, train, val=None, out_dir='log/', verbose=False):
         """
            Trains the model using the input training set.
            
            Summarizes the loss and training (and validation) accuracies.
            
            Input
            =====
            - train: a pair for the features and labels of the training set.
            - val: a pair for the features and labels of the validation set, optional.
            - out_dir: the directory location for summary files relative to root directory, optional.
            - verbose: a boolean flag. If set to true, loss values and execution time is printed to the console, optional.
        """
        X_train, y_train = train
        if(val):
            X_val, y_val = val
        tot_time = 0
        # All of the built ops will be associated with the default global graph instance
        with tf.Graph().as_default():

            # Create the model instance
            features, labels = lu.input_placeholders()
            
            # Dropout
            # keep_prob = tf.placeholder(tf.float32)

            logits = self.inference(features)

            loss = self.loss(logits, labels)

            train_step = self.training(loss)

            accuracy = self.evaluation(logits, labels)

            conf_matrix = tf.confusion_matrix(tf.argmax(labels, 1), tf.argmax(logits, 1), num_classes=config.num_activities)
            
            # Create summarizers
            loss_summary = tf.summary.scalar('loss', loss)
            acc_summary = tf.summary.scalar('accuracy', accuracy)
            
            # Create the session
            sess = tf.Session()
            
            # Create Model Saver
            saver = tf.train.Saver()

            # Train the model
            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter(out_dir, sess.graph)
            val_writer = tf.summary.FileWriter(out_dir+'val/', sess.graph)
            
            for step in range(self.num_steps):
                start_time = time.time()
                X_batch, y_batch = du.get_batch(train, batch_size=self.batch_size)
                feed_dict = {features:X_batch, labels:y_batch}

                _, loss_value, summary_str = sess.run([train_step, loss, loss_summary], feed_dict=feed_dict)
                train_writer.add_summary(summary_str, step)
                
                duration = time.time() - start_time
                tot_time += duration
                if(step % 100 == 0):
                    # evaluate model on train dataset
                    feed_dict = {features:X_train, labels:y_train}
                    _, summary_str = sess.run([accuracy, acc_summary], feed_dict=feed_dict)
                    train_writer.add_summary(summary_str, step)
                    
                    # evaluate model on validation dataset
                    if(val):
                        feed_dict = {features:X_val, labels:y_val}
                        _, summary_str = sess.run([accuracy, acc_summary], feed_dict=feed_dict)
                        val_writer.add_summary(summary_str, step)
                        
                    if(verbose):
                        print 'Step %d, loss = %.3f (%.3f sec)' % (step, loss_value, duration)
                    val_writer.flush()    
                train_writer.flush()
        if(verbose):
            print 'total time = %.3f sec' % tot_time
        self.sess = sess
        self.acc = accuracy
        self.conf_matrix = conf_matrix
        self.features = features
        self.labels = labels

    def inference(self, features):
        """
            Builds the conv net graph for running the network forward to make predictions.

            Input
            =====
            - features: features placeholder.

            Output
            ======
            - logits: output tensor with computed logits.
        """
        N = features.shape[0]
        M = config.sampling_rate
        C = config.num_channels
        V = config.num_activities
        F1, F2 = config.conv_layers_dims
        H1, H2 = config.full_layers_dims
        FW1, FW2 = config.filters_width

        # Convolutional Layer 1
        with tf.name_scope('conv_layer_1'):
            filter_tensor1 = lu.weight_variable((FW1, C, F1))
            conv_layer1 = lu.convolutional_layer(features, filter_tensor1, tf.nn.relu)

        # Convolutional Layer 2
        with tf.name_scope('conv_layer_2'):
            filter_tensor2 = lu.weight_variable((FW2, F1, F2))
            conv_layer2 = lu.convolutional_layer(conv_layer1, filter_tensor2, tf.nn.relu)

        # Flatten Layer
        flatten_layer = tf.contrib.layers.flatten(conv_layer1)

        # Fully Connected Layer 1
        with tf.name_scope('fc_layer_1'):
            fc_layer1 = lu.fully_connected_layer(flatten_layer, H1, tf.nn.relu)

        # Fully Connected Layer 2
        with tf.name_scope('fc_layer_2'):
            fc_layer2 = lu.fully_connected_layer(fc_layer1, H2, tf.nn.relu)

        # Softmax Ouput Layer
        with tf.name_scope('softmax_linear'):
            logits = lu.fully_connected_layer(fc_layer2, V)

        return logits

    def loss(self, logits, labels):
         """
            Computes the cross-entropy loss.

            Input
            =====
            - logits: kogits tensor from inference(), float - [batch_size, num_activities]
            - labels: kabels tensor for correct labels of the datapoints, float - [batch_size]

            Output
            ======
            - loss: cross-entropy loss tensor, float.
        """ 
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def training(self, loss):
          """
            Creates an optimizer and applies the gradients to all trainable variables.

            Input
            =====
            - loss: loss tensor from loss(), float

            Output
            ======
            - train_optimizer: optimizer tensor for the training to be run in the session.
        """
        optimizer = self.optimizer(self.learning_rate, momentum=0.7)
        return optimizer.minimize(loss)

    def evaluation(self, logits, labels):
        """
            Evaluates the quality of the logits at predicting labels.

            Input
            =====
            - logits: logits tensor, float - [batch_size, num_activities]
            - labels: labels tensor, float - [batch_size, num_activities]

            Output
            ======
            - A scaler tensor with the number of examples (out of the batch size)
            that were predicted correctly.
        """
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct, tf.float32))
    
    def accuracy(self, dataset):
        """
            Returns the accuracy of the model on the input dataset.
            
            Input
            =====
            - dataset: a pair of features and labels for the input dataset.
        """
        X, y = dataset
        feed_dict = {self.features:X, self.labels:y}
        return self.sess.run(self.acc, feed_dict=feed_dict)
    
    def restore(self, sess):
        """
            Sets the session of the current network to a previously saved session.
            
            Input
            =====
            - sess: the previously saved session to be restored.
        """
        self.sess=sess
         
    def confusion_matrix(self, dataset):
        """
            Return the confusion matrix for the input dataset.
            
            Input
            =====
            - Dataset: a pair of features and labels for the input dataset.
        """
        X, y = dataset
        feed_dict = {self.features:X, self.labels:y}
        return self.sess.run(self.conf_matrix, feed_dict=feed_dict)