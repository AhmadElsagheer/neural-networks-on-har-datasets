"""
    Deep Neural Network Model.
"""
import tensorflow as tf
import data_utils as du
import layer_utils as lu
import config
import time
import os

class NeuralNetwork():
  
    def __init__(self, hidden_dim=config.hidden_dim, optimizer=config.optimizer, 
                 learning_rate=config.learning_rate, num_steps=config.num_steps,
                 batch_size=config.batch_size):
        """
            Creates the neural network classifier with the configuration in the input.
            
            All configuration parameters are optional. Default values are found in config.py
            
            Input
            =====
            - hidden_dim: a tuple for the dimensions of every hidden layer in the network.
            - optimizer: the name of the optimizer used for training.
            - learning_rate: the learning rate.
            - num_steps: the number of training steps.
            - batch_size: the size of the batch used for per training step.
        
        """
        exec("opt = tf.train." + optimizer)
        
        self.hidden_dim = hidden_dim
        self.optimizer = opt
        self.learning_rate = learning_rate
        
        self.num_steps = num_steps
        self.batch_size = batch_size
        
        self.sess = None
        self.acc = None
        self.features = None
        self.labels = None
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

        # All of the built ops will be associated with the default global graph instance
        with tf.Graph().as_default():

            # Create a model instance
            features, labels = lu.input_placeholders()

            logits = self.inference(features, self.hidden_dim)

            loss = self.loss(logits, labels)

            train_step = self.training(loss)

            accuracy = self.evaluation(logits, labels)
            
            conf_matrix = tf.confusion_matrix(tf.argmax(labels, 1), tf.argmax(logits, 1), num_classes=config.NUM_ACTIVITIES)

            # Create summarizers
            loss_summary = tf.summary.scalar('loss', loss)
            weights_summary = lu.get_histogram_summary(len(self.hidden_dim))
            train_acc_summary = tf.summary.scalar('train_acc', accuracy)
            val_acc_summary = tf.summary.scalar('val_acc', accuracy)
            train_summary = tf.summary.merge([loss_summary, weights_summary, train_acc_summary])

            # Create Model Saver
            saver = tf.train.Saver()
            
            # Create a session
            sess = tf.Session()
            summary_writer = tf.summary.FileWriter(out_dir, sess.graph)

            # train the model
            init_vars = tf.global_variables_initializer()
            sess.run(init_vars)
         
            for step in range(self.num_steps):
                start_time = time.time()
                X_batch, y_batch = du.get_batch(train, batch_size=self.batch_size)
                feed_dict = {features:X_batch, labels:y_batch}

                _, loss_value, summary_str = sess.run([train_step, loss, loss_summary], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                
                
                if(step % 100 == 0):
                    # evaluate model on train dataset
                    feed_dict = {features:X_train, labels:y_train}
                    _, summary_str = sess.run([accuracy, train_summary], feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    
                    # evaluate model on validation dataset
                    if(val):
                        feed_dict = {features:X_val, labels:y_val}
                        _, summary_str = sess.run([accuracy, val_acc_summary], feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        
                    if(verbose):
                        duration = time.time() - start_time
                        print 'Step %d, loss = %.3f (%.3f sec)' % (step, loss_value, duration)
                        
                        summary_writer.flush()
                
        self.sess = sess
        self.acc = accuracy
        self.conf_matrix = conf_matrix
        self.features = features
        self.labels = labels
        
        
        # Save the model
        save_path = out_dir + 'model/'
        os.mkdir(save_path)       
        saver.save(sess, save_path + 'model.ckpt')


    def inference(self, features, hidden_dim):
        """
            Builds the model graph for running the network forward to make predictions.

            Input
            =====
            - features: features placeholder.
            - hidden_dim: tuple defining the number of nodes for each hidden layer.

            Output
            ======
            - logits: output tensor with the computed logits.
        """
        num_hidden = len(hidden_dim)
        
        # Hidden Layers
        # =============
        input_size = config.INPUT_DIM
        layer_input = features
        for i in range(num_hidden):
            with tf.name_scope('hidden%d' % (i + 1)):
                layer_output = lu.add_fully_connected_layer(layer_input,
                                    input_size, hidden_dim[i], tf.nn.relu)
                input_size = hidden_dim[i]
                layer_input = layer_output


        # Softmax Output Layer
        # ====================
        with tf.name_scope('softmax_linear'):
            logits = lu.add_fully_connected_layer(layer_input, input_size, config.NUM_ACTIVITIES)

        return logits

    def loss(self, logits, labels):
        """
            Computes the cross-entropy loss.

            Input
            =====
            - logits: kogits tensor from inference(), float - [batch_size, NUM_ACTIVITIES]
            - labels: kabels tensor for correct labels of the datapoints, float - [batch_size]

            Output
            ======
            - loss: cross-entropy loss tensor, float.
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                            labels=labels, logits=logits, name='xentropy')
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
        optimizer = self.optimizer(self.learning_rate, momentum=0.7, use_locking=True)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        return optimizer.minimize(loss, global_step=global_step)

    def evaluation(self, logits, labels):
        """
            Evaluates the quality of the logits at predicting labels.

            Input
            =====
            - logits: logits tensor, float - [batch_size, NUM_ACTIVITIES]
            - labels: labels tensor, float - [batch_size, NUM_ACTIVITIES]

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