"""
    Utilities for loading and preprocessing the datasets.
"""

import os
import csv
import pickle
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

import config

def map_labels(labels):
    """
        Maps every label from the string format to a number.

        Input
        =====
        - labels: np array for labels in the string format.

        Output
        ======
        - enum_labels: np array for labels after mapping.
    """
    
    return np.vectorize(config.activity_enum.get)(labels)

def maybe_pickle(filename, force=False):
    """
        Pickles the dataset named 'filename' that exists in the directory 'datasets'.
        This file name must have the extension 'csv'.

        Labels are converted from the string format to the following enumeration:
        1- WALKING      2- WALKING_UPSTAIRS     3- WALKING_DOWNSTAIRS
        4- SITTING      5- STANDING             6- LAYING

        Input
        =====
        - filename: the name of the csv file to be pickled.
        - force: if set to true, the dataset will be pickled even if it already exists.
    """
    filename = config.DATA_PATH + filename
    pickle_file  = os.path.splitext(filename)[0] + '.pickle'
    if(os.path.exists(pickle_file) and not force):
        print '%s already exists. Skipping pickling.' % pickle_file
    else:
        with open(filename, 'rb') as csvfile:
            dataset = np.array(list(csv.reader(csvfile)))[1:,:]
            dataset_features = dataset[:, :561].astype(np.float32)
            dataset_labels = tf.one_hot(map_labels(dataset[:, 562]),
                                        config.NUM_ACTIVITIES).eval(session=tf.Session())
        
        perm = np.random.permutation(dataset_features.shape[0])
        dataset_features = dataset_features[perm]
        dataset_labels = dataset_labels[perm]
               
        print 'Pickling', pickle_file, '...'
        with open(pickle_file, 'wb') as pf:
            save = {
                'dataset_features': dataset_features,
                'dataset_labels': dataset_labels
            }
            pickle.dump(save, pf, pickle.HIGHEST_PROTOCOL)
            print pickle_file, 'pickled successfully!'

def load_dataset(filename):
    """
        Loads the pickled dataset named 'filename' that exists in the directory 'datasets'.

        Input
        =====
        - filename: the name of the dataset to be loaded (train/test)

        Output
        ======
        A 3-tuple that contains numpy matrices for N datapoints:
        - dataset_features: The features of datapoints with shape (N, 561)
        - dataset_subjectID: An identifier for the subject, who carried out the
            experiment, with shape (N,)
        - dataset_labels: the activity label with shape (N, 6) as one-hot encoding
    """
    filename = 'datasets/' + filename + '.pickle'
    with open(filename, 'rb') as pf:
        save = pickle.load(pf)
        features = save['dataset_features']
        labels = save['dataset_labels']
    return features, labels

def get_batch(dataset, batch_size=None, indices=None):
    """
        Samples a batch from the input dataset.

        Input
        =====
        - dataset: a pair of the features and labels of the input dataset.
        - batch_size: if the sampled batch will be random and indices is None.
        - indices: the indicies of the sampled batch, optional.

        Output
        ======
        - The sampled batch.
    """
    features, labels = dataset
    
    if(indices is None):
        indices = np.random.choice(features.shape[0], batch_size, replace=False)
    return features[indices], labels[indices]
