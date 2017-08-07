"""
    Utilities for preprocessing and loading the dataset.
"""
from sklearn import preprocessing
from shutil import copyfile
import pandas as pd
import numpy as np
import pickle
import config
import time
import os

def split_datafiles(path):
    """
        Splits data files into folders, one for each participant such that each folder contains 
        this participant's csv files one for each activity record

        All csv files exist in the folder 'path' and each csv file is named x_y.csv where:
        - x is the participant id
        - y is the activity index

        A new directory will be created at 'path/../raw/' and each participant will have a folder
        with its id. In each folder will be csv files named with the activity labels as in config.py

        Input
        =====
        path: path to folder with csv files.
    """
    new_path = path + '../raw/'
    csv_files = os.listdir(path)
    for csv_file in csv_files:
        ids = os.path.splitext(csv_file)[0].split('_')
        subjectId = ids[0]
        label = config.index_label[int(ids[1])]
        
        src = path + csv_file
        dest = new_path + subjectId + '/' + label + '.csv'
        if not os.path.exists(os.path.dirname(dest)):
            os.makedirs(os.path.dirname(dest))
        copyfile(src, dest)
    print 'Data files have been splitted Successfully.'
    print 'Number of participants = %d' % len(os.listdir(new_path))
        
def maybe_preprocess_dataset(test_size, xactivities=None, force=False):
    """
        Preprocess the raw EMG data to be manipulated by the conv net.

        In the dataset folder, there is a folder for each participant. Each participant
        folder contains csv files, one for each activity. Each csv file has time frames
        associated with the corresponding EMG signals values captured during performing 
        the corresponding activity.

        Each activity record (whole csv file) is sampled to a set of datapoints (examples),
        then all the records are merged into one ndarray, shuffled and finally pickled
        into a single file to be loaded later for manipulation by the conv net.

        The result will be two ndarrays:

        + dataset_features: will have the shape (N, M, C)
            - N is the total number of datapoints
            - M is the sampling rate (i.e. the number of frames per datapoint)
            - C is the number of EMG channels

        + dataset_labels: will have the shape (N, V)
            - N is the total number of datapoints
            - V is the number of activities (classes)
            - Activity labels are numbered from 0 to 9 as described in config.py
            - One-hot encoding is used in this ndarray. In other words, each row will contain 
              value '1' in the correct activity column and value '0' everywhere else
    
        Two pickle files are saved to the hard disk 'train.pickle' and 'test.pickle'
        
        Input
        =====
        - test_size: the number of test participants to be used for test set.
        - xactivities: a tuple of activity indices to be ignored, optional.
        - force: if set to true, the dataset will be processed even if it is already processed, optional.
    """
    included = np.ones(config.num_activities)
    if(xactivities is not None):
        for x in xactivities:
            included[x] = 0
        
    pickle_file = config.dataset_path + 'data.pickle'
    if(os.path.exists(pickle_file) and not force):
        print 'Dataset has already been preprocessed before. Preprocessing skipped.'
        return
    print 'Preprocessing raw data...'
    start_time = time.time()
    
    s_rate = config.sampling_rate
    d_path = config.dataset_path + 'raw/'
        
    X_train = np.empty((0, s_rate, config.num_channels))
    y_train = np.empty((0, config.num_activities))
    
    X_test = np.empty((0, s_rate, config.num_channels))
    y_test = np.empty((0, config.num_activities))
    
    # Open each participant folder
    participant_folders = os.listdir(d_path)
    for pfolder in participant_folders:
        # Open each activity record
        participant_path = os.path.join(d_path, pfolder)
        activity_records = os.listdir(participant_path)
        p_datapoints = 0
        for activity_record in activity_records:
            root, ext = os.path.splitext(activity_record)
            if(ext != '.csv'):
                continue
            activity_label = config.label_index[root]
            if(not included[activity_label]):
                continue
            record_path = os.path.join(participant_path, activity_record)
            frames = pd.read_csv(record_path).iloc[:].as_matrix()
            # Sample the frames to a set of datapoints and add them to the result nd array
            num_frames = frames.shape[0]
            
            num_samples = num_frames // (s_rate // 2) - 1
            p_datapoints += num_samples
            for cur_sample in range(num_samples):
                sample_start = cur_sample * (s_rate // 2)
                sample_end = sample_start + s_rate
                sample_features = np.reshape(frames[sample_start: sample_end], (1, s_rate, config.num_channels))
                sample_label = np.zeros((1, config.num_activities))
                sample_label[0, activity_label] = 1
               
                if(int(pfolder) <= test_size):
                    X_test = np.concatenate((X_test, sample_features))
                    y_test = np.concatenate((y_test, sample_label))
                else:
                    X_train = np.concatenate((X_train, sample_features))
                    y_train = np.concatenate((y_train, sample_label))
           
        print '  - data for participant %s has been perprocessed by sampling %d datapoints.' % (pfolder, p_datapoints)
         
    # Shuffle the dataset
    # print 'Shuffling the datapoints...'
    # permuted_indices = np.random.permutation(num_datapoints)
    # dataset_features = dataset_features[permuted_indices]
    # dataset_labels = dataset_labels[permuted_indices]
    
    print 'Pickling...'
    # Pickle and Save
    pickle_file = config.dataset_path + 'train.pickle'
    with open(pickle_file, 'wb') as pf:
        save = {'features': X_train, 'labels': y_train}
        pickle.dump(save, pf, pickle.HIGHEST_PROTOCOL)
        
    pickle_file = config.dataset_path + 'test.pickle'
    with open(pickle_file, 'wb') as pf:
        save = {'features': X_test, 'labels': y_test}
        pickle.dump(save, pf, pickle.HIGHEST_PROTOCOL)
        
    duration = time.time() - start_time
    print 'Dataset has been preprocessed successfully.'
    print ' - Number of participants = %d' % len(participant_folders)
    print ' - Number of train datatpoints  = %d' % X_train.shape[0]
    print ' - Number of test datatpoints  = %d' % X_test.shape[0]
    print ' - Shape of datapoint = %d frames x %d channels' % (s_rate, config.num_channels)
    print ' - Processing time = %.3f sec' % duration

def load_dataset(filename):
    """
        Loads the dataset to be feeded to the conv net.

        Dataset must be preprocessed before calling this function.

        Input
        =====
        - filename: the name of the dataset in the dataset directory.
        
        Output
        ======
        - A pair of features and labels for the dataset.
        
    """
    pickle_file = config.dataset_path + filename + '.pickle'
    with open(pickle_file, 'rb') as pf:
        save = pickle.load(pf)
        features, labels = save['features'], save['labels']
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
