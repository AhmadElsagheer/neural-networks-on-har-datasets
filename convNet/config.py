"""
    Settings, constants and parameters for the conv net.
"""
# The number of activities used in classification
num_activities = 10

# The number of EMG channels
num_channels = 8

# The sampling rate for each activity record
# i.e. The number of frames per datapoint (example)
sampling_rate = 128

# The dataset path relative to the main running file (main.py)
dataset_path = 'dataset/'

# A dictionary that maps every label name to its corresponding index
label_index = {
 'walking'           : 0,
 'standing'          : 1,
 'talking_phone'     : 2,
 'typing_keyboard'   : 3,
 'writing_desk'      : 4, 
 'drawing_board'     : 5,
 'brushing_teeth'    : 6,
 'holding_sphere'    : 7,
 'holding_cylinder'  : 8, 
 'holding_thin_sheet': 9
}

index_label = ['walking', 'standing', 'talking_phone', 'typing_keyboard', 'writing_desk', 'drawing_board', 'brushing_teeth',
              'holding_sphere', 'holding_cylinder', 'holding_thin_sheet']

# Defaults for the cnn classifier
# ===============================


# The name of the optimizer used for training
optimizer = 'RMSPropOptimizer'

# The dimensions of every convolutional layer
conv_layers_dims = (16, 32)

filters_width = (7, 7)

# The dimensions of every hidden fully connected layer in the network
full_layers_dims = (64, 16)

# The learning rate
learning_rate = 0.001

# The number of training steps
num_steps = 10000

# The size of the batch used for per training step
batch_size = 128
