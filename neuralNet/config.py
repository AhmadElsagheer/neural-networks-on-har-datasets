"""
    Settings, constants and parameters for the neural net.
"""

# The number of activities to be classified by the model.
NUM_ACTIVITIES = 6

# The number of features for each datapoint.
INPUT_DIM = 561

# Enumeration for the activities
activity_enum = {
        'WALKING':0,
        'WALKING_UPSTAIRS':1,
        'WALKING_DOWNSTAIRS':2,
        'SITTING':3,
        'STANDING':4,
        'LAYING':5,
}

# Defaults for the neural network
# ===============================

# The dimensions of every hidden layer in the net.
hidden_dim = (1024, 512, 64)

# The learning rate
learning_rate = 0.0001

# The number of steps for training
num_steps = 5000

# The size of the batch used for per training step
batch_size = 256

# The optimizer used for training
optimizer = 'RMSPropOptimizer'

# Location of dataset files relative to root directory
DATA_PATH = 'datasets/'