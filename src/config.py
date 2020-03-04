import os

# Set up the paths to the train and test data
dirname = os.path.realpath('')
csv_train_examples = dirname + "\\data\\e2e-dataset\\trainset.csv"          #42,062 entries with label
csv_test_examples = dirname + "\\data\\e2e-dataset\\testset_w_refs.csv"

# Set the check point save location
checkpoint_path = "./checkpoints/train"

########## Variables ##########
VOCAB_SIZE = 2**13          # Max size of the dictionary (from 1 to the number for encoding)
FILTER_BY_LENGTH = True
EXAMPLES_MAX_LENGTH = 15    # To keep this example small and relatively fast, drop examples with a length of over 40 tokens.
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 43_000     # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
EPOCHS = 0

########## Hyperparameters ###########
num_layers = 6          # 4 vs 6
d_model = 512           # 128 vs 512
num_heads = 8           # 8
dff = 2048               # 512 vs 2048
dropout_rate = 0.1      # 0.1