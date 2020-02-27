#############
# Variables #
#############
VOCAB_SIZE = 2**13          # Max size of the dictionary (from 1 to the number for encoding)
FILTER_BY_LENGTH = True
EXAMPLES_MAX_LENGTH = 40    # To keep this example small and relatively fast, drop examples with a length of over 40 tokens.
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 43_000     # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
EPOCHS = 20
padded_shape = ([2],[None])

###################
# Hyperparameters #
###################
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
dropout_rate = 0.1

def update_padded_shape():
    global padded_shape
    padded_shape = (BATCH_SIZE, EXAMPLES_MAX_LENGTH)