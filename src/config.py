import os

# Set up the paths to the train and test data
dirname = os.path.realpath('')
csv_train_examples = dirname + "\\data\\e2e-dataset\\trainset.csv"          #42,062 entries with label
csv_test_examples = dirname + "\\data\\e2e-dataset\\testset_w_refs.csv"

# Set the check point save location
checkpoint_path = "./checkpoints/train"
results_path = "./results"

########## Variables ##########
VOCAB_SIZE = 2**13          # Max size of the dictionary (from 1 to the number for encoding)
FILTER_BY_LENGTH = False
EXAMPLES_MAX_LENGTH = 200    # To keep this example small and relatively fast, drop examples with a length of over 40 tokens.
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 43_000     # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
EPOCHS = 0

########## Hyperparameters ###########
num_layers = 6          # 4 vs 6
d_model = 512           # 128 vs 512
num_heads = 8           # 8
dff = 2048               # 512 vs 2048
dropout_rate = 0.1      # 0.1

# Save the config to a text file
def save_config():
    config_file = open('./checkpoints/config.txt', 'w+')
    config_file.write('vocab_szie = {}\n'.format(VOCAB_SIZE))
    config_file.write('filter_by_length = {}\n'.format(FILTER_BY_LENGTH))
    config_file.write('examples_max_length = {}\n'.format(EXAMPLES_MAX_LENGTH))
    config_file.write('batch_size = {}\n'.format(BATCH_SIZE))
    config_file.write('shuffle_buffer_size = {}\n'.format(SHUFFLE_BUFFER_SIZE))
    config_file.write('epochs = {}\n'.format(EPOCHS))

    config_file.write('num_layers = {}\n'.format(num_layers))
    config_file.write('d_model = {}\n'.format(d_model))
    config_file.write('num_heads = {}\n'.format(num_heads))
    config_file.write('dff = {}\n'.format(dff))
    config_file.write('dropout_rate = {}\n'.format(dropout_rate))

    config_file.close()