import tensorflow as tf
import tensorflow_datasets as tfds
import config

# Class for the pipeline
class Input_Pipeline:
    def __init__(self, train_file, test_file):
        self.train_examples = load_examples_into_dataset(train_file)
        self.test_examples = load_examples_into_dataset(test_file)
        self.mr_tokenizer, self.ref_tokenizer = create_tokenizers(self.train_examples)
        self.train_dataset, self.test_dataset = encode_examples(self.train_examples, self.test_examples,
                                                                self.mr_tokenizer, self.ref_tokenizer)

    # Test a tokenizer
    def test_tokenizer(self, tokenizer, sample_string):
        tokenized_string = tokenizer.encode(sample_string)
        print ('Tokenized string is {}'.format(tokenized_string))
        original_string = tokenizer.decode(tokenized_string)
        print ('The original string: {}'.format(original_string))
        assert original_string == sample_string
        # The tokenizer encodes the string by breaking it into subwords if the word is not in its dictionary.
        for ts in tokenized_string:
            print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

    # print an example from a dataset
    def print_dataset_example(self, dataset):
        in_batch, out_batch = next(iter(dataset))
        print(in_batch, out_batch)

# Create a CsvDatasetV2 with defaults of [tf.string]*2
def load_examples_into_dataset(filePath):
    # Defaults describing the data for making a tensorflow-dataset
    defaults = [tf.string]*2

    dataset = tf.data.experimental.CsvDataset(
        filePath,
        defaults,
        header=True)
    return dataset

# Create tokenizers
def create_tokenizers(train_examples):
    mr_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (mr.numpy() for mr, ref in train_examples), target_vocab_size=config.VOCAB_SIZE)
    ref_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (ref.numpy() for mr, ref in train_examples), target_vocab_size=config.VOCAB_SIZE)
    return mr_tokenizer, ref_tokenizer

# encode and update the datasets
def encode_examples(train_examples, test_examples, mr_tokenizer, ref_tokenizer):
    # Add a start and end token to the mr and ref
    def encode(mr, ref):
        mr = [mr_tokenizer.vocab_size] + mr_tokenizer.encode(mr.numpy()) + [mr_tokenizer.vocab_size+1]
        ref = [ref_tokenizer.vocab_size] + ref_tokenizer.encode(ref.numpy()) + [ref_tokenizer.vocab_size+1]
        return mr, ref

    # So you can't .map this function directly: You need to wrap it in a tf.py_function. 
    # The tf.py_function will pass regular tensors (with a value and a .numpy() method to access it),
    # to the wrapped python function.
    def tf_encode(mr, ref):
        result_mr, result_ref = tf.py_function(encode, [mr, ref], [tf.int64, tf.int64])
        result_mr.set_shape([None])
        result_ref.set_shape([None])
        return result_mr, result_ref

    # Filter the data by length
    def filter_max_length(x, y, max_length=config.EXAMPLES_MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


    train_dataset = train_examples.map(tf_encode)
    #padded_shape = ([None],[None])
    if config.FILTER_BY_LENGTH :
        train_dataset = train_dataset.filter(filter_max_length)
        #padded_shape = (config.BATCH_SIZE, config.EXAMPLES_MAX_LENGTH)

    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(config.SHUFFLE_BUFFER_SIZE).padded_batch(config.BATCH_SIZE, padded_shapes=([None],[None]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) #PrefetchedDataset its an optimization

    test_dataset = test_examples.map(tf_encode)
    if config.FILTER_BY_LENGTH :
        test_dataset = test_dataset.filter(filter_max_length)
        
    test_dataset = test_dataset.padded_batch(config.BATCH_SIZE, padded_shapes=([None], [None]))

    return train_dataset, test_dataset

# interface
def create_input_pipeline():
    ipl = Input_Pipeline(config.csv_train_examples, config.csv_test_examples)
    return ipl