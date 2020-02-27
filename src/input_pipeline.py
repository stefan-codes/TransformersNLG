import tensorflow as tf
import tensorflow_datasets as tfds
import config as c

# Create a CsvDatasetV2 with defaults of [tf.string]*2
def make_dataset(filePath, defaults):
    dataset = tf.data.experimental.CsvDataset(
        filePath,
        defaults,
        header=True)
    return dataset

# Load the examples from the data
def load_dataset_examples(train_file, test_file, defaults):
    train_examples = make_dataset(train_file, defaults)
    test_examples = make_dataset(test_file, defaults)
    return train_examples, test_examples

# Create tokenizers
def create_tokenizers(train_examples):
    mr_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (mr.numpy() for mr, ref in train_examples), target_vocab_size=c.VOCAB_SIZE)
    ref_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (ref.numpy() for mr, ref in train_examples), target_vocab_size=c.VOCAB_SIZE)
    return mr_tokenizer, ref_tokenizer

# Test a tokenizer
def test_tokenizer(tokenizer):
    sample_string = 'Transformer is awesome.'
    tokenized_string = tokenizer.encode(sample_string)
    print ('Tokenized string is {}'.format(tokenized_string))
    original_string = tokenizer.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))
    assert original_string == sample_string
    # The tokenizer encodes the string by breaking it into subwords if the word is not in its dictionary.
    for ts in tokenized_string:
        print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

# Filter the data by length
def filter_max_length(x, y, max_length=c.EXAMPLES_MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

mr_tokenizer = None
ref_tokenizer = None

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

# encode and update the datasets
def encode_datasets(train_examples, test_examples, mr_tok, ref_tok):
    global mr_tokenizer, ref_tokenizer
    mr_tokenizer = mr_tok
    ref_tokenizer = ref_tok

    train_dataset = train_examples.map(tf_encode)

    print(next(iter(train_dataset)))

    if c.FILTER_BY_LENGTH :
        train_dataset = train_dataset.filter(filter_max_length)
        c.update_padded_shape()

    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    print(type(train_dataset))
    train_dataset = train_dataset.shuffle(c.SHUFFLE_BUFFER_SIZE).padded_batch(c.BATCH_SIZE, padded_shapes=c.padded_shape)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE) #PrefetchedDataset its an optimization

    #test_dataset = [tf_encode(mr, ref, mr_tokenizer, ref_tokenizer) for mr, ref in test_examples]
    test_dataset = test_examples.map(tf_encode)
    if c.FILTER_BY_LENGTH :
        test_dataset = test_dataset.filter(filter_max_length)
        
    test_dataset = test_dataset.padded_batch(c.BATCH_SIZE, padded_shapes=c.padded_shape)

    return train_dataset, test_dataset


# print result
#in_batch, out_batch = next(iter(test_dataset))
#print(in_batch, out_batch)
