from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Get the path to the files
dirname = os.path.realpath('')
csv_train_file_path = dirname + "\\data\\e2e-dataset\\trainset.csv"
#csv_test_file_path = dirname + "\\data\\e2e-dataset\\testset_w_refs.csv"
#csv_train_file_path = dirname + "\\data\\e2e-dataset\\trainset3.csv"

# CsvDataset
def makeDataset(filePath):
  dataset = tf.data.experimental.CsvDataset(
    filePath,
    [tf.string]*2,
    header=True)
  return dataset

# load the data
trainDataset = makeDataset(csv_train_file_path) #CsvDatasetV2

"""
# Work on the data to updat in a row
def prepareData(*vals):
  #res = vals[0].encode('utf-8')
  features = tf.convert_to_tensor(vals[0])
  classLabel = tf.convert_to_tensor(vals[1])
  #temp = dict(zip(colNames, tf.convert_to_tensor(trainDataset)))
  return features, classLabel
#trainDataset = trainDataset.map(prepareData).batch(64)
"""

data_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((i.numpy() for i, o in trainDataset), target_vocab_size=2**13)


sample_string = 'Transformer is awesome.'
tokenized_string = data_tokenizer.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))
original_string = data_tokenizer.decode(tokenized_string)
print ('The original string: {}'.format(original_string))
assert original_string == sample_string


print('#################################################################################################')
print(list(trainDataset.take(1)))
print('#################################################################################################')

"""
def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))
show_batch(raw_train_data)
"""

# Tokenizers
"""
#data_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((mr.numpy() for mr in train_dataset), target_vocab_size=2**13)
#tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

sample_string = 'Transformer is awesome.'
tokenized_string = data_tokenizer.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))
original_string = data_tokenizer.decode(tokenized_string)
print ('The original string: {}'.format(original_string))
assert original_string == sample_string
"""


print("done")
