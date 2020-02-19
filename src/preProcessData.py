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

# Make numpy values easier to read - whatever that means?
#np.set_printoptions(precision=3, suppress=True)

defaults = [tf.string] * 2
# CsvDataset
def makeDataset(filePath):
    dataset = tf.data.experimental.CsvDataset(
      filePath,
      defaults,
      header=True)
    return dataset

# make_csv_dataset
"""
def datasetViaMakeCsvDataset(filePath, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      filePath,
      batch_size=1,
      label_name='ref',
      header=True,
      #num_epochs=1,
      #shuffle=True,                     # var
      #shuffle_buffer_size=10000,        # var
      #prefetch_buffer_size=10,          # Recommended value is the number of batches consumed per training step 
      #ignore_errors=True,
      **kwargs)
  return dataset
"""

# load the data
#trainDataset = datasetViaMakeCsvDataset(csv_train_file_path) #PrefetchDataset
trainDataset = makeDataset(csv_train_file_path) #CsvDatasetV2
#trainDataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1]) #TensorSliceDataset

#colNames = ['input','output']

def prepareData(*vals):
  #res = vals[0].encode('utf-8')
  features = tf.convert_to_tensor(vals[0])
  classLabel = tf.convert_to_tensor(vals[1])

  return features, classLabel


#temp = dict(zip(colNames, tf.convert_to_tensor(trainDataset)))

#trainDataViaCsvDataset = prepareData(trainDataViaCsvDataset)
#trainDataset = trainDataset.map(prepareData).batch(1)

for element in trainDataset:
  print(element[0])
  break

data_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus((element[0].numpy() for element in trainDataset), target_vocab_size=2**13)

for element in trainDataset:
  print(element[0])
  break

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
