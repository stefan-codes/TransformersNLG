import tensorflow as tf
import nltk
nltk.download('punkt')

from nltk.translate.bleu_score import sentence_bleu
from modules.masking import create_masks

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_mean(loss_)

# Get the 4-gram bleu score as percent (0.25, 0.25, 0.25, 0.25)
# I did research more about it, it is correct. However if we get any n-gram 0 it returns 0
def get_bleu_score(prediction, reference):
  tokenized_prediction = nltk.word_tokenize(prediction)
  tokenized_reference = nltk.word_tokenize(reference)
  list_references = []
  list_references.append(tokenized_reference)

  # bleu_score = sentence_bleu(list_references, prediction, weights=(0.25,0.25,0.25,0.25), smoothing_function=None, auto_reweigh=False)

  one_gram = sentence_bleu(list_references, prediction, weights=(1,0,0,0)) * 100
  two_gram = sentence_bleu(list_references, prediction, weights=(0.5,0.5,0,0)) * 100
  three_gram = sentence_bleu(list_references, prediction, weights=(0.33,0.33,0.33,0)) * 100
  four_gram = sentence_bleu(list_references, prediction, weights=(0.25,0.25,0.25,0.25)) * 100

  return one_gram, two_gram, three_gram, four_gram

# Get the mean validation loss for the epoch - returns float
def get_mean_val_loss(val_dataset, transformer):
  mean_val_loss = tf.keras.metrics.Mean(name='val_loss')

  for(batch, (inp, tar)) in enumerate(val_dataset):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    predictions, _ = transformer(inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
    val_loss = loss_function(tar_real, predictions)
    mean_val_loss(val_loss)

    return mean_val_loss

# Get the train loss for each inp - tar pair
def get_train_loss(inp, tar, transformer):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]

  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
  train_loss = loss_function(tar_real, predictions)

  return train_loss