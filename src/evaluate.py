import tensorflow as tf
import config as c
#import matplotlib.pyplot as plt
from train_and_checkpointing import create_masks

def evaluate(mr, mr_tokenizer, ref_tokenizer, transformer):
  start_token = [mr_tokenizer.vocab_size]
  end_token = [ref_tokenizer.vocab_size + 1]
  
  # inp sentence is input data, hence adding the start and end token
  mr = start_token + mr_tokenizer.encode(mr) + end_token
  encoder_input = tf.expand_dims(mr, 0)
  
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [mr_tokenizer.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(c.EXAMPLES_MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == ref_tokenizer.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

"""
def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))
  
  sentence = tokenizer_in.encode(sentence)
  
  attention = tf.squeeze(attention[layer], axis=0)
  
  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # plot the attention weights
    ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))
    ax.set_ylim(len(result)-1.5, -0.5)
    ax.set_xticklabels(['<start>']+[tokenizer_in.decode([i]) for i in sentence]+['<end>'], fontdict=fontdict, rotation=90)
    ax.set_yticklabels([tokenizer_out.decode([i]) for i in result if i < tokenizer_out.vocab_size], fontdict=fontdict)
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show()
"""

def generate_sentence(mr, ref, mr_tokenizer, ref_tokenizer, transformer, plot=''):
  result, attention_weights = evaluate(mr, mr_tokenizer, ref_tokenizer, transformer)
  
  predicted_sentence = ref_tokenizer.decode([i for i in result if i < ref_tokenizer.vocab_size])  

  print('Meaning Representation: {}'.format(mr))
  print('Predicted translation: {}'.format(predicted_sentence))
  print('Reference: {}'.format(ref))
  
  #if plot:
   # plot_attention_weights(attention_weights, sentence, result, plot)
