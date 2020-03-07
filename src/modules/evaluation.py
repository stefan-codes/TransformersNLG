import tensorflow as tf
import matplotlib.pyplot as plt
import config
import csv
import os

from modules.masking import create_masks
from modules.loss_and_metrics import get_bleu_score

# Evaluates a signle input
def evaluate_single_input(mr_string, input_pipeline, transformer):
  start_token = [input_pipeline.mr_tokenizer.vocab_size]
  end_token = [input_pipeline.mr_tokenizer.vocab_size + 1]
  
  # inp sentence is input data, hence adding the start and end token
  mr_string = start_token + input_pipeline.mr_tokenizer.encode(mr_string) + end_token
  encoder_input = tf.expand_dims(mr_string, 0)
  
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [input_pipeline.ref_tokenizer.vocab_size]
  output = tf.expand_dims(decoder_input, 0)

  # update to a while loop -> Keep it as for for now so it works for limiting the examples.
  # For running all of the data, put relatively large number e.g. 200.
  # Thats not a problem since we end thesentence at the end token anyway
  for i in range(config.EXAMPLES_MAX_LENGTH):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if predicted_id == input_pipeline.ref_tokenizer.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights

    # concatentate the predicted_id to the output which is given to the decoder as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

# Plot the attention weights
def plot_attention_weights(attention, sentence, result, layer, input_pipeline):
  fig = plt.figure(figsize=(16, 8))

  sentence = input_pipeline.mr_tokenizer.encode(sentence)
  attention = tf.squeeze(attention[layer], axis=0)

  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # plot the attention weights
    ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))
    ax.set_ylim(len(result)-1.5, -0.5)
    ax.set_xticklabels(['<start>']+[input_pipeline.mr_tokenizer.decode([i]) for i in sentence]+['<end>'], fontdict=fontdict, rotation=90)
    ax.set_yticklabels([input_pipeline.ref_tokenizer.decode([i]) for i in result if i < input_pipeline.ref_tokenizer.vocab_size], fontdict=fontdict)
    ax.set_xlabel('Head {}'.format(head+1))

  plt.tight_layout()
  plt.show()

# Generate a sentence
def generate_sentence(mr_string, input_pipeline, transformer, plot=''):
  result, attention_weights = evaluate_single_input(mr_string, input_pipeline, transformer)
  predicted_sentence = input_pipeline.ref_tokenizer.decode([i for i in result if i < input_pipeline.ref_tokenizer.vocab_size])

  if plot:
    plot_attention_weights(attention_weights, mr_string, result, plot)

  return predicted_sentence

# Get the bleu score for examples
def evaluate_test_data(transformer, input_pipeline, name_of_file, num_of_examples):
   # Write to a csv
  if not os.path.isdir(config.results_path):
    os.makedirs(config.results_path)
  file_name = '{}/{}.csv'.format(config.results_path, name_of_file)

  with open(file_name, 'a+', newline='') as csv_file:
    header_names = ['mr', 'ref', 'prediction', 'Bleu']
    # header_names = ['mr', 'ref', 'prediction', '1-gram', '2-gram', '3-gram', '4-gram']

    the_writer = csv.DictWriter(csv_file, fieldnames=header_names)
    the_writer.writeheader()

    counter = 0
    for entry in input_pipeline.test_examples:
      # Break the entry into mr and ref
      mr, ref = entry
      mr_example = str(mr.numpy(), 'utf-8')
      ref_example = str(ref.numpy(), 'utf-8')

      # Get a prediction
      predicted_sentence = generate_sentence(mr_example, input_pipeline, transformer)

      # Get the bleu scores
      bleu_score = get_bleu_score(predicted_sentence, ref_example)

      # write them to the file
      the_writer.writerow({'mr' : mr_example, 'ref' : ref_example, 'prediction' : predicted_sentence, 'Bleu' : '%.2f' % bleu_score})
      # the_writer.writerow({'mr' : mr_example, 'ref' : ref_example, 'prediction' : predicted_sentence, 
      #                       '1-gram' : '%.4f' % one_gram, '2-gram' : '%.4f' % two_gram, '3-gram' : '%.4f' % three_gram, '4-gram' : '%.4f' % four_gram})

      counter = counter + 1
      if counter == num_of_examples:
        break
      if counter % 10 == 0 :
        print(counter)
  # at the end
  print('All examples evaluated.')
