import time
import tensorflow as tf
import matplotlib.pyplot as plt
import config
import csv
import nltk
import os
nltk.download('punkt')
from modules.masking import create_padding_mask
from modules.masking import create_look_ahead_mask
from modules.optimizer import CustomSchedule
from modules.loss_and_metrics import loss_function
from nltk.translate.bleu_score import sentence_bleu

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask

# Evaluation #
def evaluate(mr, mr_tokenizer, ref_tokenizer, transformer):
  start_token = [mr_tokenizer.vocab_size]
  end_token = [mr_tokenizer.vocab_size + 1]
  
  # inp sentence is input data, hence adding the start and end token
  mr = start_token + mr_tokenizer.encode(mr) + end_token
  encoder_input = tf.expand_dims(mr, 0)
  
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [mr_tokenizer.vocab_size]
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
    if predicted_id == ref_tokenizer.vocab_size+1:
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights

# Get the 4-gram bleu score as percent (0.25, 0.25, 0.25, 0.25)
def get_bleu_score(prediction, reference):
  tokenized_prediction = nltk.word_tokenize(prediction)
  tokenized_reference = nltk.word_tokenize(reference)
  list_references = []
  list_references.append(tokenized_reference)

  bleu_score = sentence_bleu(list_references, prediction, weights=(0.25,0.25,0.25,0.25))

  # one_gram = sentence_bleu(list_references, prediction, weights=(1,0,0,0))
  # two_gram = sentence_bleu(list_references, prediction, weights=(0,1,0,0))
  # three_gram = sentence_bleu(list_references, prediction, weights=(0,0,1,0))
  # four_gram = sentence_bleu(list_references, prediction, weights=(0,0,0,1))

  return bleu_score * 100

def evaluate_transformer(transformer, input_pipeline, name_of_file, num_of_examples):
  # Write to a csv
  # name = datetime.datetime.now()
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
      predicted_sentence = get_prediction(mr_example, input_pipeline, transformer)

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

def get_prediction(mr, input_pipeline, transformer, plot=''):
    result, attention_weights = evaluate(mr, input_pipeline.mr_tokenizer, input_pipeline.ref_tokenizer, transformer)
    predicted_sentence = input_pipeline.ref_tokenizer.decode([i for i in result if i < input_pipeline.ref_tokenizer.vocab_size])  
    # TODO: check what the original sentence if from tensorflow website
    # if plot:
    #     plot_attention_weights(attention_weights, sentence, result, plot)
    return predicted_sentence

def train_the_transformer(transformer, input_pipeline):
  # Create a default optimizer
  learning_rate = CustomSchedule(config.d_model)
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  # setting the train loss and accuracy?
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

  # The @tf.function trace-compiles train_step into a TF graph for faster
  # execution. The function specializes to the precise shape of the argument
  # tensors. To avoid re-tracing due to the variable sequence lengths or variable
  # batch sizes (the last batch is smaller), use input_signature to specify
  # more generic shapes.

  train_step_signature = [    
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
  ]

  @tf.function(input_signature=train_step_signature)
  def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
      predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
      loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)
    # print(transformer.summary())

  # Create the checkpoint manager. This will be used to save checkpoints every n epochs.
  ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, config.checkpoint_path, max_to_keep=5)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print ('Latest checkpoint restored!!')  
    
  for epoch in range(config.EPOCHS):
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> mr, tar -> ref
    for (batch, (inp, tar)) in enumerate(input_pipeline.train_dataset):
      train_step(inp, tar)
  
      if batch % 50 == 0:
          print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    
    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      config.save_config()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
    
    if (epoch + 1) % 20 == 0:
        print('Evaluating the model...')
        evaluate_transformer(transformer, input_pipeline, 'results', 10)

    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))






