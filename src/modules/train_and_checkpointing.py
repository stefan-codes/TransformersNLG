import time
import tensorflow as tf
import config
from modules.masking import create_padding_mask
from modules.masking import create_look_ahead_mask
from modules.optimizer import CustomSchedule
from modules.loss_and_metrics import loss_function

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

def train_the_transformer(transformer, train_dataset):
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
    for (batch, (inp, tar)) in enumerate(train_dataset):
      train_step(inp, tar)
  
      if batch % 50 == 0:
          print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
    
    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      config.save_config()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
    
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


  








