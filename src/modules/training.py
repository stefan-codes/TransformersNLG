import time
import tensorflow as tf
import config

from modules.optimizer import CustomSchedule
from modules.loss_and_metrics import get_mean_val_loss
from modules.loss_and_metrics import get_train_loss

# Interface to train the transformer
def train_the_transformer(transformer, input_pipeline):
  # Initialize the learning rate
  learning_rate = CustomSchedule(config.d_model)
  # Create a Adam optimizer with the learning rate
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  # mean for the batch/epoch
  mean_train_loss = tf.keras.metrics.Mean(name='train_loss')
  #train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

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
    with tf.GradientTape() as tape:
      train_loss = get_train_loss(inp, tar, transformer)
      # Originally the next to lines were outsite the with indentation
      gradients = tape.gradient(train_loss, transformer.trainable_variables)    
      optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    mean_train_loss(train_loss)
    #train_accuracy(tar_real, predictions)

  # Create the checkpoint manager. This will be used to save checkpoints every n epochs.
  ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, config.checkpoint_path, max_to_keep=10)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print ('Latest checkpoint restored!!')  

  should_early_stop = False
  best_val_loss = 999
  epoch_since_last_update = 0

  for epoch in range(config.EPOCHS):
    start = time.time()
    mean_train_loss.reset_states()
    #train_accuracy.reset_states()

    # inp -> mr, tar -> ref
    for (batch, (inp, tar)) in enumerate(input_pipeline.train_dataset):
      train_step(inp, tar)
      config.update_train_steps()

      if batch % 50 == 0:
          print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, mean_train_loss.result()))
    
    if (epoch + 1) % 1 == 0:
      ckpt_save_path = ckpt_manager.save()
      config.save_config()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    # During training if I want to do something, do it here...
    mean_val_loss = get_mean_val_loss(input_pipeline.test_dataset, transformer)
    input_pipeline.shuffle_train_dataset()

    # Write the epoch results to a log file
    config.log('Epoch {} Train Loss {:.4f} Validation Loss {:.4f} Train Steps {} Time Taken {:.4f} secs\n'.format(epoch + 1, mean_train_loss.result() , mean_val_loss.result(), config.train_steps, time.time() - start))

    print ('Epoch {} Train Loss {:.4f} Validation Loss {:.4f}'.format(epoch + 1, mean_train_loss.result() , mean_val_loss.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    # Early stopping here! if we didnt improve the best in 5 epoch, stop
    if best_val_loss - mean_val_loss > 0.001 : 
      best_val_loss = mean_val_loss
      epoch_since_last_update = 0
    else:
      epoch_since_last_update = epoch_since_last_update + 1

    if epoch_since_last_update > 4 :
      print('5 epoch with no update of val loss. Stoping early.')
      break


    

