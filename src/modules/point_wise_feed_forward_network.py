import tensorflow as tf

# Point wise feed forward network consists of two fully-connected layers with a ReLU activation in between.
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

# Example
def pff_example():
  sample_ffn = point_wise_feed_forward_network(512, 2048)
  print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)