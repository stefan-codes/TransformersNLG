import tensorflow as tf

# Mask all the pad tokens in the batch of sequence. 
# It ensures that the model does not treat padding as the input. 
# The mask indicates where pad value 0 is present: it outputs a 1 at those locations, and a 0 otherwise.
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # add extra dimensions to add the padding to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
  
# The look-ahead mask is used to mask the future tokens in a sequence. 
# In other words, the mask indicates which entries should not be used.
# This means that to predict the third word, only the first and second word will be used. 
# Similarly to predict the fourth word, only the first, second and the third word will be used and so on.
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

# To create all the masks needed during training and evaluation
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

# Example of how it works
def masking_example():
  x = tf.random.uniform((1, 3))
  temp = create_look_ahead_mask(x.shape[1])
  print(temp)