import tensorflow as tf
import config
from modules.encoder import Encoder
from modules.decoder import Decoder

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=config.dropout_rate):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
  def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    return final_output, attention_weights
    
# interface
def create_transformer(input_pipeline):
  # Get the vocab size of the 
  mr_vocab_size = input_pipeline.mr_tokenizer.vocab_size + 2
  ref_vocab_size = input_pipeline.ref_tokenizer.vocab_size + 2

  tr = Transformer(config.num_layers, config.d_model, config.num_heads, config.dff, mr_vocab_size, ref_vocab_size, 
                        pe_input=mr_vocab_size, pe_target=ref_vocab_size, rate=config.dropout_rate)
  return tr