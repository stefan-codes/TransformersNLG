import os
import tensorflow as tf
import config as c

from input_pipeline import load_dataset_examples
from input_pipeline import create_tokenizers
from input_pipeline import encode_datasets
from transformer import Transformer
from optimizer import CustomSchedule
from train_and_checkpointing import train_the_transformer
from evaluate import generate_sentence

###############################
# mr - meaning representation #
# ref - reference             #
###############################

# Set up the paths to the train and test data
dirname = os.path.realpath('')
csv_train_examples = dirname + "\\data\\e2e-dataset\\trainset.csv"          #42,062 entries with label
csv_test_examples = dirname + "\\data\\e2e-dataset\\testset_w_refs.csv"

# Defaults describing the data for making a tensorflow-dataset
defaults = [tf.string]*2

# Make datasets with the examples
train_examples, test_examples = load_dataset_examples(csv_train_examples, csv_test_examples, defaults)

# Create the tokenizers
mr_tokenizer, ref_tokenizer = create_tokenizers(train_examples)

# Encode the data
train_dataset, test_dataset = encode_datasets(train_examples, test_examples, mr_tokenizer, ref_tokenizer)

# Get the vocab size of the 
mr_vocab_size = mr_tokenizer.vocab_size + 2
ref_vocab_size = ref_tokenizer.vocab_size + 2

# Create the transformer
transformer = Transformer(
                        c.num_layers,
                        c.d_model, 
                        c.num_heads, 
                        c.dff, 
                        mr_vocab_size, 
                        ref_vocab_size, 
                        pe_input=mr_vocab_size, 
                        pe_target=ref_vocab_size, 
                        rate=c.dropout_rate)

# Set the check point save location
checkpoint_path = "./checkpoints/train"

# Train the transformer
train_the_transformer(transformer, train_dataset, checkpoint_path)

#TODO: Run the whole validation set
"""
for kk in (mr.numpy() for mr, ref in train_examples):
    print(str(kk, 'utf-8'))
    
asd = next(iter())
print(str(asd, 'utf-8'))
"""
example = next(iter(train_examples))
mr_example = example[0]
ref_example = example[1]
print(str(mr_example.numpy(), 'utf-8'), str(ref_example.numpy(), 'utf-8'))

# Evaluate
generate_sentence(mr_example, ref_example, mr_tokenizer, ref_tokenizer, transformer)


# Generate text
#generate_sentence()

print("Done so far!")


#generate_sentence("name[Blue Spice], eatType[pub], customer rating[average], near[Burger King]")
#print ("Suggested output is: Average customer rating pub include Blue Spice near Burger King.")