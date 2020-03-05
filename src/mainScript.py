from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import config
#import tensorflow as tf
#import datetime
from modules.input_pipeline import create_input_pipeline
from modules.transformer import create_transformer
from modules.train_and_evaluate import train_the_transformer


# mr - meaning representation
# ref - reference             

# Initialize the input pipeline
input_pipeline = create_input_pipeline()
# input_pipeline.test_tokenizer(input_pipeline.ref_tokenizer, "This is so awesome!")
# input_pipeline.print_dataset_example(input_pipeline.test_dataset)

# Create the transformer
transformer = create_transformer(input_pipeline)

# Train the transformer
train_the_transformer(transformer, input_pipeline)

print("Done so far!")