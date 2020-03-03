from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import config
import tensorflow as tf
import nltk
from modules.input_pipeline import create_input_pipeline
from modules.transformer import create_transformer
from modules.train_and_checkpointing import train_the_transformer
from modules.evaluate import generate_sentence
from nltk.translate.bleu_score import sentence_bleu
# from optimizer import CustomSchedule


# mr - meaning representation
# ref - reference             

# Initialize the input pipeline
input_pipeline = create_input_pipeline()
# input_pipeline.test_tokenizer(input_pipeline.ref_tokenizer, "This is so awesome!")
# input_pipeline.print_dataset_example(input_pipeline.test_dataset)

# Create the transformer
transformer = create_transformer(input_pipeline)

# Train the transformer
train_the_transformer(transformer, input_pipeline.train_dataset)

#TODO Evaluate
example = next(iter(input_pipeline.test_examples))
mr_example = str(example[0].numpy(), 'utf-8')
ref_example = str(example[1].numpy(), 'utf-8')
# print(str(mr_example.numpy(), 'utf-8'), str(ref_example.numpy(), 'utf-8'))
predicted_sentence = generate_sentence(mr_example, ref_example, input_pipeline, transformer)

#TODO bleu score
prediction = nltk.word_tokenize(predicted_sentence)
reference = nltk.word_tokenize(ref_example)
#reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
#prediction = ['this', 'is', 'a', 'test']
score = sentence_bleu(reference, prediction, weights=(1.0,0.0,0.0,0.0))
print(score)

sys.exit()


#TODO: Run the whole validation set
"""
for kk in (mr.numpy() for mr, ref in train_examples):
    print(str(kk, 'utf-8'))
    
asd = next(iter())
print(str(asd, 'utf-8'))
"""



print("Done so far!")