from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import config
import tensorflow as tf
import nltk
import csv
import datetime
# nltk.download('punkt')
from modules.input_pipeline import create_input_pipeline
from modules.transformer import create_transformer
from modules.train_and_checkpointing import train_the_transformer
from modules.evaluate import generate_sentence
from nltk.translate.bleu_score import sentence_bleu

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

# Write to a csv
# name = datetime.datetime.now()
if not os.path.isdir(config.results_path):
    os.makedirs(config.results_path)
file_name = '{}/{}.csv'.format(config.results_path, 'results')
with open(file_name, 'a+', newline='') as csv_file:

    header_names = ['mr', 'ref', 'prediction', '1-gram', '2-gram', '3-gram', '4-gram']
    the_writer = csv.DictWriter(csv_file, fieldnames=header_names)

    the_writer.writeheader()

    counter = 0
    # Evaluate
    for entry in input_pipeline.test_examples:
        # Get a prediction
        mr,ref = entry
        mr_example = str(mr.numpy(), 'utf-8')
        ref_example = str(ref.numpy(), 'utf-8')
        predicted_sentence = generate_sentence(mr_example, input_pipeline, transformer)

        # Get the bleu scores   
        prediction = nltk.word_tokenize(predicted_sentence)
        reference = nltk.word_tokenize(ref_example)
        list_references = []
        list_references.append(reference)
        one_gram = sentence_bleu(list_references, prediction, weights=(1,0,0,0))
        two_gram = sentence_bleu(list_references, prediction, weights=(0,1,0,0))
        three_gram = sentence_bleu(list_references, prediction, weights=(0,0,1,0))
        four_gram = sentence_bleu(list_references, prediction, weights=(0,0,0,1))

        # write to the file
        the_writer.writerow({'mr' : mr_example, 'ref' : ref_example, 'prediction' : predicted_sentence, 
                            '1-gram' : '%.4f' % one_gram, '2-gram' : '%.4f' % two_gram, '3-gram' : '%.4f' % three_gram, '4-gram' : '%.4f' % four_gram})

        counter = counter + 1
        print(counter)
        if counter % 10 == 0 :
            sys.exit()
    # print('Individual 1-gram: %.4f' % one_gram)
    # print('Individual 2-gram: %.4f' % two_gram)
    # print('Individual 3-gram: %.4f' % three_gram)
    # print('Individual 4-gram: %.4f' % four_gram)
    # break



print("Done so far!")