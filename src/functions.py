def test_tokenizer(tokenizer):
    sample_string = 'Transformer is awesome.'
    tokenized_string = tokenizer.encode(sample_string)
    print ('Tokenized string is {}'.format(tokenized_string))
    original_string = tokenizer.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))
    assert original_string == sample_string
    # The tokenizer encodes the string by breaking it into subwords if the word is not in its dictionary.
    for ts in tokenized_string:
        print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))