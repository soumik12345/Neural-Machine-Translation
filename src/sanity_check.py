import tensorflow as tf


class SanityCheck:

    def __init__(self):
        pass

    @staticmethod
    def test_1(source_tokenizer, target_tokenizer):
        sample_string = 'Downy feathers kiss your face and flutter everywhere'
        tokenized_string = target_tokenizer.encode(sample_string)
        print('Sample string in English: {}'.format(sample_string))
        print('Sample string tokenized: {}'.format(tokenized_string))
        sample_string = 'Penas felpudas beijam seu rosto e flutuam por toda parte'
        tokenized_string = source_tokenizer.encode(sample_string)
        print('Sample string in Portuguese: {}'.format(sample_string))
        print('Sample string tokenized: {}'.format(tokenized_string))
    
    @staticmethod
    def test_2(train_dataset, val_dataset, batch_size):
        print(train_dataset)
        print(val_dataset)
        source_language_batch, target_language_batch = next(iter(train_dataset))
        assert list(source_language_batch.shape)[0] == batch_size and list(target_language_batch.shape)[0] == batch_size