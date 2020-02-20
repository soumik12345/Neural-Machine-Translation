import tensorflow as tf
import tensorflow_datasets as tfds


def download_dataset(tfds_address, disable_progress_bar=False):
    if disable_progress_bar:
        tfds.disable_progress_bar()
    data, metadata = tfds.load(tfds_address, with_info=True, as_supervised=True)
    train_data, val_data = data['train'], data['validation']
    return train_data, val_data


def get_tokenizers(dataset, approx_vocab_size=2 ** 13):
    source_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (i.numpy() for i, j in dataset),
        target_vocab_size=approx_vocab_size
    )
    target_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (j.numpy() for i, j in dataset),
        target_vocab_size=approx_vocab_size
    )
    return source_tokenizer, target_tokenizer


class DataLoader:

    def __init__(self, source_tokenizer, target_tokenizer, max_limit=40):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_limit = max_limit

    def preprocess(self, language_1, language_2):
        language_1 = [
            self.source_tokenizer.vocab_size
        ] + self.source_tokenizer.encode(
            language_1.numpy()
        ) + [
             self.source_tokenizer.vocab_size + 1
        ]
        language_2 = [
            self.target_tokenizer.vocab_size
        ] + self.target_tokenizer.encode(
            language_2.numpy()
        ) + [
            self.target_tokenizer.vocab_size + 1
        ]
        return language_1, language_2

    def map_function(self, language_1, language_2):
        language_1, language_2 = tf.py_function(
            self.preprocess,
            [language_1, language_2],
            [tf.int64, tf.int64]
        )
        language_1.set_shape([None])
        language_2.set_shape([None])
        return language_1, language_2

    def filter_max_length(self, x, y):
        return tf.logical_and(
            tf.size(x) <= self.max_limit,
            tf.size(y) <= self.max_limit
        )

    def get_dataset(self, dataset, buffer_size, batch_size):
        tf_dataset = dataset.map(self.map_function)
        if self.max_limit is not None:
            tf_dataset = tf_dataset.filter(self.filter_max_length)
        tf_dataset = tf_dataset.cache()
        tf_dataset = tf_dataset.shuffle(buffer_size)
        tf_dataset = tf_dataset.padded_batch(
            batch_size, padded_shapes=([None], [None])
        )
        tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return tf_dataset
