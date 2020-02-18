import tensorflow as tf
import tensorflow_datasets as tfds


def download_dataset(tfds_address, disable_progress_bar=False):
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

