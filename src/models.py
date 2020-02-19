from .blocks import *
from .encoder import *
from .decoder import *
from .utils import *
from .dataset import *
from .sanity_check import SanityCheck
import tensorflow as tf



class Transformer(tf.keras.Model):

    def __init__(
        self, n_layers, dense_units, attention_heads,
        dff, input_vocab_size, target_vocab_size,
        position_encoding_input, position_encoding_target, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            n_layers, dense_units, attention_heads, dff,
            input_vocab_size, position_encoding_input, dropout_rate
        )
        self.decoder = Decoder(
            n_layers, dense_units, attention_heads, dff,
            target_vocab_size, position_encoding_target, dropout_rate
        )
        self.output_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self, _input, _target, training, encoder_padding_mask,
        look_ahead_mask, decoder_padding_mask):
        encoder_output = self.encoder(_input, training, encoder_padding_mask)
        decoder_output, attention_weights = self.decoder(
            _target, encoder_output, training,
            look_ahead_mask, decoder_padding_mask
        )
        output = self.output_layer(decoder_output)
        return output, attention_weights



class TrainingModel(object):

    def __init__(self, configs):
        
        train_data, val_data = download_dataset(
            configs['dataset_adress'],
            disable_progress_bar=True
        )

        source_tokenizer, target_tokenizer = get_tokenizers(train_data)

        dataloader = DataLoader(source_tokenizer, target_tokenizer)
        dataloader = DataLoader(source_tokenizer, target_tokenizer)
        train_dataset = dataloader.get_dataset(
            train_data,
            configs['buffer_size'],
            configs['batch_size']
        )
        val_dataset = dataloader.get_dataset(
            train_data,
            configs['buffer_size'],
            configs['batch_size']
        )