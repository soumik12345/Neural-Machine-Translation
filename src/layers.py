from .utils import *
from .blocks import *
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, dense_units, attention_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(dense_units, attention_heads)
        self.pw_ffn = PositionWiseFFN(dense_units, dff)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        attention, _ = self.attention(x, x, x, mask)
        attention = self.dropout_1(attention, training=training)
        output_1 = self.norm_1(x + attention)
        ffn_out = self.pw_ffn(output_1)
        ffn_out = self.dropout_2(ffn_out, training=training)
        output_2 = self.norm_2(output_1 + ffn_out)
        return output_2



class Encoder(tf.keras.layers.Layer):

    def __init__(
        self, n_encoder_layers, dense_units,
        attention_heads, dff, input_vocab_size,
        max_positional_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.dense_units = dense_units
        self.n_encoder_layers = n_encoder_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, dense_units)
        self.positional_encoding = positional_encoding(max_positional_encoding, self.dense_units)
        self.encoder_layers = [
            EncoderLayer(
                self.dense_units, attention_heads,
                dff, dropout_rate
            ) for _ in range(self.n_encoder_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, training, mask):
        sequence_length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dense_units, dtype=tf.float32))
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x, training=training)
        for i in range(self.n_encoder_layers):
            x = self.encoder_layers[i](x, training, mask)
        return x



class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, dense_units, attention_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.attention_1 = MultiHeadAttention(dense_units, attention_heads)
        self.attention_2 = MultiHeadAttention(dense_units, attention_heads)
        self.ffn = PositionWiseFFN(dense_units, dff)
        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        attn_1, attn_weights_1 = self.attention_1(x, x, x, look_ahead_mask)
        attn_1 = self.dropout_1(attn_1, training=training)
        out_1 = self.norm_1(attn_1 + x)
        attn_2, attn_weights_2 = self.attention_2(
            encoder_output, encoder_output,
            out_1, padding_mask
        )
        attn_2 = self.dropout_2(attn_2, training=training)
        out_2 = self.norm_2(attn_2 + out_1)
        ffn_out = self.ffn(out_2)
        ffn_out = self.dropout_3(ffn_out, training=training)
        out_3 = self.norm_3(ffn_out + out_2)
        return out_3, attn_weights_1, attn_weights_2
