from .utils import *
from .blocks import *
import tensorflow as tf


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
