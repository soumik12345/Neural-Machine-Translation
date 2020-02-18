import tensorflow as tf


def ScaledDotProductAttention(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    softmax_logits = qk / tf.math.sqrt(dk)
    if mask is not None:
        softmax_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(softmax_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.Model):

    def __init__(self, dense_units, attention_heads):
        super(MultiHeadAttention, self).__init__()
        self.dense_units = dense_units
        self.attention_heads = attention_heads
        self.depth = self.dense_units // self.attention_heads
        self.linear_q = tf.keras.layers.Dense(self.dense_units)
        self.linear_k = tf.keras.layers.Dense(self.dense_units)
        self.linear_v = tf.keras.layers.Dense(self.dense_units)
        self.linear = tf.keras.layers.Dense(self.dense_units)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.attention_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = ScaledDotProductAttention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.dense_units))
        output = self.linear(concat_attention)
        return output, attention_weights