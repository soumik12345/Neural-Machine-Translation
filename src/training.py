import tensorflow as tf



class CustomLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, dense_units, warmup_steps=4000):
        super(CustomLRScheduler, self).__init__()
        self.dense_units = dense_units
        self.dense_units = tf.cast(self.dense_units, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return tf.math.rsqrt(self.dense_units) * tf.math.minimum(tf.math.rsqrt(step), step * (self.warmup_steps ** -1.5))