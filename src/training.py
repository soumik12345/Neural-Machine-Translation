from .utils import *
import tensorflow as tf



class CustomLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, dense_units, warmup_steps=4000):
        super(CustomLRScheduler, self).__init__()
        self.dense_units = dense_units
        self.dense_units = tf.cast(self.dense_units, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return tf.math.rsqrt(self.dense_units) * tf.math.minimum(tf.math.rsqrt(step), step * (self.warmup_steps ** -1.5))



def get_masks(x, y):
    encoder_padding_mask = create_padding_mask(x)
    decoder_padding_mask = create_padding_mask(x)
    look_ahead_mask = create_look_ahead_mask(tf.shape(y)[1])
    decoder_target_padding_mask = create_padding_mask(y)
    combined_mask = tf.maximum(
        decoder_target_padding_mask,
        look_ahead_mask
    )
    return encoder_padding_mask, combined_mask, decoder_padding_mask



def get_checkpoints(transformer, optimizer, checkpoint_dir='./checkpoints/train'):
    checkpoint = tf.train.Checkpoint(
        transformer=transformer,
        optimizer=optimizer
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir, max_to_keep=5
    )
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
    return checkpoint, checkpoint_manager