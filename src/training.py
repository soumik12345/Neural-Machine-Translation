from .loss import *
from .utils import *
from .models import *
from time import time
from tqdm import tqdm
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




def train(dataset, transformer, optimizer, epochs, checkpoint_dir):

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    summary_writer = tf.summary.create_file_writer('logs')
    checkpoint, checkpoint_manager = get_checkpoints(transformer, optimizer, checkpoint_dir)
    
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
            tf.TensorSpec(shape=(None, None), dtype=tf.int64)
        ]
    )
    def train_step(source, target):
        target_input = target[:, :-1]
        target_real = target[:, 1:]
        encoder_padding_mask, combined_mask, decoder_padding_mask = get_masks(source, target_input)
        with tf.GradientTape() as tape:
            predictions, _ = transformer(
                source, target_input, True, encoder_padding_mask,
                combined_mask, decoder_padding_mask
            )
            loss = loss_function(target_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        return train_loss(loss), train_accuracy(target_real, predictions)

    for epoch in range(epochs):
        start_time = time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        print('Epoch: {}....'.format(epoch + 1))
        with summary_writer.as_default():
            iteration = 0
            for (batch, (source, target)) in tqdm(enumerate(dataset)):
                batch_loss, batch_accuracy = train_step(source, target)
                tf.summary.scalar('Train Loss', batch_loss, step=iteration)
                tf.summary.scalar('Train Accuracy', batch_accuracy, step=iteration)
                iteration += 1
            summary_writer.flush()
        checkpoint_manager.save()
        print('Done. Time taken: {} seconds'.format(time() - start_time))