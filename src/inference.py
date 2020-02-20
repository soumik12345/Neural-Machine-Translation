from .training import *
import tensorflow as tf



def predict(sentence, transformer, source_tokenizer, target_tokenizer, max_limit):
    start_token = [source_tokenizer.vocab_size]
    end_token = [source_tokenizer.vocab_size + 1]
    sentence = start_token + source_tokenizer.encode(sentence) + end_token
    encoder_input = tf.expand_dims(sentence, 0)
    decoder_input = [target_tokenizer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)
    for i in range(max_limit):
        encoder_padding_mask, combined_mask, decoder_padding_mask = get_masks(encoder_input, output)
        preds, attention_weights = transformer(
            encoder_input, output, True,
            encoder_padding_mask, combined_mask, decoder_padding_mask
        )
        preds = preds[: , -1 :, :]
        predicted_id = tf.cast(tf.argmax(preds, axis=-1), tf.int32)
        if predicted_id == target_tokenizer.vocab_size+1:
            result = tf.squeeze(output, axis=0)
            predicted_sentence = target_tokenizer.decode([i for i in result if i < target_tokenizer.vocab_size])
            return result, predicted_sentence, attention_weights
        output = tf.concat([output, predicted_id], axis=-1)
    result = tf.squeeze(output, axis=0)
    predicted_sentence = target_tokenizer.decode([i for i in result if i < target_tokenizer.vocab_size])
    return predicted_sentence, result, attention_weights