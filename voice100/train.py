from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from .transformer import *
from .data_pipeline import train_input_fn, eval_input_fn
import time
import os

class Voice100Task(object):

  def __init__(self, flags_obj):
    self.flags_obj = flags_obj
    self.params = dict(
      vocab_size=28,
      audio_dim=27,
      hidden_size=128,
      num_hidden_layers=4,
      num_heads=8,
      filter_size=512,
      dropout=0.1,
      batch_size=50,
      num_epochs=100,
    )

  def create_model(self):
    params = self.params

    model = Transformer(
        num_layers=params['num_hidden_layers'],
        d_model=params['hidden_size'],
        num_heads=params['num_heads'],
        dff=params['filter_size'],
        input_vocab_size=params['vocab_size'],
        target_vocab_size=params['vocab_size'],
        target_audio_dim=params['audio_dim'],
        pe_input=1000,
        pe_target=1000,
        rate=params['dropout'])
    return model

  def create_optimizer(self):
    params = self.params
    learning_rate = CustomSchedule(params['hidden_size'])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)
    return optimizer

  def create_loss_function_text(self):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(real, pred, dec_target_padding_mask):
      mask = 1 - dec_target_padding_mask
      loss_ = loss_object(real, pred)
      loss_ = loss_[:, tf.newaxis, tf.newaxis, :]
      loss_ *= mask
      return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    return loss_function

  def create_accuracy_function_text(self):
    def accuracy_function(real, pred, dec_target_padding_mask):
      accuracies = tf.equal(real, tf.argmax(pred, axis=2)) # (batch_size, tar_seq_len, target_vocab_size)
      mask = 1 - dec_target_padding_mask
      accuracies = tf.cast(accuracies, dtype=tf.float32)
      accuracies = accuracies[:, tf.newaxis, tf.newaxis, :]
      accuracies *= mask
      return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    return accuracy_function

  def create_loss_function_audio(self):
    loss_object = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(real, pred, dec_target_padding_mask):
      mask = 1 - dec_target_padding_mask
      loss_ = loss_object(real, pred)
      loss_ = loss_[:, tf.newaxis, tf.newaxis, :]
      loss_ *= mask[:, :, :, :]
      return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    return loss_function

  def create_loss_function_binary(self):
    loss_object = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)

    def loss_function(real, pred, dec_target_padding_mask):
      mask = 1 - dec_target_padding_mask
      loss_ = loss_object(real[:, :, tf.newaxis], pred)
      loss_ = loss_[:, tf.newaxis, tf.newaxis, :]
      loss_ *= mask
      return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    return loss_function

  def create_accuracy_function_binary(self):
    def accuracy_function(real, pred, dec_target_padding_mask):
      # real: (batch_size, tar_seq_len)
      # pred: (batch_size, tar_seq_len, 1)
      # dec_target_padding_mask: (batch_size, 1, 1, tar_seq_len)
      pred = tf.cast(pred > 0.5, dtype=tf.int64)
      accuracies = tf.equal(real[:, :, tf.newaxis], pred)
      mask = 1 - dec_target_padding_mask
      accuracies = tf.cast(accuracies, dtype=tf.float32)
      accuracies = tf.reduce_mean(accuracies, axis=2) # (batch_size, tar_seq_len)
      accuracies = accuracies[:, tf.newaxis, tf.newaxis, :]
      accuracies *= mask
      return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    return accuracy_function

  def train(self):
    flags_obj = self.flags_obj
    params = self.params
    model = self.create_model()
    optimizer = self.create_optimizer()

    ckpt = tf.train.Checkpoint(transformer=model,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, flags_obj.model_dir, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print ('Latest checkpoint restored!!')
      start_epoch = ckpt.save_counter.numpy() * 10
    else:
      start_epoch = 0
      if flags_obj.init_checkpoint:
        ckpt.restore(flags_obj.init_checkpoint)
        print('Loaded from initial checkpoint.')

    loss_function_align = self.create_loss_function_text()
    loss_function_audio = self.create_loss_function_audio()
    loss_function_end = self.create_loss_function_binary()
    accuracy_function_align = self.create_accuracy_function_text()
    accuracy_function_end = self.create_accuracy_function_binary()

    train_loss_align = tf.keras.metrics.Mean(name='train_loss_align')
    train_loss_audio = tf.keras.metrics.Mean(name='train_loss_audio')
    train_loss_end = tf.keras.metrics.Mean(name='train_loss_end')
    train_accuracy_align = tf.keras.metrics.Mean(name='train_accuracy_align')
    train_accuracy_end = tf.keras.metrics.Mean(name='train_accuracy_end')

    train_ds = train_input_fn(params)

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None, params['audio_dim']), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None), dtype=tf.float32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tgt_align, tgt_audio, tgt_end, inp_mask, tar_mask):
      tgt_input = tgt_audio[:, :-1]
      tgt_align_real = tgt_align[:, 1:]
      tgt_audio_real = tgt_audio[:, 1:]
      tgt_end_real = tgt_end[:, 1:]
      
      enc_padding_mask = inp_mask[:, tf.newaxis, tf.newaxis, :]
      dec_padding_mask = inp_mask[:, tf.newaxis, tf.newaxis, :]
      dec_target_padding_mask = tar_mask[:, tf.newaxis, tf.newaxis, 1:]

      look_ahead_mask = create_look_ahead_mask(tf.shape(dec_target_padding_mask)[3])
      combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
      
      with tf.GradientTape() as tape:
        tgt_align_pred, tgt_audio_pred, tgt_end_pred, _ = model(inp, tgt_input, 
                                    True,
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss_align = loss_function_align(tgt_align_real, tgt_align_pred, dec_target_padding_mask)
        loss_audio = loss_function_audio(tgt_audio_real, tgt_audio_pred, dec_target_padding_mask)
        loss_end = loss_function_end(tgt_end_real, tgt_end_pred, dec_target_padding_mask)

        loss = loss_align * 0.3 + loss_audio * 0.3 + loss_end * 0.4

      gradients = tape.gradient(loss, model.trainable_variables)    
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
      train_loss_align(loss_align)
      train_loss_audio(loss_audio)
      train_loss_end(loss_end)
      train_accuracy_align(accuracy_function_align(tgt_align_real, tgt_align_pred, dec_target_padding_mask))
      train_accuracy_end(accuracy_function_end(tgt_end_real, tgt_end_pred, dec_target_padding_mask))

    for epoch in range(start_epoch, params['num_epochs']):
      start = time.time()
      
      train_loss_align.reset_states()
      train_loss_audio.reset_states()
      train_loss_end.reset_states()
      train_accuracy_align.reset_states()
      train_accuracy_end.reset_states()
      
      for batch, example in enumerate(train_ds):
        train_step(*example)
        
        if batch % 50 == 0:
          print(f'Epoch {epoch + 1} Batch {batch}')
          print(f'Align Loss {train_loss_align.result():.4f} Accuracy {train_accuracy_align.result():.4f}')
          print(f'Audio Loss {train_loss_audio.result():.4f}')
          print(f'End Loss {train_loss_end.result():.4f} Accuracy {train_accuracy_end.result():.4f}')
          
      if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        
      print(f'Epoch {epoch + 1}')
      print(f'Align Loss {train_loss_align.result():.4f} Accuracy {train_accuracy_align.result():.4f}')
      print(f'Audio Loss {train_loss_audio.result():.4f}')
      print(f'End Loss {train_loss_end.result():.4f} Accuracy {train_accuracy_end.result():.4f}')

      print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

  def predict_step(self, model, src, tgt_align, tgt_audio, tgt_end, src_mask, tgt_mask, max_length=300):

    output_align = tf.zeros([1, 1], dtype=tf.int64)
    output_audio = tf.zeros([1, 1, self.params['audio_dim']], dtype=tf.float32)
    
    for i in range(max_length):
      print(i)

      enc_padding_mask = src_mask[:, tf.newaxis, tf.newaxis, :]
      dec_padding_mask = src_mask[:, tf.newaxis, tf.newaxis, :]
      look_ahead_mask = create_look_ahead_mask(tf.shape(output_audio)[1])
      combined_mask = look_ahead_mask
          
      # predictions.shape == (batch_size, seq_len, vocab_size)
      tgt_align_pred, tgt_audio_pred, tgt_end_pred, attention_weights = model(src, 
                                                  output_audio,
                                                  False,
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)
      
      # select the last word from the seq_len dimension
      tgt_align_pred = tgt_align_pred[:, -1:, :]  # (batch_size, 1, vocab_size)
      tgt_audio_pred = tgt_audio_pred[:, -1:, :]  # (batch_size, 1, audio_dim)
      tgt_end_pred = tgt_end_pred[:, -1:, :]  # (batch_size, 1, vocab_size)

      tgt_align_pred_id = tf.argmax(tgt_align_pred, axis=-1) # (batch_size, 1)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_align = tf.concat([output_align, tgt_align_pred_id], axis=1)
      output_audio = tf.concat([output_audio, tgt_audio_pred], axis=1)

      # return the result if the predicted_id is equal to the end token
      if tf.reduce_any(tgt_end_pred > 0.1):
        break

    return output_align, output_audio, attention_weights

  def predict(self):
    from .preprocess import feature2wav, feature2text, writewav

    flags_obj = self.flags_obj
    params = self.params
    params['batch_size'] = 1
    model = self.create_model()
    optimizer = self.create_optimizer()

    ckpt = tf.train.Checkpoint(transformer=model,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, flags_obj.model_dir, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print ('Latest checkpoint restored!!')
    else:
      raise ValueError()

    eval_ds = eval_input_fn(params)

    for batch, example in enumerate(eval_ds):
      t = feature2text(example[0][0])
      print(t)
      output_align, output_audio, attention_weights = self.predict_step(model, *example)
      a = feature2text(output_align.numpy()[0])
      print(a)
      x = feature2wav(output_audio.numpy()[0])
      attention_weights = {
        k: v.numpy()
        for k, v in attention_weights.items()
      }
      np.savez('data/test.npz', x=x, t=t, a=a, **attention_weights)
      writewav('data/test.wav', x, 16000)
      #print(encoder_input)
      #print(output)
      #print(attention_weights)

    print('done')

def main(_):
  flags_obj = flags.FLAGS

  task = Voice100Task(flags_obj)

  if flags_obj.mode == 'train':
    task.train()
  elif flags_obj.mode == 'predict':
    task.predict()
  else:
    raise ValueError()

def define_voice100_flags():
  flags.DEFINE_string(
      name="model_dir",
      short_name="md",
      default="/tmp",
      help="The location of the model checkpoint files.")
  flags.DEFINE_string(
      name='mode',
      default='train',
      help='mode: train, eval, or predict')
  flags.DEFINE_string(
      'init_checkpoint', None,
      'Initial checkpoint (usually from a pre-trained BERT model).')

if __name__ == "__main__":
  define_voice100_flags()
  logging.set_verbosity(logging.INFO)
  app.run(main)
