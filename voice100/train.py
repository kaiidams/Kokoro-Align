from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from .transformer import *
import time

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
      num_epochs=20,
      target_modal=flags_obj.target_modal
    )

  def create_model(self):
    params = self.params
    target_modal = params['target_modal']
    if target_modal == 'text':
      target_vocab_size = params['vocab_size']
    elif target_modal == 'audio':
      target_vocab_size = params['audio_dim']
    else:
      raise ValueError()

    model = Transformer(
        num_layers=params['num_hidden_layers'],
        d_model=params['hidden_size'],
        num_heads=params['num_heads'],
        dff=params['filter_size'],
        input_vocab_size=params['vocab_size'],
        target_vocab_size=target_vocab_size,
        pe_input=1000,
        pe_target=1000,
        rate=params['dropout'],
        target_modal=target_modal)
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
      accuracies = tf.equal(real, tf.argmax(pred, axis=2))
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

  def create_accuracy_function_audio(self):
    def accuracy_function(real, pred, dec_target_padding_mask):
      return 0.0

    return accuracy_function

  def create_loss_function(self):
    target_modal = self.params['target_modal']
    if target_modal == 'text':
      return self.create_loss_function_text()
    elif target_modal == 'audio':
      return self.create_loss_function_audio()
    else:
      raise ValueError()

  def create_accuracy_function(self):
    target_modal = self.params['target_modal']
    if target_modal == 'text':
      return self.create_accuracy_function_text()
    elif target_modal == 'audio':
      return self.create_accuracy_function_audio()
    else:
      raise ValueError()

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
    else:
      if flags_obj.init_checkpoint:
        ckpt.restore(flags_obj.init_checkpoint)
        print('Loaded from initial checkpoint.')

    loss_function = self.create_loss_function()
    accuracy_function = self.create_accuracy_function()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    from .data_pipeline import train_input_fn
    train_ds = train_input_fn(params)

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    target_modal = self.params['target_modal']
    if target_modal == 'text':
      train_step_signature = [
          tf.TensorSpec(shape=(None, None), dtype=tf.int64),
          tf.TensorSpec(shape=(None, None), dtype=tf.int64),
          tf.TensorSpec(shape=(None, None), dtype=tf.float32),
          tf.TensorSpec(shape=(None, None), dtype=tf.float32),
      ]
    elif target_modal == 'audio':
      train_step_signature = [
          tf.TensorSpec(shape=(None, None), dtype=tf.int64),
          tf.TensorSpec(shape=(None, None, params['audio_dim']), dtype=tf.float32),
          tf.TensorSpec(shape=(None, None), dtype=tf.float32),
          tf.TensorSpec(shape=(None, None), dtype=tf.float32),
      ]
    else:
      raise ValueError()

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar, inp_mask, tar_mask):
      tar_inp = tar[:, :-1]
      tar_real = tar[:, 1:]
      
      enc_padding_mask = inp_mask[:, tf.newaxis, tf.newaxis, :]
      dec_padding_mask = inp_mask[:, tf.newaxis, tf.newaxis, :]
      dec_target_padding_mask = tar_mask[:, tf.newaxis, tf.newaxis, 1:]

      look_ahead_mask = create_look_ahead_mask(tf.shape(dec_target_padding_mask)[3])
      combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
      
      with tf.GradientTape() as tape:
        predictions, _ = model(inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = loss_function(tar_real, predictions, dec_target_padding_mask)

      gradients = tape.gradient(loss, model.trainable_variables)    
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
      train_loss(loss)
      train_accuracy(accuracy_function(tar_real, predictions, dec_target_padding_mask))

    for epoch in range(params['num_epochs']):
      start = time.time()
      
      train_loss.reset_states()
      train_accuracy.reset_states()
      
      for batch, (inp, tar, inp_mask, tar_mask) in enumerate(train_ds):
        train_step(inp, tar, inp_mask, tar_mask)
        
        if batch % 50 == 0:
          print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
          
      if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
        
      print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

      print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

  def evaluate(self, model, encoder_input, max_length=300):
    start = 0
    end = 0
    print(encoder_input.shape)
    encoder_input = tf.convert_to_tensor([encoder_input], dtype=tf.int64)
    #output = tf.convert_to_tensor([start], dtype=tf.int64)
    #output = tf.expand_dims(output, 0)
    output = tf.zeros([1, 1, self.params['audio_dim']], dtype=tf.float32)
    
    for i in range(max_length):
      print(i)
      #enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
      #    encoder_input, output)
      inp_mask = tf.zeros_like(encoder_input, dtype=tf.float32)
      tar_mask = tf.zeros_like(output[:, :, 0], dtype=tf.float32)
      enc_padding_mask = inp_mask
      enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]

      dec_padding_mask = inp_mask
      dec_padding_mask = dec_padding_mask[:, tf.newaxis, tf.newaxis, :]

      dec_target_padding_mask = tar_mask#[:, 1:]
      dec_target_padding_mask = dec_target_padding_mask[:, tf.newaxis, tf.newaxis, :]

      look_ahead_mask = create_look_ahead_mask(tf.shape(dec_target_padding_mask)[3])
      combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
          
      # predictions.shape == (batch_size, seq_len, vocab_size)
      predictions, attention_weights = model(encoder_input, 
                                                  output,
                                                  False,
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)
      
      # select the last word from the seq_len dimension
      predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

      if False:
        predicted_id = tf.argmax(predictions, axis=-1)

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == end:
          break
      else:
        #print(predictions.shape)
        output = tf.concat([output, predictions], axis=-2)      

    return output, attention_weights

  def predict(self):
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

    from .data_pipeline import eval_input_fn
    eval_dataset = eval_input_fn(params)

    from .preprocess import feature2wav, feature2text, writewav
    for example in eval_dataset.take(1):
      encoder_input = example[0][0]
      t = feature2text(encoder_input)
      print(t)
      output, attention_weights = self.evaluate(model, encoder_input)
      x = feature2wav(output.numpy()[0])
      attention_weights = {
        k: v.numpy()
        for k, v in attention_weights.items()
      }
      np.savez('data/test.npz', x=x, t=t, **attention_weights)
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
      name='target_modal',
      default=None,
      help='modal: text, audio')
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
