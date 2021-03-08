import tensorflow as tf
import numpy as np
import random

def _readdata(file):
    f = np.load(file)
    return {k:v for k, v in f.items()}

def _getdataitem(data, index):
  id_ = data['id'][index]
  text_start = data['text_index'][index - 1] if index else 0
  text_end = data['text_index'][index]
  audio_start = data['audio_index'][index - 1] if index else 0
  audio_end = data['audio_index'][index]
  text = data['text_data'][text_start:text_end]
  audio = data['audio_data'][audio_start:audio_end, :]
  assert text_start < text_end
  assert audio_start < audio_end
  alignment = get_alignment_text(text, audio)
  return id_, text, audio, alignment

def get_alignment_text(text, audio):
  a = (len(text) + 2) / len(audio)
  res = np.array([
      text[min(max(0, int((i + 2.0 * random.random() - 1.0) * a - 1.0)), len(text) - 1)]
      for i in range(len(audio))
  ], dtype=text.dtype)
  return res

def get_dataset(file, audio_dim, shuffle=False):
  data = _readdata(file)
  def gen():
    indices = list(range(len(data['id'])))
    if shuffle:
      random.shuffle(indices)
    for index in indices:
      yield _getdataitem(data, index)

  return tf.data.Dataset.from_generator(gen,
    output_signature=(
      tf.TensorSpec(shape=(), dtype=tf.string), # id
      tf.TensorSpec(shape=(None,), dtype=tf.int64), # text
      tf.TensorSpec(shape=(None, audio_dim), dtype=tf.float32), # audio
      tf.TensorSpec(shape=(None,), dtype=tf.int64))) # align

def get_input_fn(params, is_train):
  if is_train:
    ds = get_dataset('data/css10ja_train.npz', params['audio_dim'], shuffle=True)
  else:
    ds = get_dataset('data/css10ja_val.npz', params['audio_dim'])

  target_modal = params['target_modal']
  if target_modal == 'text':
    ds = ds.map(lambda id_, text, audio, align: (text, align,
      tf.zeros(tf.shape(text)[0], dtype=tf.float32),
      tf.zeros(tf.shape(align)[0], dtype=tf.float32)))
    ds = ds.padded_batch(
        params['batch_size'],
        padding_values=(
          tf.constant(0, dtype=tf.int64),
          tf.constant(0, dtype=tf.int64),
          tf.constant(1.0, dtype=tf.float32),
          tf.constant(1.0, dtype=tf.float32),
          ),
        drop_remainder=False
    )
  elif target_modal == 'audio': 
    ds = ds.map(lambda  id_, text, audio, align: (text, audio,
      tf.zeros(tf.shape(text)[0], dtype=tf.float32),
      tf.zeros(tf.shape(audio)[0], dtype=tf.float32)))
    ds = ds.padded_batch(
        params['batch_size'],
        padding_values=(
          tf.constant(0, dtype=tf.int64),
          tf.constant(0.0, dtype=tf.float32),
          tf.constant(1.0, dtype=tf.float32),
          tf.constant(1.0, dtype=tf.float32),
          ),
        drop_remainder=False
    )
  else:
    raise ValueError(f'Unknown target_model {target_modal}')

  return ds

def train_input_fn(params):
  return get_input_fn(params, True)

def eval_input_fn(params):
  return get_input_fn(params, False)

def test():
  ds = train_input_fn(dict(audio_dim=27, target_modal='audio', batch_size=3))
  for example in ds:
    print(example)
    break

if __name__ == '__main__':
  test()