import tensorflow as tf
import numpy as np
import random

def _readdata(file, align_file):
  f = np.load(file)
  data = {k:v for k, v in f.items()}
  if align_file:
    f = np.load(align_file)
    data['align_data'] = f['align_data']
    assert np.all(data['audio_index'] == f['align_index'])
  return data

def _getdataitem(data, index, vocab_size):
  id_ = data['id'][index]
  text_start = data['text_index'][index - 1] if index else 0
  text_end = data['text_index'][index]
  audio_start = data['audio_index'][index - 1] if index else 0
  audio_end = data['audio_index'][index]
  assert text_start < text_end
  assert audio_start < audio_end

  text = data['text_data'][text_start:text_end]
  audio = data['audio_data'][audio_start:audio_end, :]
  if 'align_data' in data:
    align = data['align_data'][audio_start:audio_end]
  else:
    align = np.zeros([audio_end - audio_start], dtype=np.float32)
  end = np.zeros([audio_end - audio_start], dtype=np.int64)
  end[-1] = 1
  return id_, text, align, audio, end

def get_dataset(params, file, shuffle=False, align_file=None):
  vocab_size = params['vocab_size']
  data = _readdata(file, align_file)
  def gen():
    indices = list(range(len(data['id'])))
    if shuffle:
      random.shuffle(indices)
    for index in indices:
      yield _getdataitem(data, index, vocab_size)

  return tf.data.Dataset.from_generator(gen,
    output_signature=(
      tf.TensorSpec(shape=(), dtype=tf.string), # id
      tf.TensorSpec(shape=(None,), dtype=tf.int64), # text
      tf.TensorSpec(shape=(None,), dtype=tf.int64), # align
      tf.TensorSpec(shape=(None, params['audio_dim']), dtype=tf.float32), # audio
      tf.TensorSpec(shape=(None,), dtype=tf.int64))) # end

def get_input_fn(params, split=None, use_align=True, **kwargs):
  if split == 'train':
    ds = get_dataset(params, 'data/css10ja_train.npz',
      align_file='data/css10ja_train_align.npz' if use_align else None,
      **kwargs)
  else:
    ds = get_dataset(params, 'data/css10ja_val.npz')

  ds = ds.map(lambda id_, text, align, audio, end: (
    text,
    tf.cast(tf.shape(text)[0], tf.int32),
    align,
    audio,
    end,
    tf.cast(tf.shape(audio)[0], tf.int32)))
  ds = ds.padded_batch(
      params['batch_size'],
      padding_values=(
        tf.constant(0, dtype=tf.int64), # text
        tf.constant(0, dtype=tf.int32), # text_len
        tf.constant(0, dtype=tf.int64), # align
        tf.constant(0.0, dtype=tf.float32), # audio
        tf.constant(0, dtype=tf.int64), # end
        tf.constant(0, dtype=tf.int32), # audio_len
        ),
      drop_remainder=False
  )

  return ds

def train_input_fn(params, shuffle=True, **kwargs):
  return get_input_fn(params, split='train', shuffle=shuffle, **kwargs)

def eval_input_fn(params, **kwargs):
  return get_input_fn(params, split='eval', shuffle=False, **kwargs)

def test():
  import sys
  from .preprocess import feature2text
  params = dict(vocab_size=29, audio_dim=27, batch_size=3)
  if sys.argv[1] == 'train':
    ds = train_input_fn(params, use_align=True)
  else:
    ds = eval_input_fn(params, use_align=False)
  for example in ds:
    text, text_len, align, audio, end, audio_len = example
    for i in range(text.shape[0]):
      print('---')
      print(feature2text(text[i, :text_len[i]].numpy()))
      print(feature2text(align[i, :audio_len[i]].numpy()))

if __name__ == '__main__':
  test()