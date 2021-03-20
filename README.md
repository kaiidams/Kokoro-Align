# Voice100

Voice100 is a PyTorch speech-transcript alignment tool for LibriVox.
It splits audio files in silent positions and find CTC best path to
align transcript texts with the audio files.

## Objectives

- Don't depend non-commercially licensed dataset

## How to train CTC model

CTC model predicts phonemes from MFCC audio features. You can download
the pretrained model checkpoint and skip this process.

### Download data

Get [CSS10 Japanese corpus](https://github.com/Kyubyong/css10) and extract
the data under `./data`.
`./data/japanese-single-speaker-speech-dataset/transcript.txt` should be
the path to the transcript data.

### Preprocessing

Run this to preprocess CSS10 corpus.

```
$ python -m voice100.preprocess --dataset css10ja
```

This generates two files.
`data/css10ja_text.npz` contains phonemes
and `data/css10ja_audio.npz` contains MFCC features.

### Run training

Run this to train the CTC model.

```
$ python -m voice100.train --train --dataset css10ja --model-dir model/ctc
```

It achieve loss similar to this after 100 epochs.

```
Epoch 100
-------------------------------
loss: 0.191120  [    0/ 6062]
loss: 0.203137  [ 1280/ 6062]
loss: 0.231901  [ 2560/ 6062]
loss: 0.236112  [ 3840/ 6062]
loss: 0.254331  [ 5120/ 6062]
Avg loss: 0.444975 
```

## How to build dataset

### Download data

```
$ mkdir data
$ python -m voice100.aozora http://www.aozora.gr.jp/cards/000121/files/628_14895.html data/gongitsune.txt
```

python -m voice100.aozora https://www.aozora.gr.jp/cards/000148/files/773_14560.html data/kokoro.txt

```
$ (cd data && curl -LO http://archive.org/download/gongitsune_um_librivox/gongitsune_um_librivox_64kb_mp3.zip)
$ unzip data/gongitsune_um_librivox_64kb_mp3.zip -d data/gongitsune_um_librivox_64kb_mp3
$ ls data/gongitsune_um_librivox_64kb_mp3/*.mp3 | sort > data/gongitsune_audio_files.txt
```

### Fix text data manually

Often, the content of the text and the audio doesn't match even they say they read the text.
For example, the text contains meta data like copyrights, date of creation which are not included in the audio.
The audio contains additional information about the audio.

Modifying text files to reduce those mismatch, helps the better results. The previous process download
the text as `data/gongitsune.txt`.

### Preprocessing

This uses MeCab Unidic Lite to get phonemes and save the result in `data/gongitsune_transcript.txt`.



```
$ python -m voice100.transcript --dataset gongitsune
```

This MFCC features of audio files in `data/gongitsune_audio.npz`.

```
$ python -m voice100.preprocess --dataset gongitsune
```

### Estimate phonemes

This try to predict phonemes from MFCC.

```
$ python -m voice100.train --predict --dataset gongitsune --model-dir model/ctc
```

This predict the alignment of audio and text. It takes longer time than the other 
process.

```
$ python -m voice100.align --best_path --dataset gongitsune
```

```
$ python -m voice100.align --align --dataset gongitsune  
```
