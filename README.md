# Kokoro-Align

Kokoro-Align is a PyTorch speech-transcript alignment tool for LibriVox.
It splits audio files in silent positions and find CTC best path to
align transcript texts with the audio files. Kokoro-Align is used for
[Kokoro Speech Dataset](https://github.com/kaiidams/Kokoro-Speech-Dataset).

## Objectives

- Not depend on non-commercially licensed datasets

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
$ python -m kokoro_align.preprocess --dataset css10ja
```

This generates two files.
`data/css10ja_text.npz` contains phonemes
and `data/css10ja_audio.npz` contains MFCC features.

### Run training

Run this to train the CTC model.

```
$ python -m kokoro_align.train --train --dataset css10ja --model-dir model/ctc
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

## How to build Kokoro-Speech-Dataset

### Download data

```
$ mkdir data
$ python -m kokoro_align.aozora http://www.aozora.gr.jp/cards/000121/files/628_14895.html data/gongitsune.txt
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
$ python -m kokoro_align.transcript --dataset gongitsune
```

This MFCC features of audio files in `data/gongitsune_audio.npz`.

```
$ python -m kokoro_align.preprocess --dataset gongitsune
```

### Estimate phonemes

This try to predict phonemes from MFCC.

```
$ python -m kokoro_align.train --predict --dataset gongitsune --model-dir model/ctc
```

This predict the alignment of audio and text. It takes longer time than the other 
process.

```
$ python -m kokoro_align.align --best_path --dataset gongitsune
```

```
$ python -m kokoro_align.align --align --dataset gongitsune  
```

### Dataset

- [明暗 (Meian)](https://librivox.org/meian-by-soseki-natsume/) 16:39:29 
    [Online text](http://www.aozora.gr.jp/cards/000148/files/782_14969.html)
- [こころ (Kokoro)](https://librivox.org/kokoro-by-soseki-natsume/) 08:46:41
    [Online text](http://www.aozora.gr.jp/cards/000148/files/773_14560.html)
- [雁 (Gan)](https://librivox.org/gan-by-ogai-mori/) 03:41:31
    [Online text](http://www.aozora.gr.jp/cards/000129/files/45224_19919.html)
- [草枕 (Kusamakura)](https://librivox.org/kusamakura-by-soseki-natsume/) 04:27:35
    [Online text](http://www.aozora.gr.jp/cards/000148/files/776_14941.html)
- [田舎教師 (Inaka Kyoshi)](https://librivox.org/inakakyoshi-by-katai-tayama/) 08:13:26
    [Online text](http://www.aozora.gr.jp/cards/000214/files/1668_26031.html)
- [坊っちゃん (Botchan)](https://librivox.org/botchan-by-soseki-natsume-2/) 04:26:27
    [Online text](http://www.aozora.gr.jp/cards/000148/files/752_14964.html)
- [野分 (Nowaki)](https://librivox.org/nowaki-by-soseki-natsume/) 4:40:49
    [Online text](http://www.aozora.gr.jp/cards/000148/files/791_14959.html)
- [ごん狐 (Gon gitsune)](https://librivox.org/gongitsune-by-nankichi-niimi/) 0:15:42
    [Online text](http://www.aozora.gr.jp/cards/000121/files/628_14895.html)
- [コーカサスの禿鷹 (Caucasus no Hagetaka)](https://librivox.org/caucasus-no-hagetaka-by-yoshio-toyoshima/) 0:13:04
    [Online text](http://www.aozora.gr.jp/cards/000906/files/42633_22951.html)
