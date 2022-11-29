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

Run this to preprocess Kokoro corpus.

```
$ python -m kokoro_align.prepare --dataset kokoro --data [directory]
```

This generates two files.
`data/kokoro-text.npz` contains phonemes
and `data/kokoro-audio.npz` contains MFCC features.

### Run training

Run this to train the CTC model.

```
$ python -m kokoro_align.train --train --dataset kokoro --model-dir model/ctc
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

### Download audio data

```
$ mkdir -p data
$ (cd data && curl -LO http://archive.org/download/gongitsune_um_librivox/gongitsune_um_librivox_64kb_mp3.zip)
$ unzip data/gongitsune_um_librivox_64kb_mp3.zip -d data/gongitsune-by-nankichi-niimi
$ ls data/gongitsune-by-nankichi-niimi/*.mp3 | sort > data/gongitsune_audio_files.txt
$ sed -e 's/\.mp3$/.plain.txt/' data/gongitsune_audio_files.txt > data/gongitsune_original_text_files.txt
```

You can see a shell script to download data by running

```
$ python run_example.py --download --dataset gongitsune-by-nankichi-niimi 
```

### Download transcripts

```
$ mkdir -p data
$ python -m kokoro_align.aozora http://www.aozora.gr.jp/cards/000121/files/628_14895.html $(cat data/gongitsune_original_text_files.txt)
```

### Fix text data manually

Often, the content of the text and the audio doesn't match even they say they read the text.
For example, the text contains meta data like copyrights, date of creation which are not included in the audio.
The audio contains additional information about the audio.

Modifying text files to reduce those mismatches, helps the better results. The previous process downloads
the text as `data/gongitsune-by-nankichi-niimi/*.plain.txt`.

### Preprocessing

This uses MeCab Unidic to get phonemes and save the results.

```
$ python -m kokoro_align.transcript \
    data/gongitsune-by-nankichi-niimi/gongitsune_01_niimi_64kb.plain.txt \
    data/gongitsune-by-nankichi-niimi/gongitsune_01_niimi_64kb.voca.txt 
```

This MFCC features of audio files in `data/gongitsune_audio.npz`.

```
$ python -m kokoro_align.preprocess \
    data/gongitsune-by-nankichi-niimi/gongitsune_01_niimi_64kb.mp3 \
    data/gongitsune-by-nankichi-niimi/gongitsune_01_niimi_64kb.split.txt \
    data/gongitsune-by-nankichi-niimi/gongitsune_01_niimi_64kb.mfcc.npz
```

### Estimate phonemes

This try to predict phonemes from MFCC.

```
$ python -m kokoro_align.train --predict \
--dataset data/gongitsune-by-nankichi-niimi \
--model-dir model/ctc-20210319
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
