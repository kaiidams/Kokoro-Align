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

Get [Kokoro Speech Dataset](https://github.com/kaiidams/Kokoro-Speech-Dataset) and extract
the data under `./data`.
`./data/kokoro-speech-v1_2-small/metadata.csv` should be
the path of the transcript data.

### Preprocessing

Run this to preprocess Kokoro corpus.

```
$ python -m kokoro_align.prepare \
    --dataset kokoro \
    --data data/kokoro-speech-v1_2-small
```

This generates two files.
`data/kokoro-text.npz` contains phonemes
and `data/kokoro-audio.npz` contains MFCC features.

### Run training

Run this to train the CTC model.

```
$ python -m kokoro_align.train \
    --train --dataset kokoro --model-dir model/ctc
```

It achieve loss similar to this after 100 epochs.

```
train epoch 199: 100% 65/65 [00:20<00:00,  3.16it/s, loss=0.156]
train epoch 199: 100% 8/8 [00:00<00:00, 18.63it/s]
Avg loss: 0.306335
```

## How to build Kokoro-Speech-Dataset

You can use the model trained in the above process or use
[the pretraine model](https://github.com/kaiidams/Kokoro-Align/releases/download/v0.2/ctc-20221201.tar.gz)

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

### Make metadata

```
$ python run_example.py
```

### Copy index

You can use output directory to make datasets with 
[Kokoro Speech Dataset](https://github.com/kaiidams/Kokoro-Speech-Dataset)
using `output` direcotry.

```
$ python run_example.py --copy-index
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
