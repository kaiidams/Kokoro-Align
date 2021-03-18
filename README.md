# Voice100

## Objectives

- Don't depend non-commercially licensed dataset
- Small enough to run on normal PCs, Raspberry Pi or smartphones.

# Sample synthesis

- [Sample synthesis 1 (From eval datset)](docs/sample1.wav)
- [Sample synthesis 2 (From JVS corpus)](docs/sample2.wav)

## How to build dataset

### Get text data

```
$ mkdir data
$ python -m voice100.aozora http://www.aozora.gr.jp/cards/000121/files/628_14895.html data/gongitsune.txt
```

```
$ (cd data && curl -LO http://archive.org/download/gongitsune_um_librivox/gongitsune_um_librivox_64kb_mp3.zip)
$ unzip data/gongitsune_um_librivox_64kb_mp3.zip -d data/gongitsune_um_librivox_64kb_mp3
$ ls data/gongitsune_um_librivox_64kb_mp3/*.mp3 | sort > data/gongitsune_audio_files.txt
```

### Encode and split data

```
$ python -m voice100.preprocess --dataset gongitsune
```

### Estimate phonemes

```
$ python -m voice100.segmentation --phoneme --dataset gongitsune
```

```
$ python -m voice100.segmentation --best_path --dataset gongitsune
```

python -m voice100.transcript

```
$ python -m voice100.segmentation --align --dataset gongitsune  
```

## How to train

### Preprocessing

Get CSS10 Japanese corpus and extract the data under `./data`.
`./data/japanese-single-speaker-speech-dataset/transcript.txt` should be
the path to the transcript data.

Run the preprocess,

```
$ python -m voice100.preprocess --dataset css10ja
```

This generates `data/css10ja_train.npz` and `data/css10ja_val.npz`

### Training alighment model

![Training CTC](./docs/train_ctc.png)

The alignment model align text and audio of the dataset.

```
$ python -m voice100.train_ctc --mode train --dataset css10ja --model_dir model/ctc
```

### Estimate alighment

This makes `data/css10ja_train_aligh.npz`.

```
$ python -m voice100.train_ctc --mode convert --dataset css10ja --model_dir model/ctc
```

### Train TTS model

TTS model is a plain Transformer with multitask of three tasks,
predicting alignment, audio and end of audio.

```
$ python -m voice100.train_ctc --mode train --model_dir model/audio
```

It takes 3.5 hours with T4 to train 239 epochs.

![train_loss_align](./docs/train_loss_align.png)
![train_loss_audio](./docs/train_loss_audio.png)

## Data

https://raw.githubusercontent.com/voice-statistics/voice-statistics.github.com/master/assets/doc/balance_sentences.txt

- JSUT is 10 hour recording in female voice.
- CSS10 is 14 hour recording in male voice.
- JVS is 0.1 hour recording.

## References

- 声優統計コーパス http://voice-statistics.github.io/
- CSS10: A Collection of Single Speaker Speech Datasets for 10 Languages https://github.com/Kyubyong/css10
- Deep Speech 3 https://arxiv.org/abs/1710.07654
- Tacotron 2 https://arxiv.org/abs/1712.05884
- Tacotron https://github.com/keithito/tacotron
- Tacotron 3 https://github.com/StevenLOL/tacotron-3
- Tacotron 3 https://github.com/kingulight/Tacotron-3
- Mellotron https://github.com/NVIDIA/mellotron
- Deep Voice 3 https://github.com/Kyubyong/deepvoice3
- WORLD http://www.kki.yamanashi.ac.jp/~mmorise/world/
- OpenJTALK http://open-jtalk.sp.nitech.ac.jp/
- 月ノ美兎さんの音声合成ツール(Text To Speech) を作ってみた https://qiita.com/K2_ML/items/2804594454b39180c909
- Mozilla TTS (Tacotron2) を使って日本語音声合成 https://qiita.com/tset-tset-tset/items/7b388b0536fcc774b2ad
- Tacotron2系における日本語のunidecodeの不確かさ https://qiita.com/nishiha/items/6e2a2ddaafe03fa7f924
- Tacotron2を日本語で学習してみる（０から学習編） https://shirowanisan.com/entry/2020/12/05/184426
- NVIDIA/tacotron2 で日本語の音声合成を試す https://note.com/npaka/n/n2a91c3ca9f34
- JSUT https://sites.google.com/site/shinnosuketakamichi/publication/jsut