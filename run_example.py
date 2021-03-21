# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import json
import os
import argparse
import urllib.request
import sys
import re
from glob import glob

DATA_DIR = './data'
OUTPUT_DIR = './output'

def replace_ext(files, fromext, toext):
    return [re.sub(fromext + '$', toext, file) for file in files]

def combine_files(transcript_file, source_file, align_files, audio_files, segment_files):
    os.makedirs(os.path.dirname(transcript_file), exist_ok=True)
    os.makedirs(os.path.dirname(source_file), exist_ok=True)
    with open(transcript_file, 'wt') as transcript_f:
        with open(source_file, 'wt') as source_f:
            idx = 1
            for align_file, audio_file, segment_file in zip(align_files, audio_files, segment_files): 
                audio_file = os.path.basename(audio_file)
                with open(align_file, 'rt') as align_f:
                    with open(segment_file, 'rt') as segment_f:
                        start_frame = 0
                        for align, segment in zip(align_f, segment_f):
                            align_parts = align.rstrip('\r\n').split('|')
                            segment_parts = segment.rstrip('\r\n').split('|')
                            _, text, voca, score = align_parts
                            end_frame, = segment_parts
                            transcript_f.write(f'{args.dataset}-{idx}|{text}|{voca}|{score}\n')
                            source_f.write(f'{args.dataset}-{idx}|{audio_file}|{start_frame}|{end_frame}\n')
                            idx += 1
                            start_frame = end_frame

def write_wav_files(audio_dir, source_file, output_dir, expected_sample_rate=22050):
    import torchaudio
    with open(source_file, 'rt') as f:
        current_file = None
        current_audio = None
        for line in f:
            parts = line.rstrip('\r\n').split('|')
            id_, audio_file, audio_start, audio_end = parts
            audio_start, audio_end = int(audio_start), int(audio_end)
            if current_file != audio_file:
                file = os.path.join(audio_dir, audio_file)
                print(f'Reading {file}')
                y, sr = torchaudio.load(file)
                assert len(y.shape) == 2 and y.shape[0] == 1
                assert sr == expected_sample_rate
                current_file = audio_file
                current_audio = y
            output_file = os.path.join(output_dir, f'{id_}.wav')
            print(f'Writing {output_file}')
            y = current_audio[:, audio_start:audio_end]
            torchaudio.save(output_file, y, expected_sample_rate)

def process(args, params):

    ##################################################
    # Check if audio files are available locally
    ##################################################

    archive_url = params['archive_url']
    audio_dir = os.path.join(DATA_DIR, re.sub('.zip$', '', os.path.basename(archive_url)))
    if not os.path.exists(audio_dir):
        print(f"""Audio files are missing. Please download audio files from
{archive_url} 
and extract the archive file in `{audio_dir}'. This scripts
read audio files from `{audio_dir}/*.mp3'.""")
        sys.exit(1)
    else:
        audio_files = sorted(glob(os.path.join(audio_dir, '*.mp3')))

    ##################################################
    # Get text data from Aozora
    ##################################################

    aozora_url = params['aozora_url']
    aozora_file = os.path.join(audio_dir, os.path.basename(aozora_url))
    if glob(os.path.join(aozora_file)):
        print(f"Skip downloading Aozora HTML from `{aozora_url}'.")
    if not os.path.exists(aozora_file):
        print(f"Downloading Aozora HTML from `{aozora_url}'.")
        urllib.request.urlretrieve(aozora_url, aozora_file)

    ##################################################
    # Convert Aozora HTML to text
    ##################################################

    text_files = replace_ext(audio_files, '.mp3', '.plain.txt')
    if all(os.path.exists(file) for file in text_files):
        print(f'Skip converting Aozora HTML to text files')
    else:
        print(f'Converting Aozora HTML to text files')
        from voice100.aozora import convert_aozora
        convert_aozora(aozora_file, text_files)

    ##################################################
    # Convert text to phonemes
    ##################################################

    voca_files = replace_ext(audio_files, '.mp3', '.voca.txt')
    for text_file, voca_file in zip(text_files, voca_files):
        if os.path.exists(voca_file):
            print(f'Skip writing {voca_file}')
        else:
            print(f'Writing {voca_file}')
            from voice100.transcript import write_transcript
            write_transcript(text_file, voca_file)

    ##################################################
    # Convert audio to MFCC
    ##################################################

    mfcc_files = replace_ext(audio_files, '.mp3', '.mfcc.npz')
    segment_files = replace_ext(audio_files, '.mp3', '.split.txt')
    for audio_file, segment_file, mfcc_file in zip(audio_files, segment_files, mfcc_files):
        if os.path.exists(mfcc_file):
            print(f'Skip converting {audio_file} to MFCC')
        else:
            print(f'Converting to {audio_file} MFCC')
            from voice100.preprocess import split_audio
            split_audio(
                audio_file, segment_file, mfcc_file
            )

    ##################################################
    # Predict phonemes from audio
    ##################################################

    logits_files = replace_ext(audio_files, '.mp3', '.logits.npz')
    greed_files = replace_ext(audio_files, '.mp3', '.greed.txt')
    for mfcc_file, logits_file, greed_file in zip(mfcc_files, logits_files, greed_files): 
        if os.path.exists(logits_file):
            print(f'Skip predicting phonemes of {mfcc_file}')
        else:
            print(f'Predicting phonemes of {mfcc_file}')
            from voice100.train import predict
            import torch
            use_cuda = not args.no_cuda and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
            cargs = argparse.Namespace()
            cargs.audio = mfcc_file
            cargs.output = logits_file
            cargs.text = greed_file
            cargs.batch_size = args.batch_size
            cargs.model_dir = args.model_dir
            predict(cargs, device)

    ##################################################
    # Predict the best path
    ##################################################

    best_path_files = replace_ext(audio_files, '.mp3', '.best_path.npz')
    for logits_file, voca_file, best_path_file in zip(logits_files, voca_files, best_path_files): 
        if os.path.exists(best_path_file):
            print(f'Skip writing {best_path_file}')
        else:
            print(f'Writing {best_path_file}')
            from voice100.align import best_path
            best_path(logits_file, voca_file, best_path_file)

    ##################################################
    # Writing alignment file
    ##################################################

    align_files = replace_ext(audio_files, '.mp3', '.align.txt')
    for best_path_file, mfcc_file, voca_file, align_file in zip(best_path_files, mfcc_files, voca_files, align_files): 
        if os.path.exists(align_file):
            print(f'Skip writing {align_file}')
        else:
            print(f'Writing {align_file}')
            from voice100.align import align
            align(best_path_file, mfcc_file, voca_file, align_file)

    ##################################################
    # Combine files
    ##################################################

    transcript_file = os.path.join(OUTPUT_DIR, f'{args.dataset}.transcript.txt')
    source_file = os.path.join(OUTPUT_DIR, f'{args.dataset}.source.txt')
    if os.path.exists(transcript_file) and os.path.exists(source_file):
        print(f"Skip writing {transcript_file}")
        print(f"    and {source_file}")
    else:
        print(f"Writing {transcript_file}")
        print(f"    and {source_file}")
        combine_files(transcript_file, source_file, align_files, audio_files, segment_files)

    ##################################################
    # Write WAV files
    ##################################################

    wav_files = glob(os.path.join(OUTPUT_DIR, f'{args.dataset}-*.wav'))
    if wav_files:
        print(f"There are some WAV files `{wav_files[0]}'")
        print(f"Skip writing WAV files")
    else:
        write_wav_files(audio_dir, source_file, OUTPUT_DIR)

    print('All done!')

def main(args):
    with open('example.json') as f:
        example = json.load(f)

    if args.list:
        print("""List of supported dataset name:

    ID                                 Time      Name      """)
        for x in example:
            print(f"    {x['id']:35s}{x['totaltime']:10s}{x['name']:10s}")
    elif args.download:
        for x in example:
            print(f"curl -LO {x['aozora_url']}")
            print(f"curl -LO {x['archive_url']}")
    else:
        example = { x['id']: x for x in example }
        os.makedirs('data', exist_ok=True)
        params = example[args.dataset]
        process(args, params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--list', action='store_true', help='List supported dataset ID.')
    parser.add_argument('--download', action='store_true', help='Print download script.')
    parser.add_argument('--dataset', default='gongitsune-by-nankichi-niimi', 
        help='Dataset ID to process')
    parser.add_argument('--model-dir', 
        default='./model/ctc-20210319', help='Directory to load checkpoints.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    main(args)