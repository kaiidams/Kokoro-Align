# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import json
import os
import argparse
import urllib.request
import sys
import re
from glob import glob

DATA_DIR = './data'

def replace_ext(files, fromext, toext):
    return [re.sub(fromext + '$', toext, file) for file in files]

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

    voca_files = replace_ext(audio_files, '.mp3', '.voca.txt')
    for text_file, voca_file in zip(text_files, voca_files):
        if os.path.exists(voca_file):
            print(f'Skip writing {voca_file}')
        else:
            print(f'Writing {voca_file}')
            from voice100.transcript import write_transcript
            write_transcript(text_file, voca_file)

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

    best_path_files = replace_ext(audio_files, '.mp3', '.best_path.npz')
    for logits_file, voca_file, best_path_file in zip(logits_files, voca_files, best_path_files): 
        if os.path.exists(best_path_file):
            print(f'Skip writing {best_path_file}')
        else:
            print(f'Writing {best_path_file}')
            from voice100.align import best_path
            best_path(logits_file, voca_file, best_path_file)

    align_files = replace_ext(audio_files, '.mp3', '.align.txt')
    for best_path_file, mfcc_file, voca_file, align_file in zip(best_path_files, mfcc_files, voca_files, align_files): 
        if os.path.exists(align_file):
            print(f'Skip writing {align_file}')
        else:
            print(f'Writing {align_file}')
            from voice100.align import align
            align(best_path_file, mfcc_file, voca_file, align_file)

    print('Done!')

def main(args):
    with open('example.json') as f:
        example = json.load(f)

    if args.list:
        print("""List of supported dataset name:

    ID                                 Name""")
        for x in example:
            print(f"    {x['id']:35s}{x['name']:10s}")
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
    parser.add_argument('--dataset', default='gongitsune-by-nankichi-niimi', 
        help='Dataset ID to process')
    parser.add_argument('--model-dir', 
        default='./model/ctc-20210319', help='Directory to load checkpoints.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    main(args)