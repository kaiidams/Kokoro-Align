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

def main(args):
    with open('example.json') as f:
        example = json.load(f)

    example = { x['id']: x for x in example }

    os.makedirs('data', exist_ok=True)

    params = example[args.dataset]
    
    # Get text data from Aozora
    aozora_url = params['aozora_url']
    aozora_file = os.path.join(DATA_DIR, os.path.basename(aozora_url))
    if not os.path.exists(aozora_file):
        urllib.request.urlretrieve(aozora_url, aozora_file)

    audio_dir = os.path.join(DATA_DIR, re.sub('.zip$', '', os.path.basename(params['archive_url'])))
    if not os.path.exists(audio_dir):
        print(f"Download audio files from {params['archive_url']}")
        sys.exit(1)

    audio_files = sorted(glob(os.path.join(audio_dir, '*.mp3')))

    process_file = os.path.join(DATA_DIR, f'{args.dataset}.txt')
    text_files = replace_ext(audio_files, '.mp3', '.plain.txt')
    if os.path.exists(process_file):
        print(f'Skip writing {process_file}')
    else:
        print(f'Writing {process_file}')
        audio_files = sorted(glob(os.path.join(audio_dir, '*.mp3')))
        with open(process_file, 'wt') as f:
            for x, y in zip(text_files, audio_files):
                f.write(f'{x}|{y}\n')

    if glob(os.path.join(audio_dir, '*.plain.txt')):
        print(f'Skip converting Aozora HTML to text files')
    else:
        print(f'Converting Aozora HTML to text files')
        import voice100.aozora
        cargs = argparse.Namespace()
        cargs.input = process_file
        cargs.aozora = aozora_file
        voice100.aozora.main(cargs)

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
    for mfcc_file, logits_file, greed_file in zip(mfcc_files, logits_files, greed_files): 
        if os.path.exists(logits_file):
            print(f'Skip predicting best CTC path of {logits_file}')
        else:
            print(f'Predicting best CTC path of {logits_file}')

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dataset', default='gongitsune-by-nankichi-niimi', 
        help='Dataset ID to process')
    parser.add_argument('--model-dir', 
        default='./model/ctc-20210319', help='Directory to load checkpoints.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    main(args)