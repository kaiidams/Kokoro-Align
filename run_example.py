# Copyright (C) 2021 Katsuya Iida. All rights reserved.

import json
import os
import argparse
import urllib.request
import sys
import re
from glob import glob

MODEL_URL = "https://github.com/kaiidams/Kokoro-Align/releases/download/0.0.2/ctc-20210319.tar.gz"


def replace_ext(files, fromext, toext):
    return [re.sub(f'\\.{fromext}$', f'.{toext}', file) for file in files]


def list_datasets(params_list):
    print("""List of supported dataset name:

    ID                                       Time      Name""")
    for params in params_list:
        id_ = params['id']
        totaltime = params['totaltime']
        name = params['name']
        print(f"    {id_:41s}{totaltime:10s}{name:10s}")


def download_script(data_dir, dataset, params_list):
    r"""Prints a shell script to download original audio files.
    """
    print(f'cd {data_dir}')
    print()
    if not dataset:
        print(f"curl -LO {MODEL_URL}")
        print()
    for params in params_list:
        id_ = params['id']
        if not dataset or id_ == dataset:
            print(f"curl -LO {params['aozora_url']}")
    print()
    for params in params_list:
        id_ = params['id']
        if not dataset or id_ == dataset:
            print(f"curl -LO {params['archive_url']}")
    print()
    for params in params_list:
        id_ = params['id']
        if not dataset or id_ == dataset:
            archive_url = params['archive_url']
            archive_file = os.path.basename(archive_url)
            print(f"unzip {archive_file} -d {id_}")


def combine_files(dataset, align_files, audio_files, segment_files, metadata_file):

    from kokoro_align.encoder import is_valid_text, encode_text

    ng_list = [
        'リブリボックス',
        'ボランティアについてなど',
        'この録音はパブリックドメイン',
        'ために録音されました',
    ]

    def block_text_voca(text, voca):
        if text.strip() and voca.strip():
            x = text.replace(' ', '')
            return any(y in x for y in ng_list)
        return True

    def block_voca_decoded(voca, decoded):
        """Check if more than 70% of decoded labels match with the original transcript
        """
        voca_len = len(encode_text(voca))
        decoded_len = len(decoded.split())
        return not (voca_len and decoded_len and decoded_len / voca_len > 0.7)

    def block_unknown_yomi(voca):
        return not is_valid_text(voca)

    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    try:
        with open(metadata_file, 'wt') as metadata_f:
            idx = 1
            for align_file, audio_file, segment_file in zip(align_files, audio_files, segment_files): 
                audio_file = os.path.basename(audio_file)
                with open(align_file, 'rt') as align_f:
                    with open(segment_file, 'rt') as segment_f:
                        start_frame = 0
                        for align, segment in zip(align_f, segment_f):
                            align_parts = align.rstrip('\r\n').split('|')
                            segment_parts = segment.rstrip('\r\n').split('|')
                            _, text, voca, decoded, _, _, _ = align_parts                            
                            end_frame, = segment_parts
                            if block_text_voca(text, voca):
                                print(f'Blocking by NG word: {text}')
                            elif block_unknown_yomi(voca):
                                print(f'Blocking by unknown yomi {voca}')
                            elif block_voca_decoded(voca, decoded):
                                print('Blocking by too few match')
                                print(f'voca:    {voca}')
                                print(f'decoded: {decoded}')
                            else:
                                id_ = f'{dataset}-{idx:05d}'
                                metadata_f.write(f'{id_}|{audio_file}|{start_frame}|{end_frame}|{text}|{voca}\n')
                                idx += 1
                            start_frame = end_frame
    except:
        os.unlink(metadata_file)
        raise


def copy_index(params_list, index_file):
    params_list = [
        params for params in params_list if params['enabled']
    ]
    try:
        with open(index_file, 'wt') as f:
            json.dump(params_list, f, indent=4, ensure_ascii=False)
    except:
        os.unlink(index_file)
        raise


def process(args, params):

    if not params['enabled']:
        print(f"Skipping disabled {params['id']}.")
        return

    ##################################################
    # Check if audio files are available locally
    ##################################################

    archive_url = params['archive_url']
    audio_dir = os.path.join(args.data_dir, params['id'])
    if not os.path.exists(audio_dir):
        print(f"""Audio files are missing. Please download the archive file from
{archive_url} 
and extract files in `{audio_dir}'. This scripts
read audio files from `{audio_dir}/*.mp3'.""")
        sys.exit(1)
    else:
        audio_files = sorted(glob(os.path.join(audio_dir, '*.mp3')))

    ##################################################
    # Get text data from Aozora
    ##################################################

    aozora_url = params['aozora_url']
    aozora_file = os.path.join(args.data_dir, os.path.basename(aozora_url))
    if glob(os.path.join(aozora_file)):
        print(f"Skip downloading Aozora HTML from `{aozora_url}'.")
    if not os.path.exists(aozora_file):
        print(f"Downloading Aozora HTML from `{aozora_url}'.")
        urllib.request.urlretrieve(aozora_url, aozora_file)

    ##################################################
    # Convert Aozora HTML to text
    ##################################################

    text_files = replace_ext(audio_files, 'mp3', 'plain.txt')
    if all(os.path.exists(file) for file in text_files):
        print(f'Skip converting Aozora HTML to text files')
    else:
        print(f'Converting Aozora HTML to text files')
        from kokoro_align.aozora import convert_aozora
        convert_aozora(aozora_file, text_files)

    ##################################################
    # Convert text to phonemes
    ##################################################

    voca_files = replace_ext(audio_files, 'mp3', 'voca.txt')
    for text_file, voca_file in zip(text_files, voca_files):
        if os.path.exists(voca_file):
            print(f'Skip writing {voca_file}')
        else:
            print(f'Writing {voca_file}')
            from kokoro_align.transcript import write_transcript
            write_transcript(text_file, voca_file)

    ##################################################
    # Convert audio to MFCC
    ##################################################

    mfcc_files = replace_ext(audio_files, 'mp3', 'mfcc.npz')
    segment_files = replace_ext(audio_files, 'mp3', 'split.txt')
    for audio_file, segment_file, mfcc_file in zip(audio_files, segment_files, mfcc_files):
        if os.path.exists(segment_file) and os.path.exists(mfcc_file):
            print(f'Skip converting {audio_file} to MFCC')
        else:
            print(f'Converting {audio_file} to MFCC')
            from kokoro_align.preprocess import split_audio
            split_audio(
                audio_file, segment_file, mfcc_file
            )

    ##################################################
    # Predict phonemes from audio
    ##################################################

    logits_files = replace_ext(audio_files, 'mp3', 'logits.npz')
    greed_files = replace_ext(audio_files, 'mp3', 'greed.txt')
    for mfcc_file, logits_file, greed_file in zip(mfcc_files, logits_files, greed_files): 
        if os.path.exists(logits_file) and os.path.exists(greed_file):
            print(f'Skip predicting phonemes of {mfcc_file}')
        else:
            print(f'Predicting phonemes of {mfcc_file}')
            from kokoro_align.train import predict
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

    best_path_files = replace_ext(audio_files, 'mp3', 'best_path.npz')
    for logits_file, voca_file, best_path_file in zip(logits_files, voca_files, best_path_files): 
        if os.path.exists(best_path_file):
            print(f'Skip writing {best_path_file}')
        else:
            print(f'Writing {best_path_file}')
            from kokoro_align.align import best_path
            best_path(logits_file, voca_file, best_path_file)

    ##################################################
    # Writing alignment file
    ##################################################

    align_files = replace_ext(audio_files, 'mp3', 'align.txt')
    for best_path_file, mfcc_file, voca_file, align_file in zip(best_path_files, mfcc_files, voca_files, align_files): 
        if os.path.exists(align_file):
            print(f'Skip writing {align_file}')
        else:
            print(f'Writing {align_file}')
            from kokoro_align.align import align
            align(best_path_file, mfcc_file, voca_file, align_file)

    ##################################################
    # Write metadata
    ##################################################

    metadata_file = os.path.join(args.output_dir, f"{params['id']}.metadata.txt")
    if os.path.exists(metadata_file):
        print(f"Skip writing {metadata_file}")
    else:
        print(f"Writing {metadata_file}")
        combine_files(params['id'], align_files, audio_files, segment_files, metadata_file)

    print('Done!')


def main(args):
    with open('example.json') as f:
        params_list = json.load(f)

    if args.dataset:
        if not any(params['id'] == args.dataset for params in params_list):
            print(f"Unknown dataset `{args.dataset}'")
            sys.exit(1)

    if args.list:
        list_datasets(params_list)
    elif args.download:
        download_script(args.data_dir, args.dataset, params_list)
    elif args.copy_index:
        index_file = os.path.join(args.output_dir, 'index.json')
        copy_index(params_list, index_file)
    else:
        for params in params_list:
            if not args.dataset or params['id'] == args.dataset:
                process(args, params)


def main_cli():
    parser = argparse.ArgumentParser(description="Kokoro-Align is a speech-text aligner.")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--list', action='store_true', help='List supported dataset ID.')
    parser.add_argument('--download', action='store_true', help='Prints a shell script to download original audio files.')
    parser.add_argument('--copy-index', action='store_true', help='Copy index file.')
    parser.add_argument('--dataset', help='Dataset ID to process')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--model-dir', 
        default='./model/ctc-20221201', help='Directory to load checkpoints.')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    main_cli()