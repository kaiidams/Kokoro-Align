import json
import os
import argparse
import urllib.request

DATA_DIR = './data'

def main(args):
    with open('example.json') as f:
        example = json.load(f)

    example = { x['id']: x for x in example }

    os.makedirs('data', exist_ok=True)

    params = example[args.dataset]
    
    aozora_url = params['aozora_url']
    aozora_file = os.path.join(DATA_DIR, os.path.basename(aozora_url))
    if not os.path.exists(aozora_file):
        urllib.request.urlretrieve(aozora_url, aozora_file)

    
    

    print(aozora_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Dataset ID to process')
    args = parser.parse_args()
    main(args)