def verify_jvs(name, fs=SAMPLE_RATE):
    output_file = OUTPUT_PATH % name
    with open(output_file, 'rb') as f:
        data = pickle.load(f)
    print(len(data['text']))
    print(len(data['audio']))
    for i in tqdm(range(10)):
        id_ = data['id'][i]
        x = feature2wav(data['audio'][i])
        file = WAVTEST_PATH % (name, name, id_)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writewav(file, x, fs)

def verify_css10ja(name='css10ja'):
    fs = 16000
    file = 'data/css10ja_train.npz'
    f = np.load(file)
    data = {k:v for k, v in f.items()}
    for index in tqdm(range(10)):
        text_start = data['text_index'][index - 1] if index else 0
        text_end = data['text_index'][index]
        audio_start = data['audio_index'][index - 1] if index else 0
        audio_end = data['audio_index'][index]
        text = data['text_data'][text_start:text_end]
        audio = data['audio_data'][audio_start:audio_end, :]
        print(feature2text(text))
        x = feature2wav(audio)
        file = 'data/%s_synthesized/meian_%04d.wav' % (name, index)
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writewav(file, x, fs)
