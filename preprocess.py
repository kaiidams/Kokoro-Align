import soundfile as sf
import os
import numpy as np
import pickle
import librosa
from tqdm import tqdm

from vocoder import estimatef0, analyze, synthesize

CORPUSDATA_PATH = 'data/balance_sentences.txt'

WAVDATA_PATH = {
    'tsuchiya_normal': 'data/tsuchiya_normal/tsuchiya_normal_%s.wav',
    'hiroshiba_normal': 'data/hiroshiba_normal/hiroshiba_normal_%s.wav',
    'tsukuyomi_normal': 'data/つくよみちゃんコーパス Vol.1 声優統計コーパス（JVSコーパス準拠）'
        '/01 WAV（収録時の音量のまま）/VOICEACTRESS100_%s.wav',
}

OUTPUT_PATH = 'data/train_%s.pkl'
WAVTEST_PATH = 'data/%s_synthesized/%s_%s.wav'

SAMPLE_RATE = 16000

vocab = list(':Nabcdefghijkmnopqrstuwyz')
v2i = {v: i for i, v in enumerate(vocab)}
assert len(v2i) == 25

normparams = np.array([
    [282.67361826246565, 236.92293935472185],
    [11.328978802837387, -7.109625552021011],
    [4.126377353770099, 1.533374557597003],
    [2.6263609817338622, 0.27137485238784176],
    [2.0097988720757587, 0.29294477661677437],
    [1.543488852405658, -0.1280490237234923],
    [1.6440655610450512, 0.22163170362627446],
    [1.1526827715084913, -0.24250710981627668],
    [1.2658236104277272, 0.0655040663453442],
    [0.9866418739238294, -0.0821013549885241],
    [1.0882266363704751, 0.0348202786052773],
    [0.9928068293429729, -0.07903293503297537],
    [1.0148471211922974, 0.04406074528566904],
    [0.9052042182215183, -0.10379131476798364],
    [0.7913949640830251, 0.14247308649223117],
    [0.7066236644344697, -0.07754881090351255],
    [0.6164336833717089, 0.05662846233828259],
    [0.6834507819242062, -0.08894137181572302],
    [0.6112295659013609, 0.09387329502622088],
    [0.5517705959994136, -0.08169052174503225],
    [0.5300752465529857, 0.07638933704332687],
    [0.5266028739836691, -0.06978268403081882],
    [0.540707296265464, 0.08316265927423938],
    [0.4468212622714532, -0.08611203250164863],
    [0.40220457772049667, 0.07025092383589981],
    [0.48744988996484434, -0.04962998073599184],
    [32.80534646915437, -3.0824670989608625],
], dtype=np.float)

def readcorpus(file):
    corpus = []
    with open(file) as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            id_, _, _, monophone = parts
            monophone = monophone.replace('/', '').replace(',', '')
            corpus.append((id_, monophone))

    if False:
        vocab = set(''.join([monophone for _, monophone in corpus]))
        vocab = sorted(list(vocab))
        assert vocab == list(':Nabcdefghijkmnopqrstuwyz')
        v2i = {v: i for i, v in enumerate(vocab)}

    return corpus

def readwav(file, fs):
    x, origfs = sf.read(file)
    x = x / np.max(x)
    x = librosa.resample(x, origfs, fs)
    return x

def writewav(file, x, fs):
    sf.write(file, x, fs, 'PCM_16')

def text2feature(text):
    return np.array([v2i[ch] for ch in text], dtype=np.int)

def wav2feature(x, fs, normed=True):
    f0, mcep, codeap = analyze(x, fs)
    feature = np.hstack((f0.reshape((-1, 1)), mcep, codeap))
    # Normalize
    if normed:
        feature = (feature - normparams[:, 1]) / normparams[:, 0]
    return feature.astype(np.float32)

def feature2wav(feature):
    feature = feature.astype(np.float)
    # Unnormalize
    feature = normparams[:, 0] * feature + normparams[:, 1]
    f0 = feature[:, 0].copy()
    mcep = feature[:, 1:26].copy()
    codeap = feature[:, 26:].copy()
    y = synthesize(f0, mcep, codeap)
    return y

def analyzef0(name, fs=SAMPLE_RATE):
    corpus = readcorpus(CORPUSDATA_PATH)
    data = {
        'id': [],
        'text': [],
        'audio': []
    }
    l = []
    for id_, monophone in tqdm(corpus[:10]):
        assert len(id_) == 3
        file = WAVDATA_PATH[name] % id_
        x = readwav(file, fs)
        f0 = estimatef0(x, fs)
        f0 = f0[f0 > 0].copy()
        l.append(f0)
    f0 = np.concatenate(l)
    return f0

def analyzerange(name, fs=SAMPLE_RATE):
    corpus = readcorpus(CORPUSDATA_PATH)
    data = {
        'id': [],
        'text': [],
        'audio': []
    }

    xmin = []
    xmax = []
    xsum = np.zeros(27)
    n = 0

    for id_, monophone in tqdm(corpus):
        assert len(id_) == 3
        file = WAVDATA_PATH[name] % id_
        x = readwav(file, fs)
        feature = wav2feature(x, fs, normed=False)
        xmin.append(np.min(feature, axis=0))
        xmax.append(np.max(feature, axis=0))
        xsum += np.sum(feature, axis=0)
        n += feature.shape[0]
    xmin = np.min(xmin, axis=0)
    xmax = np.max(xmax, axis=0)
    xmean = xsum / n

    alpha = np.max([xmean - xmin, xmax - xmean], axis=0)
    beta = xmean
    for i in range(27):
        print(f'    [{alpha[i]}, {beta[i]}],')
    print()

def preprocess(name, fs=SAMPLE_RATE):
    corpus = readcorpus(CORPUSDATA_PATH)
    data = {
        'id': [],
        'text': [],
        'audio': []
    }
    for id_, monophone in tqdm(corpus):
        assert len(id_) == 3
        file = WAVDATA_PATH[name] % id_
        x = readwav(file, fs)
        data['id'].append(id_)
        data['text'].append(text2feature(monophone))
        data['audio'].append(wav2feature(x, fs))

    output_file = OUTPUT_PATH % name
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

def test(name, fs=SAMPLE_RATE):
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

if __name__ == '__main__':
    preprocess('tsukuyomi_normal')
    test('tsukuyomi_normal')
    #test('tsuchiya_normal')
    #preprocess('tsuchiya_normal')
