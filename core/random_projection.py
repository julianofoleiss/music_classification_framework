import numpy as np
from six import integer_types
import librosa
import scipy.signal
import dill

from .data.predicted_features import PredictedFeatures
from .utils.encodings import to_label_matrix, tags_to_matrix, inv_dict

def get_rp_features(input_file, tags, params=None, output_file=None, **kwargs):

    sr = params['random_projection']['target_sr']
    n_fft = params['random_projection']['n_fft']
    hop_length = params['random_projection']['hop_length']
    projection_matrix = params['random_projection']['projection_matrix']
    non_linearity = params['random_projection']['non_linearity']
    mel_bins = params['random_projection']['mel_bins']
    in_db = params['random_projection']['in_db']
    ld_text = open(params['general']['label_dict_file']).read()
    exec(ld_text)

    y, sr = librosa.load(input_file, sr=sr, mono=True)

    m = np.mean(y)
    v = np.var(y)
    y = (y-m) / np.sqrt(v)

    w = scipy.signal.hamming(n_fft, sym=False)

    spectrum = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, window=w, win_length=n_fft, center=True))

    if mel_bins is not None:
        spectrum = librosa.feature.melspectrogram(S=spectrum, sr=sr, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=mel_bins)

    if in_db:
        spectrum = librosa.power_to_db(spectrum)

    feats = spectrum.T.dot(projection_matrix)

    if non_linearity is not None:
        if non_linearity == 'tanh':
            feats = np.tanh(feats)

    if params['random_projection']['deltas']:
        pad = np.zeros((1, feats.shape[1]))
        d = np.vstack((pad, np.diff(feats, axis=0)))
        if params['random_projection']['delta_deltas']:
            pad = np.zeros((1, d.shape[1]))
            dd = np.vstack((pad, np.diff(d, axis=0)))    
            feats = np.hstack((feats, d, dd))
        else:
            feats = np.hstack((feats, d))

    pf = PredictedFeatures([input_file], [feats.shape[0]], feats, inv_dict(label_dict))
    
    pf.labels = tags_to_matrix([tags], label_dict)
    pf.feature_filenames = [None]

    if output_file is not None:
        dill.dump(pf, open(output_file, 'w'))
        pf = None

    return pf

def get_random_matrix(shape, random_state=None, out_file=None):

    if random_state is None:
        random_state = np.random

    if isinstance(random_state, integer_types):
        random_state = np.random.RandomState(random_state)

    m = random_state.rand(*shape)

    if out_file is not None:
        dill.dump(m, open(out_file, 'w'))
    
    return m


def get_gaussian_random_matrix(mean, stdev, shape, random_state=None, out_file=None):

    if random_state is None:
        random_state = np.random

    if isinstance(random_state, integer_types):
        random_state = np.random.RandomState(random_state)

    m = random_state.normal(mean, stdev, size=shape)

    if out_file is not None:
        dill.dump(m, open(out_file, 'w'))
    
    return m
