import numpy as np
from six import integer_types
from sklearn.decomposition import PCA
import librosa
import scipy.signal
import dill

from .data.predicted_features import PredictedFeatures
from .utils.encodings import to_label_matrix, tags_to_matrix, inv_dict

def get_pca_features(input_file, tags, params=None, output_file=None, **kwargs):

    sr = params['pca_features']['target_sr']
    n_fft = params['pca_features']['n_fft']
    hop_length = params['pca_features']['hop_length']
    mel_bins = params['pca_features']['mel_bins']
    in_db = params['pca_features']['in_db']
    n_components = params['pca_features']['n_components']
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

    pca = PCA(n_components=n_components)

    feats = pca.fit_transform(spectrum.T)

    if params['pca_features']['deltas']:
        pad = np.zeros((1, feats.shape[1]))
        d = np.vstack((pad, np.diff(feats, axis=0)))
        if params['pca_features']['delta_deltas']:
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
