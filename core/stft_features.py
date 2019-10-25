# -*- coding: utf-8 -*-

import librosa
import numpy as np
import dill, sys
import importlib
import scipy
from .data.predicted_features import PredictedFeatures
from .utils.encodings import to_label_matrix, tags_to_matrix, inv_dict
from time import time

def get_stft_features(input_file, tags, params=None, output_file=None):
    # type (str, list[str], str) -> PredictedFeatures
    """
    Computes the STFT and returns it as PredictedFeatures.

    :param input_file: path to input audio file
    :param tags: a list of strings for each tag
    :param output_file: path to output feature file. If None, returns PredictedFeatures instance.
    :returns: if output_file is None, return PredictedFeatures instance. Else, returns None.

    :type input_file: str
    :type tags: list[str]
    :type output_file: str
    :rtype: PredictedFeatures
input
    """

    n_fft = params['stft_features']['n_fft'] 
    hop_length = params['stft_features']['hop_length'] 
    target_sr = params['stft_features']['target_sr']
    mel_bins = params['stft_features']['mel_bins']
    ld_text = open(params['general']['label_dict_file']).read()
    exec(ld_text)

    #print("Calculating Features for %s..." % input_file)
    st =time()

    y, sr = librosa.load(input_file, sr=target_sr, mono=True)

    #centering data around mean zero, std_dev 1
    m = np.mean(y)
    v = np.var(y)
    y = (y-m) / np.sqrt(v)

    w = scipy.signal.hamming(n_fft, sym=False)

    stft = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, window=w, win_length=n_fft, center=True))

    if mel_bins is not None:
        stft = librosa.feature.melspectrogram(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_length, power=2.0, n_mels=mel_bins)

    feats = stft

    if params['stft_features']['deltas']:
        d = librosa.feature.delta(feats, order=1)
    else:
        d = np.empty((0,feats.shape[1]))

    if params['stft_features']['delta_deltas']:
        dd = librosa.feature.delta(feats, order=2)
    else:
        dd = np.empty((0,feats.shape[1]))

    feats = np.vstack((feats, d, dd))

    pf = PredictedFeatures([input_file], [feats.shape[1]], feats.T, inv_dict(label_dict))
    
    pf.labels = tags_to_matrix([tags], label_dict)
    pf.feature_filenames = [None]

    if output_file is not None:
        dill.dump(pf, open(output_file, 'w'))
        pf = None

    return pf

if __name__ == '__main__':
    import yaml
    from core.texture_avg_stdev_filter import TextureAvgStdevFilter
    params = yaml.load( open('parameters.yaml').read() )

    #feats = get_gtzan_features('/home/juliano/Doutorado/datasets/gtzan44/rock_00000.wav', ['rock'], params=params, output_file='rock_00000.feats' )

    txt = TextureAvgStdevFilter(params)
    feats = get_gtzan_features('/home/juliano/Doutorado/datasets/gtzan44/blues_00099.wav', ['rock'], params=params, output_file=None )
    
    print feats.estimator_output.shape

    textured = txt.filter(feats)

    print textured.estimator_output.shape, textured.estimator_output.dtype
    dill.dump(textured.estimator_output, open('blues99.allfeats', 'w'))

    #print textured.estimator_output.shape

    #from core.beat_filter import BeatAggregatorFilter

    #ba = BeatAggregatorFilter(params)

    #agg = ba.filter(textured, audio_filename='/home/juliano/Doutorado/datasets/gtzan44/rock_00000.wav', sr=44100, hop_length=1024)

    #print agg.estimator_output.shape
    #print textured.estimator_output.dtype

    dill.dump ( textured , open('blues99.feats', 'w')    )

