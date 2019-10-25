# -*- coding: utf-8 -*-

import librosa
import numpy as np
import dill, sys
import importlib
import scipy
from .data.predicted_features import PredictedFeatures
from .utils.encodings import to_label_matrix, tags_to_matrix, inv_dict
from time import time

from scipy.fftpack.realtransforms import dct

def zero_crossings(wav_data, frame_length, window_size):
    """Calculates the Time Domain Zero Crossings feature as in (Tzanetakis, 2002)"""

    def sign(n):
        return 1 if n>=0 else 0

    f = np.sign #numpy.vectorize(sign, otypes=[int])

    wav_data = f(wav_data)
    wav_data[wav_data == 0] = 1
    wav_data[wav_data == -1] = 0

    size = 1 + int((len(wav_data) - frame_length) / float(window_size))

    #size = int(((len(wav_data) -\
    #    (frame_length-window_size))/float(window_size))) - 1
    ret = np.zeros(size + 2)

    for k in xrange (size):
        this_start = k* window_size
        this_end = min(this_start + frame_length, len(wav_data))
        zc = np.array((np.sum(np.abs(np.diff(wav_data[this_start:this_end])), dtype=float)))
        zc/=2
        ret[k] = zc

    return ret

def hz2mel(f):
    """Convert an array of frequency in Hz into mel."""
    return 1127.01048 * np.log(f/700 +1)

def mel2hz(m):
    """Convert an array of frequency in Hz into mel."""
    return (np.exp(m / 1127.01048) - 1) * 700


def trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfilt, nlogfilt):
    """Compute triangular filterbank for MFCC computation."""
    # Total number of filters
    nfilt = nlinfilt + nlogfilt

    #------------------------
    # Compute the filter bank
    #------------------------
    # Compute start/middle/end points of the triangular filters in spectral
    # domain
    freqs = np.zeros(nfilt+2)
    freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc
    freqs[nlinfilt:] = freqs[nlinfilt-1] * logsc ** np.arange(1, nlogfilt + 3)
    heights = 2./(freqs[2:] - freqs[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((nfilt, nfft))
    # FFT bins (in Hz)
    nfreqs = np.arange(nfft) / (1. * nfft) * fs
    for i in range(nfilt):
        low = freqs[i]
        cen = freqs[i+1]
        hi = freqs[i+2]

        lid = np.arange(np.floor(low * nfft / fs) + 1,
                        np.floor(cen * nfft / fs) + 1, dtype=np.int)
        lslope = heights[i] / (cen - low)
        rid = np.arange(np.floor(cen * nfft / fs) + 1,
                        np.floor(hi * nfft / fs) + 1, dtype=np.int)
        rslope = heights[i] / (hi - cen)
        fbank[i][lid] = lslope * (nfreqs[lid] - low)
        fbank[i][rid] = rslope * (hi - nfreqs[rid])

    return fbank, freqs

def mfcc(S, nceps=13, n_fft=2048, sr=44100, hop_length=1024):
    """Compute Mel Frequency Cepstral Coefficients.
    Parameters
    ----------
    input: ndarray
        input spectrogram from which the coefficients are computed
    Returns
    -------
    ceps: ndarray
        Mel-cepstrum coefficients
    mspec: ndarray
        Log-spectrum in the mel-domain.
    Notes
    -----
    MFCC are computed as follows:
        * Pre-processing in time-domain (pre-emphasizing)
        * Compute the spectrum amplitude by windowing with a Hamming window
        * Filter the signal in the spectral domain with a triangular
        filter-bank, whose filters are approximatively linearly spaced on the
        mel scale, and have equal bandwith in the mel scale
        * Compute the DCT of the log-spectrum
    This is based on the talkbox module:
    http://pydoc.net/Python/scikits.talkbox/0.2.4.dev/scikits.talkbox.features.mfcc/
    References
    ----------
    .. [1] S.B. Davis and P. Mermelstein, "Comparison of parametric
           representations for monosyllabic word recognition in continuously
           spoken sentences", IEEE Trans. Acoustics. Speech, Signal Proc.
           ASSP-28 (4): 357-366, August 1980."""


    nfft = n_fft
    fs = sr
    over = hop_length

    #lowfreq = 400 / 3.
    lowfreq = 133.33
    #highfreq = 6855.4976
    linsc = 200/3.
    logsc = 1.0711703

    nlinfil = 13
    nlogfil = 27
    nlinfil + nlogfil

    fbank = trfbank(fs, nfft, lowfreq, linsc, logsc, nlinfil, nlogfil)[0]
    fbank = fbank.T[0:S.shape[0], :]

    mspec = np.log10(np.maximum(np.dot(fbank.T, S), 0.0000001)).T

    # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]

    return ceps

def flatness(A):
    """Spectral flatness of each frame"""
    return np.exp(  np.mean(np.log(np.maximum(A, 0.0001)), 0) ) / \
       (np.mean(A, 0) + (10**(-6)))

def flux(A):
    """Spectral flux of each frame"""
    a = np.diff(A, axis = 1)
    s = np.sum(np.maximum(a, 0), axis=0)
    s0 = np.sum(A, axis=0) + (10**(-6))
    return np.hstack ((np.array([0]), s))/np.maximum(s0, 0.0000001)

def energy(A):
    """Energy of each frame"""
    return np.sum(A**2 , 0)

def centroid(A):
    """Centroid of each frame"""
    lin = A.shape[0]
    col = A.shape[1]

    return np.sum( A * \
            np.transpose(np.tile(np.arange(lin), (col,1))), 0)\
            / np.maximum(0.00001, np.sum(A, 0))

def rolloff(A, alpha=0.95):
    """Rolloff of each frame"""
    return np.sum ( (np.cumsum(A, 0)/\
                        np.maximum(0.0000001, np.sum(A, 0))) < alpha, 0)

def get_gtzan_features_pymir(input_file, tags, params=None, output_file=None):
    # type (str, list[str], str) -> PredictedFeatures
    """
    Computes gtzan features for an input audio file.

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

    if params is not None:
        n_mfcc = params['gtzan_features']['n_mfcc'] 
        n_fft = params['gtzan_features']['n_fft'] 
        hop_length = params['gtzan_features']['hop_length'] 
        target_sr = params['gtzan_features']['target_sr']
        ld_text = open(params['general']['label_dict_file']).read()
        exec(ld_text)
        
    else:
        n_mfcc = 20
        n_fft = 2048
        hop_length = 1024
        target_sr = 44100

    #print("Calculating Features for %s..." % input_file)
    st =time()

    import mir3.modules.tool.wav2spectrogram as spec
    converter = spec.Wav2Spectrogram()
    s = converter.convert(input_file, window_length=2048, dft_length=2048,
                window_step=1024, spectrum_type='magnitude', save_metadata=True, wav_rate=44100)

    rate, y = converter.load_audio(input_file)
    stft = s.data

    mfccs = mfcc(stft, nceps=n_mfcc, n_fft=n_fft, sr=target_sr, hop_length=hop_length).T
    scentroid = centroid(stft[0:1024])
    srolloff = rolloff(stft[0:1024])
    flat = flatness(stft[0:1024])
    flx = flux(stft[0:1024])
    en = energy(stft[0:1024])

    zcr = zero_crossings(y, n_fft, hop_length) #librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

    feats = np.vstack((mfccs, scentroid, srolloff, flat, flx , en, zcr))

    feats = np.array(feats).T

    if params['gtzan_features']['deltas']:
        pad = np.zeros((1, feats.shape[1]))
        d = np.vstack((pad, np.diff(feats, axis=0)))
        if params['gtzan_features']['delta_deltas']:
            pad = np.zeros((1, d.shape[1]))
            dd = np.vstack((pad, np.diff(d, axis=0)))
        else:
            dd = np.empty((0,feats.shape[1]))
    else:
        d = np.empty((0,feats.shape[1]))
        dd = np.empty((0,feats.shape[1]))
        
    feats = np.hstack((feats, d, dd))

    pf = PredictedFeatures([input_file], [feats.shape[0]], feats, inv_dict(label_dict))
    pf.labels = tags_to_matrix([tags], label_dict)
    pf.feature_filenames = [None]

    if output_file is not None:
        dill.dump(pf, open(output_file, 'w'))
        pf = None

    return pf

def get_gtzan_features(input_file, tags, params=None, output_file=None):
    # type (str, list[str], str) -> PredictedFeatures
    """
    Computes gtzan features for an input audio file.

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

    if params is not None:
        n_mfcc = params['gtzan_features']['n_mfcc'] 
        n_fft = params['gtzan_features']['n_fft'] 
        hop_length = params['gtzan_features']['hop_length'] 
        target_sr = params['gtzan_features']['target_sr']
        ld_text = open(params['general']['label_dict_file']).read()
        exec(ld_text)
        
    else:
        n_mfcc = 20
        n_fft = 2048
        hop_length = 1024
        target_sr = 44100
        from dft_net.gtzan_data import label_dict

    #print("Calculating Features for %s..." % input_file)
    st =time()

    y, sr = librosa.load(input_file, sr=target_sr, mono=True)

    #centering data around mean zero, std_dev 1
    m = np.mean(y)
    v = np.var(y)
    y = (y-m) / np.sqrt(v)

    w = scipy.signal.hamming(n_fft, sym=False)

    stft = np.abs(librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length, window=w, win_length=n_fft, center=True))

    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    #rmse = librosa.feature.rmse(S=stft, frame_length=n_fft, hop_length=hop_length )
    
    en = energy(stft)
    scentroid = librosa.feature.spectral_centroid(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_length, freq=None)
    srolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, n_fft=n_fft, hop_length=hop_length, freq=None )
    flat = flatness(stft)
    flx = flux(stft)

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)

    feats = np.vstack((mfccs, scentroid, srolloff, flat, flx , en, zcr))

    if params['gtzan_features']['deltas']:
        d = librosa.feature.delta(feats, order=1)
    else:
        d = np.empty((0,feats.shape[1]))

    if params['gtzan_features']['delta_deltas']:
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

