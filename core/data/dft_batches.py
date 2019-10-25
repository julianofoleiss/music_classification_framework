from core.data.data_batch_gen import TensorBatches
import numpy as np
import dill
import os
import librosa
import scipy

class DFTBatches(TensorBatches):
    
    def __init__(self, inputs, targets, spec_directory='stfts', target_fs=44100, to_mono=True, 
        zero_pad_resampling=True, in_db=True, dft_len=1024, window_step=512, window_len=1024, 
        standardizer=None, mel_bins=None):

        super(DFTBatches, self).__init__(inputs, targets)
        self.target_fs = target_fs
        self.to_mono = to_mono
        self.in_db = in_db
        self.dft_len = dft_len
        self.window_step = window_step
        self.window_len = window_len
        self.spec_directory = spec_directory
        self.track_nframes = []
        self.track_names = []
        self.standardizer = standardizer
        self.mel_bins = mel_bins

    def consume_buffer(self, buffer, n):
        r = buffer[:n]
        return buffer[n:], r

    def save_stft(self, trackname, stft_data):
        stft_filename = self.make_stft_filename(trackname)

        if not os.path.exists(self.spec_directory):
            os.makedirs(self.spec_directory)
        
        dill.dump(stft_data, open(self.spec_directory + "/" + stft_filename, 'w'), protocol=dill.HIGHEST_PROTOCOL)

    def make_stft_filename(self, trackname):
        filename = os.path.basename(trackname)
        filename, ext = os.path.splitext(filename)
        return filename + ".stft"

    def STFT(self, track):
        audio_data, rate = librosa.load(track, sr=self.target_fs, mono=self.to_mono)

        m = np.mean(audio_data)
        v = np.var(audio_data)
        audio_data = (audio_data - m) / np.sqrt(v)

        w = scipy.signal.hamming(self.dft_len, sym=False)

        spectrum = np.abs(librosa.core.stft(audio_data, n_fft=self.dft_len, hop_length=self.window_step, window=w, win_length=self.dft_len, center=True))

        if self.mel_bins is not None:
            spectrum = librosa.feature.melspectrogram(S=spectrum, sr=rate, n_fft=self.dft_len, hop_length=self.window_step, power=2.0, n_mels=self.mel_bins)

        if self.in_db:
            spectrum = librosa.power_to_db(spectrum)
        
        return spectrum.T
        
    def load_stft(self, stft_file):
        return dill.load(open(stft_file, 'r'))

    def standardize(self, X):
        if self.standardizer is not None:
            return self.standardizer.transform(X)
        else:
            return X

    def next(self, batch_size=None, shuffle=False):

        buffer_X = []
        buffer_Y = []
        self.track_nframes = []
        
        for X, Y in super(DFTBatches, self).next(1, shuffle):
            X = X[0]
            Y = Y[0] if Y is not None else None

            #If no spec_directory is given, STFTS are always computed on the fly.
            #Otherwise, spec_directory is queried for the file containing the stft.
            #If the stft file is found, then it is loaded. Otherwise, the STFT is calculated
            #and saved to spec_directory.
            stft = None
            if self.spec_directory != None:
                stft_filename = self.spec_directory + '/' + self.make_stft_filename(X)
                if os.path.exists(stft_filename):
                    stft = self.load_stft(stft_filename)
            if stft is None:
                stft = self.STFT(X)
                if self.spec_directory != None:
                    self.save_stft(X, stft)

            self.track_nframes.append(stft.shape[0])
            self.track_names.append(X)

            buffer_X.extend(self.standardize(stft).astype(np.float32))
            buffer_Y.extend(stft.shape[0] * [Y])

            while len(buffer_X) >= batch_size:
                buffer_X, X = self.consume_buffer(buffer_X, batch_size)
                buffer_Y, Y = self.consume_buffer(buffer_Y, batch_size)
                yield X, Y

        while len(buffer_X) >= batch_size:
            buffer_X, X = self.consume_buffer(buffer_X, batch_size)
            buffer_Y, Y = self.consume_buffer(buffer_Y, batch_size)
            yield X, Y    

        yield buffer_X, buffer_Y

class DFTAutoEncoderBatches(DFTBatches):

    def __init__(self, inputs, *args, **kwargs):
        super(DFTAutoEncoderBatches, self).__init__(inputs, None, *args, **kwargs)

    def next(self, batch_size=None, shuffle=False):
        
        for X, _ in super(DFTAutoEncoderBatches, self).next(batch_size, shuffle):
            yield X, X
