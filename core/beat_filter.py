# -*- coding: utf-8 -*-
from feature_filter import FeatureFilter
import librosa
import numpy as np

class BeatAggregatorFilter(FeatureFilter):

    def __init__(self, params=None):
        self.params = params

    def _filter(self, matrix, audio_filename=None, sr=44100, hop_length=1024, aggregation=np.mean):

        y, sr = librosa.load(audio_filename, sr=sr, mono=True)
        tempo, beats = librosa.beat.beat_track(y, sr=sr, hop_length=hop_length, trim=False)
        beats = librosa.util.fix_frames(beats, x_max=matrix.shape[0])

        synced = librosa.util.sync(matrix, beats, aggregate=aggregation, axis=0)

        return synced
        
        