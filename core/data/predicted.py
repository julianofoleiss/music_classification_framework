# -*- coding: utf-8 -*-

import numpy as np
from pytable import create_earray
import codecs
import tables

class Predicted(object):
    
    def __init__(self, track_filenames, track_nframes, estimator_output, label_idx_dict):
        assert len(track_filenames) == len(track_nframes), 'You should provide the number of frames for each track!'
        assert np.sum(track_nframes) == estimator_output.shape[0], 'There must a prediction for every frame of every track!'

        if hasattr(estimator_output, 'attrs'):
            self.h5filename = estimator_output.attrs.h5filename
        else:
            self.h5filename = None

        self.track_filenames = track_filenames
        self.track_nframes = track_nframes
        self.estimator_output = estimator_output
        self.label_idx_dict = label_idx_dict

        #maps filenames to (start_frame, track_idx). track_idx indexes track_{filenames|n_frames}
        self.track_filenames_frame_idx_dict = dict() 
        i = 0
        for t in xrange(len(self.track_filenames)):
            self.track_filenames_frame_idx_dict[self.track_filenames[t]] = (i, t)
            i += self.track_nframes[t]

    def refresh_track_nframes(self, track_nframes):
        self.track_nframes = track_nframes
        self.track_filenames_frame_idx_dict = dict() 
        i = 0
        for t in xrange(len(self.track_filenames)):
            self.track_filenames_frame_idx_dict[self.track_filenames[t]] = (i, t)
            i += self.track_nframes[t]        

    def convert_storage_to_h5(self, h5filename):        
        filename, h5, arr = create_earray(h5filename, fixdim_size=self.estimator_output.shape[1])
        self.h5filename = filename
        arr.append(self.estimator_output)
        h5.flush()
        h5.close()
        #self.h5file = tables.open_file(filename, mode='a')
        self.reload_h5_storage()
        #self.set_estimator_output(arr)
        #self.estimator_output = self.h5file.root.features
        return filename
            
    def has_h5_storage(self):
        return True if self.h5filename is not None else False
        
    def reload_h5_storage(self):
        self.h5file = tables.open_file(self.h5filename, mode='a')
        self.estimator_output = self.h5file.root.features
        
    def set_estimator_output(self, estimator_output):
        if hasattr(estimator_output, 'attrs'):
            self.h5filename = estimator_output.attrs.h5filename
        else:
            self.h5filename = None
            
        self.estimator_output = estimator_output

    #this is called upon pickling this object
    def __getstate__(self):
        if self.h5filename is not None:
            old_estimator_output = self.estimator_output
            self.__dict__['estimator_output'] = None
            self.__dict__['h5file'] = None
        return self.__dict__

    #this is called upon unpickling this object
    def __setstate__(self, state):
        if state['h5filename'] is not None:
            state['h5file'] = tables.open_file(state['h5filename'], mode='a')
            state['estimator_output'] = state['h5file'].root.features
        for k in state:
            setattr(self, k, state[k])

    def max_prediction(self):
        #this function assumes that estimator_output is a matrix containing a row
        #for every audio frame and the columns are the probability of each frame
        #to belong to every class (such as the output of a softmax layer in a NN)
    
        return self.estimator_output.argmax(axis=1)

    def get_track_idxs(self):
        i = 0
        for t in xrange(len(self.track_filenames)):
            yield np.arange(self.track_nframes[t]) + i
            i += self.track_nframes[t]

    def get_single_track_idxs(self, filename):
        return np.arange(self.track_nframes[self.track_filenames_frame_idx_dict[filename][1]]) + self.track_filenames_frame_idx_dict[filename][0]

    def output_max_prediction(self, filename):
    
        predicted_classes = self.max_prediction()

        predicted_output = codecs.open(filename, 'w', encoding='utf-8')
        i = 0
        for t in xrange(len(self.track_filenames)):
            for f in xrange(self.track_nframes[t]):
                predicted_output.write("%s\t%s\n" % (self.track_filenames[t], 
                    self.label_idx_dict[predicted_classes[i]] if self.label_idx_dict is not None else predicted_classes[i]))
                i+=1

        predicted_output.close()        

    def close_h5(self):
        self.h5file.close()

