import numpy as np
import six
from core.data import filelists
from core.data.predicted_features import PredictedFeatures
class ClassifierPipeline(object):

    def __init__(self, params):
        self.params = params

    def fit(self, pf, train_list_file):

        if isinstance(pf, six.string_types):
            pf = dill.load(open(pf))

        assert isinstance(pf, PredictedFeatures), "pf must be in PredictedFeatures format!"

        feats = pf.estimator_output

        #pt_path, h5, arr = pytable.create_earray('none', fixdim_size=feats.shape[1], temp=True)

        train_feats, train_labels, track_idxs = get_filelist_features(train_list_file, pf)

        model = self._fit(train_feats, train_labels, track_idxs=track_idxs, pf=pf)

        #pytable.remove_earray(pt_path, h5)

        return model

    def _fit(self, train_features, train_labels, track_idxs=None, pf=None):
        raise NotImplementedError


def get_filelist_features(filelist_name, pf, feat_arr=[], label_arr=[], idxs_arr=[]):

    files, _ = filelists.parse_filelist(filelist_name)
    
    feat_arr = []
    label_arr = []
    idxs_arr = []
    
    #print ('vazio?!', idxs_arr)
    
    for f in files:
        #print pf.get_track_idxs(f)
        feat_arr.append(pf.estimator_output[pf.get_single_track_idxs(f),:])
        label_arr.append(pf.labels[pf.track_filenames_frame_idx_dict[f][1]])
        idxs_arr.append(pf.track_filenames_frame_idx_dict[f][1])

    if isinstance(feat_arr, list):
        feat_arr = np.concatenate(feat_arr)

    if isinstance(label_arr, list):
        label_arr = np.concatenate(label_arr).reshape((-1, len(label_arr[0])))

    if isinstance(label_arr, list):
        idxs_arr = np.array(idxs_arr)
        
    return feat_arr, label_arr, idxs_arr
