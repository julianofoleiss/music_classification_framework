import numpy as np
import dill
import librosa
from core.utils.encodings import to_label_matrix, tags_to_matrix, inv_dict
from core.data.predicted_features import PredictedFeatures

def get_identity_features(input_file, tags, params=None, output_file=None, **kwargs):

    predicted_features_input = params['identity_features']['predicted_features_input']
    ld_text = open(params['general']['label_dict_file']).read()
    exec(ld_text)

    if predicted_features_input is not None:
        pfi = dill.load(open(predicted_features_input))
    else:
        raise TypeError, 'identity_features.predicted_features_output must be set through the command line!'

    feats = pfi.estimator_output[pfi.get_single_track_idxs(input_file),:].T

    if params['identity_features']['deltas']:
        d = librosa.feature.delta(feats, order=1)
    else:
        d = np.empty((0,feats.shape[1]))

    if params['identity_features']['delta_deltas']:
        dd = librosa.feature.delta(feats, order=2)
    else:
        dd = np.empty((0,feats.shape[1]))

    feats = np.vstack((feats, d, dd))

    pf = PredictedFeatures([input_file], [feats.shape[1]], feats.T, inv_dict(label_dict))
    
    pf.labels = tags_to_matrix([tags], label_dict)
    pf.feature_filenames = [None]

    pfi.close_h5()

    if output_file is not None:
        dill.dump(pf, open(output_file, 'w'))
        pf = None

    return pf
