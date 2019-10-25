import six
import dill
import numpy as np

from core.data.predicted import Predicted
from core.data.predicted_features import PredictedFeatures
from core.classifier_pipelines.classifier_pipeline import get_filelist_features

class ModelTester(object):

    def __init__(self, params):
        self.params = params
        self.model = None

    def predict(self, model, pf, test_list_file, voting_model=False, **kwargs):
        
        if isinstance(model, six.string_types):
            #This is a shameless hack to make it work with thundersvm while I don't make it pickable.
            if model.endswith('thundersvm'):
                from thundersvmScikit import SVC
                from sklearn.pipeline import Pipeline
                import os
                svm = SVC()
                svm.load_from_file(model)
                ss = dill.load(open(model + '.ss'))
                anova_file = model + '.anova'
                if os.path.isfile(anova_file):
                    an = dill.load(open(anova_file))
                    print('detected anova file. building pipeline with anova.')
                    model = Pipeline( [('standardizer', ss ), ('anova', an ), ('svm', svm) ] )
                else:
                    model = Pipeline( [('standardizer', ss ), ('svm', svm) ] )
            else:
                model = dill.load(open(model))

        self.model = model

        if isinstance(pf, six.string_types):
            pf = dill.load(open(pf))

        assert isinstance(pf, PredictedFeatures), "pf must be in PredictedFeatures format!"

#        print 'init of model_tester.py'
        test_feats, test_labels, track_idxs = get_filelist_features(test_list_file, pf)

#         pprint (test_list_file)
#         print (track_idxs)
#         pprint (pf.estimator_output.shape)
#         pprint (pf.track_filenames)
#         print(len(pf.track_filenames))
        
        predictions = self._predict(model, test_feats, track_idxs=track_idxs, pf=pf, **kwargs)
        
#        print(predictions.shape)
        
        tracks_filenames = np.array(pf.track_filenames)[track_idxs]
        tracks_nframes = np.array(pf.track_nframes)[track_idxs]
        
#        print tracks_nframes, sum(tracks_nframes)
        
        if voting_model:
            tracks_nframes = [1] * tracks_nframes.shape[0]
            #print(tracks_nframes)

        predictions = Predicted(tracks_filenames, tracks_nframes, predictions, pf.label_idx_dict)

        return predictions

    def _predict(self, model, test_features, track_idxs=None, pf=None, **kwargs):
        raise NotImplementedError
