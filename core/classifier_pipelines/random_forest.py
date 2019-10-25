import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from core.single_split_gridsearch import SingleSplitGridSearch
from core.classifier_pipelines.classifier_pipeline import ClassifierPipeline
from core.utils.encodings import label_matrix_to_list, get_label_array

class RandomForest(ClassifierPipeline):

    def __init__(self, params):
        super(RandomForest, self).__init__(params)

    def _fit(self, train_features, train_labels, track_idxs=None, pf=None):
        

        if track_idxs == None:
            raise ValueError, "track_idxs is needed to compute the train label list correctly!"

        if pf==None:
            raise ValueError, "pf is needed for this to work!"

        tracks_nframes = np.array(pf.track_nframes)[track_idxs]

        min_samples_leaf = self.params['random_forest']['min_samples_leaf']
        n_estimators = self.params['random_forest']['n_estimators']
        max_features = self.params['random_forest']['max_features_ratio']

        #label_dict = ....
        exec( open(self.params['general']['label_dict_file']).read() )
        train_labels = label_matrix_to_list(train_labels, label_dict)       
        train_labels = get_label_array(train_labels, tracks_nframes)

        ss = StandardScaler()
        rf = RandomForestClassifier(max_depth=None, oob_score=True)

        clf = Pipeline([('ss', ss), ('rf', rf)])

        parameter_combos = { 'rf__min_samples_leaf' : min_samples_leaf, 'rf__n_estimators' : n_estimators, 'rf__max_features' : max_features }

        estimator = SingleSplitGridSearch(clf, 
            parameter_combos, 
            self.params['random_forest']['num_workers'],
            self.params['random_forest']['grid_verbose'], 
            refit=True)

        T0 = time.time()
        estimator.fit(train_features, train_labels)
        T1 = time.time()

        print "model training took %f seconds" % (T1-T0)
        print "best model score: %f" % (estimator.best_score_)
        best_min_samples_leaf = estimator.best_estimator_.named_steps['rf'].min_samples_leaf
        best_n_estimators = estimator.best_estimator_.named_steps['rf'].n_estimators
        best_max_features = estimator.best_estimator_.named_steps['rf'].max_features
        print "best params found for RF: min_samples_leaf = %d, n_estimators = %d, max_features = %.2f" \
            % (best_min_samples_leaf, best_n_estimators, best_max_features)

        return estimator.best_estimator_
