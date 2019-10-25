import time

from classifier_pipeline import ClassifierPipeline
from ..utils.encodings import label_matrix_to_list, get_label_array
from ..single_split_gridsearch import SingleSplitGridSearch

from sklearn.feature_selection import f_classif as anova
from sklearn.feature_selection import SelectPercentile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np

class KNN(ClassifierPipeline):

    def __init__(self, params):
        super(KNN, self).__init__(params)

    def _fit(self, train_features, train_labels, track_idxs=None, pf=None):
        
        if track_idxs == None:
            raise ValueError, "track_idxs is needed to compute the train label list correctly!"

        if pf==None:
            raise ValueError, "pf is needed for this to work!"

        tracks_nframes = np.array(pf.track_nframes)[track_idxs]

        n_neighbors = self.params['knn']['n_neighbors']
        anova_percentiles = self.params['knn']['anova_percentiles']
        
        #label_dict = ....
        exec( open(self.params['general']['label_dict_file']).read() )
        train_labels = label_matrix_to_list(train_labels, label_dict)       
        train_labels = get_label_array(train_labels, tracks_nframes)

        ss = StandardScaler()
        an = SelectPercentile(anova)
        knnc = KNeighborsClassifier(n_neighbors=n_neighbors)

        parameter_combos = dict(knn__n_neighbors=n_neighbors)

        if self.params['knn']['no_anova']:
            clf = Pipeline( [('standardizer', ss ), ('knn', knnc) ] )
        else:
            clf = Pipeline( [('standardizer', ss ), ('anova', an), ('knn', knnc) ] )
            parameter_combos.update( dict(anova__percentile=anova_percentiles) )            

        estimator = SingleSplitGridSearch(clf, 
            parameter_combos, 
            self.params['knn']['num_workers'],
            self.params['knn']['grid_verbose'], 
            refit=True,
            groups_file=self.params['single_split_gs']['groups_file'])

        T0 = time.time()
        estimator.fit(train_features, train_labels)
        T1 = time.time()

        print "model training took %f seconds" % (T1-T0)
        print "best model score: %f" % (estimator.best_score_)
        best_nn = estimator.best_estimator_.named_steps['knn'].n_neighbors
        print "best params found for KNN: nn = %d" % (best_nn)

        if not self.params['knn']['no_anova']:
            best_percentile = estimator.best_estimator_.named_steps['anova'].percentile
            print "best params found for ANOVA: percetile = %d" % (best_percentile)

        return estimator.best_estimator_
