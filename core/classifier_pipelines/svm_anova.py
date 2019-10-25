import time

from classifier_pipeline import ClassifierPipeline
from core.utils.encodings import label_matrix_to_list, get_label_array
from core.single_split_gridsearch import SingleSplitGridSearch

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif as anova
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ShuffleSplit

import numpy as np

class SvmAnova(ClassifierPipeline):

    def __init__(self, params):
        super(SvmAnova, self).__init__(params)

    def _fit(self, train_features, train_labels, track_idxs=None, pf=None):
        
        if track_idxs == None:
            raise ValueError, "track_idxs is needed to compute the train label list correctly!"

        if pf==None:
            raise ValueError, "pf is needed for this to work!"

        tracks_nframes = np.array(pf.track_nframes)[track_idxs]

        gammas = self.params['svm_anova']['gammas']
        Cs = self.params['svm_anova']['Cs']
        calc_prob = self.params['svm_anova']['probability']
        anova_percentiles = self.params['svm_anova']['anova_percentiles']
        kernel_type = self.params['svm_anova']['kernel']

        #label_dict = ....
        exec( open(self.params['general']['label_dict_file']).read() )
        train_labels = label_matrix_to_list(train_labels, label_dict)       
        train_labels = get_label_array(train_labels, tracks_nframes)

        ss = StandardScaler()
        an = SelectPercentile(anova)
        svmc = SVC(kernel=kernel_type, probability=calc_prob)

        parameter_combos = dict(svm__C=Cs, 
            svm__gamma=gammas)

        if self.params['svm_anova']['no_anova']:
            clf = Pipeline( [('standardizer', ss ), ('svm', svmc) ] )
        else:
            clf = Pipeline( [('standardizer', ss ), ('anova', an), ('svm', svmc) ] )
            parameter_combos.update( dict(anova__percentile=anova_percentiles) )

        if self.params['svm_anova']['hyperparameter_tuning'] == 'single_split':
            estimator = SingleSplitGridSearch(clf, 
                parameter_combos, 
                self.params['svm_anova']['num_workers'],
                self.params['svm_anova']['grid_verbose'], 
                refit=True,
                groups_file=self.params['single_split_gs']['groups_file'])

        elif self.params['svm_anova']['hyperparameter_tuning'] == 'cv':
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

            estimator = GridSearchCV(clf,
                parameter_combos,
                cv=cv,
                verbose=self.params['svm_anova']['grid_verbose'],
                n_jobs=self.params['svm_anova']['num_workers']
            )

        else:
            raise ValueError('Invalid hyperparameter_tuning parameter: %s.' %  self.params['svm_anova']['hyperparameter_tuning'] )

        T0 = time.time()
        estimator.fit(train_features, train_labels)
        T1 = time.time()

        print "model training took %f seconds" % (T1-T0)
        print "best model score: %f" % (estimator.best_score_)
        best_c = estimator.best_estimator_.named_steps['svm'].C
        best_gamma = estimator.best_estimator_.named_steps['svm'].gamma
        print "best params found for SVM: C = %.2ef, gamma = %s" % (best_c, best_gamma)
        
        if not self.params['svm_anova']['no_anova']:
            best_percentile = estimator.best_estimator_.named_steps['anova'].percentile
            print "best params found for ANOVA: percetile = %d" % (best_percentile)

        return estimator.best_estimator_
