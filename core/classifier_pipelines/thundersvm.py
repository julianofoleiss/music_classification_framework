import time

from classifier_pipeline import ClassifierPipeline
from core.utils.encodings import label_matrix_to_list, get_label_array, inv_dict 
from core.single_split_gridsearch import SingleSplitGridSearch

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif as anova
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ShuffleSplit

from thundersvmScikit import SVC
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

import multiprocessing as mp
import numpy as np

def _fit_and_score(id, estimator, params, X_train, y_train, X_val, y_val, output):
    estimator.set_params(**params)
    estimator.fit(X_train, y_train)
    output.put((id, estimator.score(X_val, y_val)))

class SingleSplitGridSearchThunderSVM(object):

    def __init__(self, estimator, param_grid, n_gpus=[0], refit=True):
        self.pg = ParameterGrid(param_grid)
        self.estimator = estimator
        self.n_gpus = n_gpus
        self.refit = refit
        self.best_score_ = -1
        self.best_params_ = None
        self.best_estimator_ = None
        self.best_index_ = 0

    def fit(self, X, y):
        grid = list(self.pg)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        print(X.shape, y.shape)
        print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

        # scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose) \
        #     (delayed(_fit_and_score)(clone(self.estimator), params, X_train, y_train, X_val, y_val) for params in grid )

        scores = []
        for i in range(0, len(grid), len(self.n_gpus)):
            ps = []
            out = mp.Queue()
            for j in range(len(self.n_gpus)):
                if i+j >= len(grid):
                    continue
                grid[i+j]['svm__gpu_id'] = self.n_gpus[j]
                print ('appending thundersvm job ' + str(grid[i+j]) + ' ' + time.strftime("%c"))
                ps.append(mp.Process(target=_fit_and_score, args=(j, clone(self.estimator), grid[i+j], X_train, y_train, X_val, y_val, out)))

            for p in ps:
                p.start()
            
            for p in ps:
                p.join()
            
            res = [out.get() for p in ps]
            res.sort()
            res = [r[1] for r in res]
            scores.extend(res)

        self.best_index_ = np.argmax(scores)
        self.best_params_ = grid [ self.best_index_ ]
        self.best_score_ = scores[ self.best_index_ ]

        if self.refit:
            print(self.best_params_)
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

    def predict(self, X):
        if not self.refit:
            raise NotFittedError('The Grid Search must be initialized with refit=True to make predictions.')
        
        check_is_fitted(self.best_estimator_, 'predict')

        return self.best_estimator_.predict(X)

class ThunderSVM(ClassifierPipeline):

    def __init__(self, params):
        super(ThunderSVM, self).__init__(params)

    def _fit(self, train_features, train_labels, track_idxs=None, pf=None):
        
        if track_idxs == None:
            raise ValueError, "track_idxs is needed to compute the train label list correctly!"

        if pf==None:
            raise ValueError, "pf is needed for this to work!"

        tracks_nframes = np.array(pf.track_nframes)[track_idxs]

        gammas = self.params['thundersvm']['gammas']
        Cs = self.params['thundersvm']['Cs']
        calc_prob = self.params['thundersvm']['probability']
        kernel_type = self.params['thundersvm']['kernel']
        anova_percentiles = self.params['thundersvm']['anova_percentiles']

        #label_dict = ....
        exec( open(self.params['general']['label_dict_file']).read() )
        train_labels = label_matrix_to_list(train_labels, label_dict)       
        train_labels = get_label_array(train_labels, tracks_nframes)

        #internally, thundersvm does not encode class strings into integers automatically
        #Do it.
        d = label_dict #inv_dict(label_dict)
        #print d
        train_labels = np.array([d[i] for i in train_labels])

        svmc = SVC(kernel=kernel_type, probability=calc_prob, verbose=0)

        parameter_combos = dict(svm__C=Cs, 
            svm__gamma=gammas)

        an = SelectPercentile(anova)
        ss = StandardScaler()

        if self.params['thundersvm']['no_anova']:
            clf = Pipeline( [('standardizer', ss ), ('svm', svmc) ] )
        else:
            clf = Pipeline( [('standardizer', ss ), ('anova', an), ('svm', svmc) ] )
            parameter_combos.update( dict(anova__percentile=anova_percentiles) )

        if self.params['thundersvm']['hyperparameter_tuning'] == 'single_split':
            estimator = SingleSplitGridSearchThunderSVM(clf, 
                parameter_combos, 
                self.params['thundersvm']['num_gpus'],
                refit=True)
        else:
            raise ValueError('Invalid hyperparameter_tuning parameter: %s.' %  self.params['thundersvm']['hyperparameter_tuning'] )

        T0 = time.time()
        estimator.fit(train_features, train_labels)
        T1 = time.time()

        print "model training took %f seconds" % (T1-T0)
        print "best model score: %f" % (estimator.best_score_)
        best_c = estimator.best_estimator_.named_steps['svm'].C
        best_gamma = estimator.best_estimator_.named_steps['svm'].gamma
        print "best params found for SVM: C = %.2ef, gamma = %s" % (best_c, best_gamma)

        if not self.params['thundersvm']['no_anova']:
            best_percentile = estimator.best_estimator_.named_steps['anova'].percentile
            print "best params found for ANOVA: percetile = %d" % (best_percentile)

        return estimator.best_estimator_
