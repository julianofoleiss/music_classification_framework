from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.base import clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np
import itertools
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def stratified_group_shuffle_split(Y, groups, test_size=0.2):
    all_classes = sorted(list(set(Y)))
    
    assert len(Y) == len(groups)
    
    #get the number of samples of each class
    fn = [len(np.where(Y==l)[0]) for l in all_classes]
    
    #compute the average number of samples
    avg_fn = np.floor(np.mean(fn)).astype(int)
    
    #keep approximately this number of samples from each class
    sn = np.floor(avg_fn * test_size).astype(int)
    
    #This will store the indexes of the test samples
    ts_idx = []
    
    for i in all_classes:
        #get all the unique groups in this class
        g = np.unique(groups[np.where(Y==i)])
        #commpute the histograms for each group
        hg = np.unique(groups[np.where(Y==i)], return_counts=True)[1]
        #sort the histograms in reverse order. Keep only 16 groups so it won't take forever.
        #do argsort to allow sorting both g and hg
        hg_s = np.fliplr([np.argsort(hg)])[0][:16]
        g = g[hg_s]
        hg = hg[hg_s]
        
        #make sure that there are at least two groups per class.
        assert len(g) >= 2
        
        #computer the powerset of the histogram. This may take a while. For up to 16 groups this is ok.
        p = list(powerset(hg[1:]))
        
        #compute the sums of the powerset. This is a naive implementation of the
        #Sums NP-Complete problem. I'm not too worried about it because I know the use
        #cases are quite small in practice. Perhaps I should think of a 
        #dynamic programming solution for this, or use some heuristics.
        s = np.array(map(sum,p))
        
        #Find the sum that's closest to the number of samples I want from each class
        s = np.argmin(np.abs(s-sn))
        
        #Get the set that's equivalent to the closest sum
        c = p[s]
        grs = []
        
        #build the array of the groups equivalent
        #Remember that c is a view of the HISTOGRAM! We need to
        #map this back into the original groups!
        for k in c:
            j = np.where(hg==k)[0][0]
            grs.append(g[j])
            #this quick and dirty hack handles groups with repeated number of elements.
            hg=np.delete(hg,j)
            g=np.delete(g,j)
            
        #print i, c, sum(c), grs
        
        #Compute the posititions in the input groups that belong to the selected groups
        for k in grs:
            ts_idx.extend(np.where(groups==k)[0])
    
    #The train samples are the complement of the test samples with respect to all samples
    tr_idx = set(range(len(Y))) - set(ts_idx)
    
    #print len(ts_idx), np.unique(Y[ts_idx], return_counts=True)
    #print len(Y), len(tr_idx)
    
    return sorted(list(tr_idx)), sorted(ts_idx)

def _fit_and_score(estimator, params, X_train, y_train, X_val, y_val):
    estimator.set_params(**params)
    estimator.fit(X_train, y_train)
    return estimator.score(X_val, y_val)

class SingleSplitGridSearch(object):

    def __init__(self, estimator, param_grid , n_jobs=1, verbose=0, refit=True, groups_file=None):
        self.pg = ParameterGrid(param_grid)
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose= verbose
        self.refit = refit
        self.best_score_ = -1
        self.best_params_ = None
        self.best_estimator_ = None
        self.best_index_ = 0
        self.groups_file = groups_file

    def fit(self, X, y):
        grid = list(self.pg)

        #if a groups_file was supplied, use it to split the train set in
        #a train and a validation set such that the samples from the same group
        #are not simultaneously in both train and validation sets. It also tries to stratify the validation set.
        if self.groups_file is not None:
            print('SSGS: Groups File provided.')
            with open(self.groups_file) as f:
                c = f.readlines()
            l = np.array([ s.split('\t')[0].split('/')[4].split('_')[0] for s in c])
            g = np.array([ s.split('\t')[1].strip() for s in c]).astype(int)

            tr_idx, val_idx = stratified_group_shuffle_split(l,g,0.2)

            n_frames = (X.shape[0] / len(l))
            print (n_frames)

            tr_idx =  np.concatenate([ np.arange(n_frames) + (idx * n_frames) for idx in tr_idx ])
            val_idx = np.concatenate([ np.arange(n_frames) + (idx * n_frames) for idx in val_idx ])

            X_train = X[tr_idx]
            y_train = y[tr_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]

        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        print(X.shape, y.shape)
        print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

        scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose) \
            (delayed(_fit_and_score)(clone(self.estimator), params, X_train, y_train, X_val, y_val) for params in grid )

        self.best_index_ = np.argmax(scores)
        self.best_params_ = grid [ self.best_index_ ]
        self.best_score_ = scores[ self.best_index_ ]

        if self.refit:
            self.best_estimator_ = clone(self.estimator)
            self.best_estimator_.set_params(**self.best_params_)
            self.best_estimator_.fit(X, y)

    def predict(self, X):
        if not self.refit:
            raise NotFittedError('The Grid Search must be initialized with refit=True to make predictions.')
        
        check_is_fitted(self.best_estimator_, 'predict')

        return self.best_estimator_.predict(X)

