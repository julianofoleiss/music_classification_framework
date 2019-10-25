
# coding: utf-8
import librosa
import numpy as np
import sklearn
import time
import os
import click
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pprint import pprint
import dill
import matplotlib.pyplot as plt
from core.utils.dataset_utils import parse_filelist
from core.gtzan_features import get_gtzan_features
from core.stft_features import get_stft_features
from core.parallel_extraction import from_filelist
from core.texture_avg_stdev_filter import TextureAvgStdevFilter
from core.data.predicted_features import PredictedFeatures
from core import select_frames
from core.single_split_gridsearch import SingleSplitGridSearch
from core.classifier_pipelines.svm_anova import SvmAnova
from core.test_pipelines.sklearnlike_tester import SklearnLike
from core.prediction_voting.max_voting import MaxVoting
from core.data.filelists import parse_filelist
from core.classifier_pipelines.knn import KNN
from eval_classification import eval_classification

def linspace_textures(textures, texture_hop):
    n_texture_windows = textures.shape[0] / texture_hop
    tidx = np.floor(np.linspace(0, textures.shape[0], n_texture_windows, endpoint=False)).astype(int)
    return textures[tidx]

#features extraction

def _parallel_fe(args):
    params, input_files, tags = args
    out = []
    extraction_fcn = params['_parallel_fe']['extraction_fcn']
    for i in range(len(input_files)):
        feats = extraction_fcn(input_files[i], tags[i], params=params, output_file=None)
        f = TextureAvgStdevFilter(params)
        feats = f.filter(feats)
        out.append(feats)
    return out

def extract_features(filelist, fe_params, output_filename, feature_set='marsyas'):
    if feature_set == 'marsyas':
        fe_params.update({'_parallel_fe': {'extraction_fcn' : get_gtzan_features}})
    elif feature_set == 'mel':
        fe_params.update({'_parallel_fe': {'extraction_fcn' : get_stft_features}})
    else:
        raise ValueError('feature set not supported.')

    pf = from_filelist(filelist, 100, 5, 8, _parallel_fe, 
                    fe_params, output_filename, out_format='h5', inscript=True)
                   
# Get "All" textures (from all files in the dataset, choose textures 0.23s apart)
def get_textures(texture_hop, features_filename, output_file):
    features = dill.load(open(features_filename))
    out = []
    out_len = []
    for f in features.track_filenames:
        out.append(linspace_textures(features.estimator_output[features.get_single_track_idxs(f),:], texture_hop))
        out_len.append(out[-1].shape[0])

    np.unique(out_len, return_counts=True)

    all_textures = PredictedFeatures(features.track_filenames, out_len, np.concatenate(out), features.label_idx_dict, 
                        [None] * len(features.track_filenames) , features.labels )
    out = None

    all_textures.convert_storage_to_h5(output_file + '.h5')
    all_textures.h5file.close()

    dill.dump(all_textures, open(output_file, 'w'))

# Selecting textures using KMEANS and LINSPACE

def _select_textures(args):
    params, input_files, tags = args 
    
    ss = params['_select_frames']['ss']
    pf = params['_select_frames']['pf']
    selector = params['_select_frames']['selector']
    n_frames = params['_select_frames']['n_frames']
    select_subset = params['_select_frames']['select_subset']
    
    out = []    
    
    #Here's the workaround to the h5 pickling issue. Just reload the storage =)
    if pf.has_h5_storage():
        pf.reload_h5_storage()
    
    for f in input_files:
        features = ss.transform(pf.estimator_output[pf.get_single_track_idxs(f),:])
        fs = select_frames.FrameSelector.get_selector( selector, params )

        selected = fs._select_frames(features, n_frames)

        if select_subset:
            selected = selected[:select_subset,:]
        
        selected = PredictedFeatures([f], [selected.shape[0]], selected, pf.label_idx_dict, [None], pf.get_track_label(f))
        out.append(selected)

    if pf.has_h5_storage():
        pf.h5file.close()

    return out

def select_textures(pf_filename, fold_filename, output_filename, ss=None, **selector_arguments):
    
    default_selector_args = {
        '_select_frames' : {
            'ss' : None,            #Don't change this
            'pf' : None,            #Don't change this
            'selector' : 'kmeansc',
            'n_frames' : 5,
            'select_subset': False
        },
        'kmeansc_selector' : {
            'sort_centroids_by_common' : True
        }
    }    
    
    #print ('default args: ', default_selector_args)
    #print ('passed args: ', selector_arguments)
    
    for d in default_selector_args:
        if d in selector_arguments:
            print('updating ' + d)
            default_selector_args[d].update(selector_arguments[d])
            
    for d in selector_arguments:
        if d not in default_selector_args:
            default_selector_args[d] = selector_arguments[d]
            
    #print ('updated args: ', default_selector_args)
    
    pf = dill.load(open(pf_filename))
    
    if ss is None:
        ss = StandardScaler()
        ss.fit(pf.estimator_output)
        
    default_selector_args['_select_frames']['ss'] = ss
    
    #h5 arrays cannot be pickled. Thus, before passing an object with a h5
    #to a (process-based) parallel job, it must be explicitly closed.
    #see function _select_textures (the parallel job) to check how to
    #get around this.
    if pf.has_h5_storage():
        pf.h5file.close()
        
    default_selector_args['_select_frames']['pf'] = pf
    
    textures = from_filelist(fold_filename, 100, 10, 8, _select_textures, 
                       default_selector_args, output_filename, out_format='h5', inscript=True)
    
    return textures

def set_scratch_lock(scratch_dir_path):
    lock_path = scratch_dir_path + 'lock'
    if os.path.isfile(lock_path):
        with open(lock_path) as f:
            pid = int(f.read())
        raise Exception('Scratch directory %s is already being used by process %d!' % (lock_path, pid))

    with open(lock_path, 'w') as f:
        f.write('%d'% os.getpid())

def release_scratch_lock(scratch_dir_path):
    lock_path = scratch_dir_path + 'lock'
    try:
        os.unlink(lock_path)
    except OSError as err:
        if err.args[0] == 2:
            print('WARNING: lock was not in the scratch directory.')


if __name__ == "__main__":

    # Classification pipeline (frame selection)

    ######SETTINGS FOR GTZAN 3F ARTIST FILTER############
    # folds_folder = 'gtzan_folds_filter_3f_strat/'
    # scratch_folder = 'scratch3/'
    # labels_file = folds_folder + 'gtzan_labels.txt'
    # dataset = 'gtzan'
    # texture_length = 216
    # texture_hop = 10
    # dict_file = folds_folder + 'gtzan_dict.py'
    # n_folds = 3

    ######SETTINGS FOR GTZAN 10F RANDOM############
    # scratch_folder = 'scratch6/'
    # folds_folder = 'gtzan_folds_10fs/'
    # labels_file = folds_folder + 'gtzan_labels.txt'
    # dataset = 'gtzan'
    # texture_length = 216
    # texture_hop = 10
    # dict_file = folds_folder + 'gtzan_dict.py'
    # n_folds = 10

    ######SETTINGS FOR LMD############
    # scratch_folder = 'scratch_lmd_2/'
    # folds_folder = 'lmd_folds/'
    # labels_file = folds_folder + 'lmd_labels.txt'
    # dataset = 'lmd'
    # texture_length = 216
    # texture_hop = 10
    # dict_file = folds_folder + 'lmd_dict.py'
    # n_folds = 3

    ######SETTINGS FOR HOMBURG############
    # scratch_folder = 'scratch_homburg_2/'
    # folds_folder = 'homburg_folds/'
    # labels_file = folds_folder + 'homburg_labels.txt'
    # dataset = 'homburg'
    # texture_length = 216
    # texture_hop = 10
    # dict_file = folds_folder + 'homburg_dict.py'
    # n_folds = 10

    ######SETTINGS FOR ISMIR############
    # scratch_folder = 'scratch_ismir_2/'
    # folds_folder = 'ismir_folds/'
    # labels_file = folds_folder + 'ismir_labels.txt'
    # dataset = 'ismir'
    # texture_length = 216
    # texture_hop = 10
    # dict_file = folds_folder + 'ismir_dict.py'
    # n_folds = 1

    ######SETTINGS FOR EXBALLROOM############
    scratch_folder = 'scratch_exballroom_1/'
    #scratch_folder = 'scratch11/'
    folds_folder = 'exballroom_folds/'
    labels_file = folds_folder + 'exballroom_labels.txt'
    dataset = 'exballroom'
    texture_length = 216
    texture_hop = 10
    dict_file = folds_folder + 'exballroom_dict.py'
    n_folds = 10

    print("Dataset Settings:")
    print("dataset: %s" % dataset)
    print('scratch_folder: %s' % scratch_folder)
    print('folds_folder: %s' % folds_folder)
    print('labels_file: %s' % labels_file)
    print('texture_length: %d' % texture_length)
    print('texture_hop: %d' % texture_hop)
    print('n_folds: %d' % n_folds)

    ################EXPERIMENT SETTINGS##########################
    
    texture_selectors = ['kmeansc', 'linspace']
    texture_numbers = [5, 20, 40]
    feature_extractor = 'mel'

    # texture_selectors = ['linspace']
    # texture_numbers = [5, 20]

    print('Experiment settings:')
    print('texture_selectors:'),
    print(texture_selectors)
    print('texture_numbers:'),
    print(texture_numbers)
    print('feature_extractor:'),
    print(feature_extractor)

    features_filename = scratch_folder + 'features.dill'
    all_textures_filename = scratch_folder + 'all_features.dill'
    train_textures_filename = scratch_folder + 'train_textures.dill'
    test_textures_filename = scratch_folder + 'test_textures.dill'

    task_extract = True

    marsyas_fe_params = {
        'gtzan_features' : {
            'n_mfcc' : 20,
            'n_fft' : 2048,
            'hop_length' : 1024,
            'target_sr' : 44100,
            'deltas' : True,
            'delta_deltas' : True
        },
        'texture_avg_stdev_filter' : {
            'window_size' : texture_length,
            'mean' : True,
            'variance': True
        },
        'general' : {
            'label_dict_file' : dict_file
        }
    }

    mel_fe_params = {
        'stft_features' : {
            'n_mfcc' : 20,
            'n_fft' : 2048,
            'hop_length' : 1024,
            'target_sr' : 44100,
            'deltas' : True,
            'delta_deltas' : True,
            'mel_bins' : 128            
        },
        'texture_avg_stdev_filter' : {
            'window_size' : texture_length,
            'mean' : True,
            'variance': True
        },
        'general' : {
            'label_dict_file' : dict_file
        }
    }

    svm_params={
        'svm_anova': {
            'gammas': ['auto', 0.1, 0.001, 0.0001],
            'Cs': [10, 100, 1000],
            'probability': False,
            'anova_percentiles': [],
            'kernel': 'rbf',
            'hyperparameter_tuning': 'single_split',
            'num_workers': 8,
            'grid_verbose': 10,
            'no_anova' : True
        },
        'general': {
            'label_dict_file': dict_file
        },
        'single_split_gs':{
            'groups_file': None
        }
    }

    knn_anova_params={
        'knn' : {
            'n_neighbors' : [1, 3, 5, 7, 9],
            'anova_percentiles' : [20, 40, 60, 80],
            'no_anova': False,
            'num_workers': 8,
            'grid_verbose': 10,

        },
        'general': {
            'label_dict_file': dict_file
        },
        'single_split_gs':{
            'groups_file': None
        }    
    }

    knn_params={
        'knn' : {
            'n_neighbors' : [1, 3, 5, 7, 9],
            'anova_percentiles' : [],
            'no_anova': True,
            'num_workers': 8,
            'grid_verbose': 10,

        },
        'general': {
            'label_dict_file': dict_file
        },
        'single_split_gs':{
            'groups_file': None
        }    
    }

    tester_params = {
        'general' : {
            'label_dict_file': dict_file
        }
    }

    set_scratch_lock(scratch_folder)

    if task_extract:
        fe_params = eval(feature_extractor + '_fe_params')
        extract_features(labels_file, fe_params, features_filename, feature_set=feature_extractor)
        get_textures(texture_hop, features_filename, all_textures_filename)

    # exit(0)

    #skip this number of experiments
    skip = 0
    skipped = 0

    for textures_number in texture_numbers:
        print("+++ TEXTURES NUMBER %d" % textures_number)
        
        for texture_selector in texture_selectors:
            print("### TEXTURE SELECTOR %s" % texture_selector)

            for fold in (np.arange(n_folds) + 1):
                print("@@@ FOLD %d" % fold)
                
                if skipped < skip:
                    skipped+=1
                    print('skipping..')
                    continue
                
                selector_args = {
                    '_select_frames' : {
                        'selector' : texture_selector,
                        'n_frames' : textures_number
                    }
                }              

                train_filelist = folds_folder + ('f%d_train.txt' % fold)
                test_filelist = folds_folder +  ('f%d_test.txt' % fold)
                evaluate_filelist = folds_folder + ('f%d_evaluate.txt' % fold)

                ####################### TRAINING ##########################
                train_files, _ = parse_filelist(train_filelist)
                all_textures = dill.load(open(all_textures_filename))
                ss = StandardScaler()
                print('estimating standard scaler...')
                for f in train_files:
                    ss.partial_fit(all_textures.estimator_output[all_textures.get_single_track_idxs(f),:])
                all_textures.close_h5()

                t0 = time.time()
                try:
                    print('selecting textures for training...')
                    train_textures = select_textures(all_textures_filename, 
                                                    train_filelist,
                                                    train_textures_filename,
                                                    ss=ss,
                                                    **selector_args)
                except IndexError as ie:
                    print("The following exception happened while selecting textures: %s" % ie)
                    print('Probably an error in the sklearn kmeans implementation.')
                    print('Skipping this fold...')
                    
                    continue
                print('Texture selection took %.2f seconds.' % (time.time() - t0))

                train_textures.reload_h5_storage()

                print('training SVM Model...')
                svm = SvmAnova(params=svm_params)
                svm = svm.fit(train_textures, train_list_file=train_filelist)
                dill.dump(svm, open(scratch_folder + 'trained_model.dill', 'w'))

                print('training KNN (NO ANOVA) model...')
                knn = KNN(params=knn_params)
                knn = knn.fit(train_textures, train_list_file=train_filelist)
                dill.dump(knn, open(scratch_folder + 'trained_model_knn.dill', 'w'))

                print('training KNN (ANOVA) model...')
                knn_anova = KNN(params=knn_anova_params)
                knn_anova = knn_anova.fit(train_textures, train_list_file=train_filelist)
                dill.dump(knn_anova, open(scratch_folder + 'trained_model_knn_anova.dill', 'w'))

                print('done training.')
                train_textures.close_h5()
                dill.dump(ss, open(scratch_folder + 'ss.dill', 'w'))

                ############################TESTING##########################
                t0 = time.time()
                try:
                    print('selecting textures for testing...')
                    test_textures = select_textures(all_textures_filename,
                                                    test_filelist, 
                                                    test_textures_filename,
                                                    ss=ss,
                                                    **selector_args)

                except IndexError as ie:
                    print("The following exception happened while selecting textures: %s" % ie)
                    print('Probably an error in the sklearn kmeans implementation.')
                    print('Skipping this fold...')
            
                    continue                
                print('Texture selection took %.2f seconds.' % (time.time() - t0))

                test_textures.reload_h5_storage()

                print('classifying frames...')
                mt = SklearnLike(params=tester_params)

                print('classifying with SVM...')
                predictions = mt.predict(scratch_folder+'trained_model.dill', 
                                        test_textures, 
                                        test_filelist, 
                                        voting_model=False)
                
                dill.dump(predictions, open(scratch_folder + 'frame_predictions.dill', 'w'))

                print('classifying with KNN (NO ANOVA)...')
                predictions_knn = mt.predict(scratch_folder+'trained_model_knn.dill', 
                                        test_textures, 
                                        test_filelist, 
                                        voting_model=False)

                dill.dump(predictions_knn, open(scratch_folder + 'frame_predictions_knn.dill', 'w'))

                print('classifying with KNN (ANOVA)...')
                predictions_knn_anova = mt.predict(scratch_folder+'trained_model_knn_anova.dill', 
                                        test_textures, 
                                        test_filelist, 
                                        voting_model=False)

                dill.dump(predictions_knn_anova, open(scratch_folder + 'frame_predictions_knn_anova.dill', 'w'))

                print('done classifying frames...')
                test_textures.close_h5()

                ###########################EVALUATING###########################
                
                voter = MaxVoting(params=None)

                print('max voting & evaluating SVM results...')
                final_predictions = voter.vote(predictions, output_fmt='text')
                with open(scratch_folder + 'final_predictions.txt', 'w') as f:
                    f.write(final_predictions)
                
                eval_classification(['script',scratch_folder + 'final_predictions.txt', evaluate_filelist])

                print('max voting & evaluating KNN (NO ANOVA) results...')
                final_predictions = voter.vote(predictions_knn, output_fmt='text')
                with open(scratch_folder + 'final_predictions_knn.txt', 'w') as f:
                    f.write(final_predictions)
                
                eval_classification(['script',scratch_folder + 'final_predictions_knn.txt', evaluate_filelist])

                print('max voting & evaluating KNN (ANOVA) results...')
                final_predictions = voter.vote(predictions_knn_anova, output_fmt='text')
                with open(scratch_folder + 'final_predictions_knn_anova.txt', 'w') as f:
                    f.write(final_predictions)
                
                eval_classification(['script',scratch_folder + 'final_predictions_knn_anova.txt', evaluate_filelist])

                print('done!')

    release_scratch_lock(scratch_folder)
