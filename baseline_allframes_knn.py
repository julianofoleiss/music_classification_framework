
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
from core.random_projection import get_rp_features
from core.identity_features import get_identity_features
from core.parallel_extraction import from_filelist
from core.texture_avg_stdev_filter import TextureAvgStdevFilter
from core.data.predicted_features import PredictedFeatures
from core import select_frames
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

        if params['_parallel_fe']['centerk']:
            #print('selecting center frames', params['_parallel_fe']['centerk'])
            centerk_params = {
                'centerk_selector' : {
                    'pad': False
                },
                'frame_selector': {
                    'n_frames': params['_parallel_fe']['centerk'],
                    'normalize': False,
                    'subset_size': None
                }
            }
            fs = select_frames.FrameSelector.get_selector('centerk', centerk_params )
            feats = fs.select_frames(feats)

        out.append(feats)
    return out

def extract_features(filelist, fe_params, output_filename, centerk, feature_set='marsyas'):
    if feature_set == 'marsyas':
        fe_params.update({'_parallel_fe': {'extraction_fcn' : get_gtzan_features, 'centerk': centerk}})
    elif feature_set == 'mel':
        fe_params.update({'_parallel_fe': {'extraction_fcn' : get_stft_features, 'centerk': centerk}})
    elif feature_set == 'ae':
        fe_params.update({'_parallel_fe': {'extraction_fcn' : get_identity_features, 'centerk': centerk}})        
    elif feature_set == 'rp':
        fe_params['random_projection']['projection_matrix'] = dill.load(open(fe_params['random_projection']['projection_matrix']))
        fe_params.update({'_parallel_fe': {'extraction_fcn' : get_rp_features, 'centerk': centerk}})
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


#if __name__ == "__main__":
@click.command()
@click.argument(
    'fold_num',
    type=int,
)
@click.argument(
    'scratch-folder',
    type=str,
    )
@click.option(
    '--extract-feats',
    is_flag=True,
    help='if set, features are extracted. otherwise, use features already in the scratch folder.'
)   
@click.option(
    '--centerk',
    default=-1,
    help='if != -1, keep this number of frames from the center of each track!'
)
@click.option(
    '--ae_hs',
    default=16,
    help='size of the bottleneck for autoencoder features'
)
@click.option(
    '--ae_path',
    default=None,
    help='path to the autoencoders'
)
@click.option(
    '--feature-set',
    type=click.Choice(['ae', 'marsyas', 'mel', 'rp']),
    default='marsyas',
    help='the feature set used to represent audio'
)
@click.option(
    '--dataset',
    type=click.Choice(['gtzan_10fs', 'gtzan_art3f', 'lmd', 'homburg', 'ismir', 'lmd_10s', 'ismir_10s']),
    default='gtzan_10fs',
    help='the dataset to be used'
)
@click.option(
    '--extract-only',
    is_flag=True,
    help='feature extraction only'
)
@click.option(
    '--rp-target',
    default=8,
    help='the target dimensionality of the random projection matrix.'
)
@click.option(
    '--rp-path-pattern',
    default=None,
    help='path to random projection matrices in string format pattern with %d marking the target dimensionality'
)
def main(fold_num, scratch_folder, extract_feats, centerk, ae_hs, ae_path, feature_set, dataset, extract_only, rp_target, rp_path_pattern):

    # Classification pipeline (frame selection)

    ######SETTINGS FOR GTZAN 3F ARTIST FILTER############
    if dataset == 'gtzan_art3f':
        if feature_set == 'ae':
            folds_folder = 'gtzan_folds_filter_3f_strat_brava/'
        else:
            folds_folder = 'gtzan_folds_filter_3f_strat/'
        
        labels_file = folds_folder + 'gtzan_labels.txt'
        dataset = 'gtzan'
        texture_length = 216
        texture_hop = 10
        dict_file = folds_folder + 'gtzan_dict.py'
        n_folds = 3

    ######SETTINGS FOR GTZAN 10F RANDOM############
    if dataset == 'gtzan_10fs':
        if feature_set == 'ae':    
            folds_folder = 'gtzan_folds_10fs_brava/'            
        else:
            folds_folder = 'gtzan_folds_10fs/'

        labels_file = folds_folder + 'gtzan_labels.txt'
        dataset = 'gtzan'
        texture_length = 216
        texture_hop = 10
        dict_file = folds_folder + 'gtzan_dict.py'
        n_folds = 10

    ######SETTINGS FOR LMD############
    if dataset == 'lmd':
        if feature_set == 'ae':     
            folds_folder = 'lmd_folds_brava/'
        else:
            folds_folder = 'lmd_folds/'

        labels_file = folds_folder + 'lmd_labels.txt'
        dataset = 'lmd'
        texture_length = 216
        texture_hop = 10
        dict_file = folds_folder + 'lmd_dict.py'
        n_folds = 3

    ######SETTINGS FOR LMD 10s############
    if dataset == 'lmd_10s':
        if feature_set == 'ae':     
            folds_folder = 'lmd_folds_10s_brava/'
        else:
            folds_folder = 'lmd_folds_10s/'

        labels_file = folds_folder + 'lmd_labels.txt'
        dataset = 'lmd_10s'
        texture_length = 216
        texture_hop = 10
        dict_file = folds_folder + 'lmd_dict.py'
        n_folds = 3

    ######SETTINGS FOR HOMBURG############
    if dataset == 'homburg':
        if feature_set == 'ae':     
            folds_folder = 'homburg_folds_brava/'
        else:
            folds_folder = 'homburg_folds/'
    
        labels_file = folds_folder + 'homburg_labels.txt'
        dataset = 'homburg'
        texture_length = 216
        texture_hop = 10
        dict_file = folds_folder + 'homburg_dict.py'
        n_folds = 10

    ######SETTINGS FOR ISMIR############
    if dataset == 'ismir':
        if feature_set == 'ae':     
            folds_folder = 'ismir_folds_brava/'
        else:
            folds_folder = 'ismir_folds/'    
        labels_file = folds_folder + 'ismir_labels.txt'
        dataset = 'ismir'
        texture_length = 216
        texture_hop = 10
        dict_file = folds_folder + 'ismir_dict.py'
        n_folds = 1

    ######SETTINGS FOR ISMIR 10s############
    if dataset == 'ismir_10s':
        if feature_set == 'ae':     
            folds_folder = 'ismir_folds_10s_brava/'
        else:
            folds_folder = 'ismir_folds_10s/'    
        labels_file = folds_folder + 'ismir_labels.txt'
        dataset = 'ismir_10s'
        texture_length = 216
        texture_hop = 10
        dict_file = folds_folder + 'ismir_dict.py'
        n_folds = 1        

    ######SETTINGS FOR EXBALLROOM############
    # #scratch_folder = 'scratch11/'
    # folds_folder = 'exballroom_folds/'
    # labels_file = folds_folder + 'exballroom_labels.txt'
    # dataset = 'exballroom'
    # texture_length = 216
    # texture_hop = 10
    # dict_file = folds_folder + 'exballroom_dict.py'
    # n_folds = 10

    print("Dataset Settings:")
    print("dataset: %s" % dataset)
    print('scratch_folder: %s' % scratch_folder)
    print('folds_folder: %s' % folds_folder)
    print('labels_file: %s' % labels_file)
    print('texture_length: %d' % texture_length)
    print('texture_hop: %d' % texture_hop)
    print('n_folds: %d' % n_folds)

    ################EXPERIMENT SETTINGS##########################

    features_filename = scratch_folder + 'features.dill'
    all_textures_filename = scratch_folder + 'all_features.dill'
    train_textures_filename = scratch_folder + 'train_textures.dill'
    test_textures_filename = scratch_folder + 'test_textures.dill'

    feature_extractor = feature_set

    task_extract = extract_feats
    #task_extract = False

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
            'n_fft' : 2048,
            'hop_length' : 1024,
            'target_sr' : 44100,
            'deltas' : True,
            'delta_deltas' : True,
            'mel_bins': 128,
            'in_db': False
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

    rp_fe_params = {
        'random_projection' : {
            'n_fft' : 2048,
            'hop_length' : 1024,
            'target_sr' : 44100,
            'deltas' : True,
            'delta_deltas' : True,
            'mel_bins': 128,
            'non_linearity': None,
            'in_db': False,
            'projection_matrix': None
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

    ae_fe_params = {
        'identity_features': {
            'predicted_features_input': None,
            'deltas': True,
            'delta_deltas': True
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

    tsvm_params={
        'thundersvm': {
            'gammas': ['auto', 0.1, 0.001, 0.0001],
            'Cs': [10, 100, 1000],
            # 'gammas': ['auto'],
            # 'Cs': [10],            
            'probability': False,
            'kernel': 'rbf',
            'hyperparameter_tuning': 'single_split',
            'num_gpus': [1],
        },
        'general': {
            'label_dict_file': dict_file
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

        if centerk <= 0:
            centerk = False

        fe_params = eval(feature_extractor + '_fe_params')

        if feature_extractor == 'ae':
            features_file = ae_path + ('f%d_features_%d-feats.dill' % (fold_num, ae_hs))
            print('Loading autoencoder features from %s...' % features_file)
            fe_params['identity_features']['predicted_features_input'] = features_file

        if feature_extractor == 'rp':
            rpm_file = rp_path_pattern % (rp_target)
            print('Loading random projection matrix from %s...' % rpm_file)
            fe_params['random_projection']['projection_matrix'] = rpm_file

        extract_features(labels_file, fe_params, features_filename, centerk, feature_set=feature_extractor)
        get_textures(texture_hop, features_filename, all_textures_filename)

    if extract_only:
        release_scratch_lock(scratch_folder)
        exit(0)

    #skip this number of experiments
    skip = 0
    skipped = 0

    #for fold in (np.arange(n_folds) + 1):
    for fold in [fold_num]:
        print("@@@ FOLD %d" % fold)
        
        if skipped < skip:
            skipped+=1
            print('skipping..')
            continue        

        train_filelist = folds_folder + ('f%d_train.txt' % fold)
        test_filelist = folds_folder +  ('f%d_test.txt' % fold)
        evaluate_filelist = folds_folder + ('f%d_evaluate.txt' % fold)

        ####################### TRAINING ##########################
        train_files, _ = parse_filelist(train_filelist)
        all_textures = dill.load(open(all_textures_filename))

        train_data = []
        train_len = []
        train_filenames = []
        train_labels = []

        for f in train_files:
            train_data.append(all_textures.estimator_output[all_textures.get_single_track_idxs(f),:])
            train_len.append(train_data[-1].shape[0])
            train_filenames.append(f)
            train_labels.append(all_textures.get_track_label(f))

        train_textures = PredictedFeatures(train_filenames, train_len, np.concatenate(train_data), all_textures.label_idx_dict,
            [None] * len(train_filenames), train_labels)

        # print('training SVM Model...')
        # svm = ThunderSVM(params=tsvm_params)
        # svm = svm.fit(train_textures, train_list_file=train_filelist)
        # #dill.dump(svm, open(scratch_folder + 'trained_model.dill', 'w'))
        
        # svm.named_steps['svm'].save_to_file(scratch_folder + 'trained_model.thundersvm')
        # dill.dump(svm.named_steps['standardizer'], open(scratch_folder + 'trained_model.thundersvm.ss', 'w'))

        print('training KNN (NO ANOVA) model...')
        knn = KNN(params=knn_params)
        knn = knn.fit(train_textures, train_list_file=train_filelist)
        dill.dump(knn, open(scratch_folder + 'trained_model_knn.dill', 'w'))

        print('training KNN (ANOVA) model...')
        knn_anova = KNN(params=knn_anova_params)
        knn_anova = knn_anova.fit(train_textures, train_list_file=train_filelist)
        dill.dump(knn_anova, open(scratch_folder + 'trained_model_knn_anova.dill', 'w'))

        print('done training.')        

        ############################TESTING##########################
        print('classifying frames...')
        mt = SklearnLike(params=tester_params)

        # print('classifying with SVM...')
        # predictions = mt.predict(scratch_folder+'trained_model.thundersvm', 
        #                         all_textures, 
        #                         test_filelist, 
        #                         voting_model=False)
        
        # mt.model = None

        # dill.dump(predictions, open(scratch_folder + 'frame_predictions.dill', 'w'))

        print('classifying with KNN (NO ANOVA)...')
        predictions_knn = mt.predict(scratch_folder+'trained_model_knn.dill', 
                                all_textures, 
                                test_filelist, 
                                voting_model=False)

        dill.dump(predictions_knn, open(scratch_folder + 'frame_predictions_knn.dill', 'w'))

        print('classifying with KNN (ANOVA)...')
        predictions_knn_anova = mt.predict(scratch_folder+'trained_model_knn_anova.dill', 
                                all_textures, 
                                test_filelist, 
                                voting_model=False)

        dill.dump(predictions_knn_anova, open(scratch_folder + 'frame_predictions_knn_anova.dill', 'w'))

        print('done classifying frames...')
        all_textures.close_h5()

        ###########################EVALUATING###########################
        
        voter = MaxVoting(params=None)

        # print('max voting & evaluating SVM results...')
        # final_predictions = voter.vote(predictions, output_fmt='text')
        # with open(scratch_folder + 'final_predictions.txt', 'w') as f:
        #     f.write(final_predictions)
        
        # eval_classification(['script',scratch_folder + 'final_predictions.txt', evaluate_filelist])

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

        os.system('nvidia-smi')
        os.system('free -h')

    release_scratch_lock(scratch_folder)

if __name__ == '__main__':
    main()

