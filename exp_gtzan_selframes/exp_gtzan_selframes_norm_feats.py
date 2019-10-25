# -*- coding: utf-8 -*-

from __future__ import print_function
import os, sys
import yaml
import dill
import numpy as np
import time

import core.utils.param_handling
from core import gtzan_features, select_frames, texture_avg_stdev_filter, ivector_filter, beat_filter, random_projection, identity_features, stft_features, pca_features, cqt_features, linspace_downsample_filter
from core.classifier_pipelines.svm_anova import SvmAnova
from core.test_pipelines.sklearnlike_tester import SklearnLike
from core.prediction_voting.max_voting import MaxVoting
from core.prediction_voting.sum_voting import SumVoting
from core.data.pytable import create_earray
from core.data.filelists import parse_filelist
from core.parallel_extraction import from_filelist
from core.data.predicted_features import PredictedFeatures
from core.classifier_pipelines.random_forest import RandomForest
from core.classifier_pipelines.knn import KNN

from core.classifier_pipelines.attention_net import make_keras_pickable, AttentionNetClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import Parallel, delayed

def _parallel_fe(args):

    params, input_files, tags = args

    out = []

    extraction_fcn = params['_parallel_fe']['extraction_fcn']

    for i in range(len(input_files)):
        
        feats = extraction_fcn(input_files[i], tags[i], params=params, output_file=None)

        if params['exp_gtzan_selframes_norm_feats']['texture_aggregation']:
            avg_filter = texture_avg_stdev_filter.TextureAvgStdevFilter(params)
            filtered = avg_filter.filter(feats)
        else:
            filtered = feats

        if params['exp_gtzan_selframes_norm_feats']['beat_aggregation']:
            bf = beat_filter.BeatAggregatorFilter(params)
            agg_fcn = np.median if params['exp_gtzan_selframes_norm_feats']['ba_aggregate'] == 'median' else np.mean
            filtered =  bf.filter(filtered, 
                audio_filename=input_files[i], 
                sr=params['exp_gtzan_selframes_norm_feats']['ba_sr'], 
                hop_length=params['exp_gtzan_selframes_norm_feats']['ba_hop_length'], 
                aggregation=agg_fcn)

        if params['exp_gtzan_selframes_norm_feats']['linspace_downsampling']:
            downsampler = linspace_downsample_filter.LinspaceDownsampleFilter(params)
            filtered = downsampler.filter(filtered)                
                
        if params['exp_gtzan_selframes_norm_feats']['ivector']:
            ivec_filter = ivector_filter.IVectorFilter(params)
            out.append(ivec_filter.filter(filtered))
        else:
            out.append(filtered)

    return out


def feature_extraction(params):

    features_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes_norm_feats']['low_level_feature_filename']
    extraction_workers = params['exp_gtzan_selframes']['extraction_workers']

    #Extract all low level features -- frame selection should be done just before training.
    #See training for further details.
    print('Extracting low level features...')

    extraction_fcn = None

    if params['exp_gtzan_selframes_norm_feats']['feature_extractor'] == 'random_projection':
        projection_matrix = dill.load(open(params['exp_gtzan_selframes_norm_feats']['rp_projection_matrix']))
        params['random_projection']['projection_matrix'] = projection_matrix
        extraction_params = { '_parallel_fe' : { 'extraction_fcn' : random_projection.get_rp_features } }

    elif params['exp_gtzan_selframes_norm_feats']['feature_extractor'] == 'gtzan':
        extraction_params = { '_parallel_fe' : { 'extraction_fcn' : gtzan_features.get_gtzan_features } }

    elif params['exp_gtzan_selframes_norm_feats']['feature_extractor'] == 'identity':
        extraction_params = { '_parallel_fe' : { 'extraction_fcn' : identity_features.get_identity_features } }

    elif params['exp_gtzan_selframes_norm_feats']['feature_extractor'] == 'stft':
        extraction_params = { '_parallel_fe' : { 'extraction_fcn' : stft_features.get_stft_features } }

    elif params['exp_gtzan_selframes_norm_feats']['feature_extractor'] == 'pca':
        extraction_params = { '_parallel_fe' : { 'extraction_fcn' : pca_features.get_pca_features} }

    elif params['exp_gtzan_selframes_norm_feats']['feature_extractor'] == 'cqt':
        extraction_params = { '_parallel_fe' : { 'extraction_fcn' : cqt_features.get_cqt_features} }

    extraction_params.update(params)

    from_filelist(params['exp_gtzan_selframes']['feature_extraction_filelist'], 100, 3, extraction_workers, 
        _parallel_fe, extraction_params, features_filename, out_format='h5', inscript=params['general']['in_script'])

def _select_frames(args):

    params, input_files, tags = args

    ss = params['_select_frames']['ss']
    pf = dill.load(open(params['_select_frames']['pf_filename']))
    out = []

    # idxs = [ pf.get_single_track_idxs(f) for f in input_files ]
    # all_features = pf.estimator_output[np.concatenate(idxs),:]
    # print (all_features.shape)
    # all_features = ss.transform(all_features)
    # all_idxs = np.array(idxs) - idxs[0][0]
    
    # for i in xrange(len(input_files)):
    #     features = all_features[all_idxs[i],:]
        
    #     fs = select_frames.FrameSelector.get_selector( params['exp_gtzan_selframes']['frame_selector'], params )
    #     selected = fs._select_frames(features, params['frame_selector']['n_frames'])
    #     selected = PredictedFeatures([input_files[i]], [selected.shape[0]], selected, pf.label_idx_dict, [None], pf.get_track_label(input_files[i]))
    #     out.append(selected)    

    for f in input_files:
        features = pf.estimator_output[pf.get_single_track_idxs(f),:]
        features = ss.transform(features)
        fs = select_frames.FrameSelector.get_selector( params['exp_gtzan_selframes']['frame_selector'], params )

        selected = fs._select_frames(features, params['frame_selector']['n_frames'])

        if params['frame_selector']['subset_size'] is not None:
            selected = selected[:params['frame_selector']['subset_size']]

        selected = PredictedFeatures([f], [selected.shape[0]], selected, pf.label_idx_dict, [None], pf.get_track_label(f))
        out.append(selected)

    if hasattr(pf, 'h5file'):
        pf.close_h5()

    return out

def train_model(params):
    features_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes_norm_feats']['low_level_feature_filename']
    
    train_filelist = params['exp_gtzan_selframes']['train_filelist']
    train_features_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes_norm_feats']['train_features_filename']
    feature_scaler_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes_norm_feats']['feature_scaler_filename']
    extraction_workers = params['exp_gtzan_selframes']['extraction_workers']

    pf = dill.load(open(features_filename))
    
    #First it is necessary to go through all train examples to determine the features' means and std_devs.
    train_files, _ = parse_filelist(train_filelist)
    ss = StandardScaler()

    fit_ss = True

    if params['exp_gtzan_selframes_norm_feats']['smart_feature_scaler_loading']:
        #build the path to the standard scaler
        fold = params['exp_gtzan_selframes']['train_filelist'].split('/')[-1].split('_')[0]
        print (params['exp_gtzan_selframes']['train_filelist'])
        ss_filename = params['general']['scratch_directory'] + params['general']['dataset_name'] + '_' + fold + '.ss.dill'

        if os.path.isfile(ss_filename):
            print('Standard Scaler found! Loading from file %s...' % (ss_filename))
            ss = dill.load(open(ss_filename))
            fit_ss = False
        else:
            print('Standard Scaler not found: %s.' % (ss_filename))

    if fit_ss:
        print('Fitting standard scaler on train data...')
        for k in train_files:
            ss.partial_fit(pf.estimator_output[pf.get_single_track_idxs(k),:])

        #dump standard scaler. This will be used again to scale the test data.
        if params['exp_gtzan_selframes_norm_feats']['smart_feature_scaler_loading']:
            outf = ss_filename
        else:
            outf = feature_scaler_filename
        dill.dump(ss, open(outf, 'w'))

        # print('loading standard scaler for fold 1 of LMD (remove this if using another fold')
        # ss = dill.load(open(feature_scaler_filename))

        if hasattr(pf, 'h5file'):
            pf.close_h5()

    #select frames!
    select_frames_params = {'_select_frames' : {'ss': ss, 'pf_filename': features_filename} }
    select_frames_params.update(params)

    print('Scaling training data and selecting frames...')

    t0 = time.time()

    #DESCOMENTAR ISSOO!!!!!!!!!!!!!!!!!!
    train_feats = from_filelist(train_filelist, 100, 3, extraction_workers, _select_frames, select_frames_params, 
        train_features_filename, out_format='h5', inscript=params['general']['in_script'])
    print('Frame selection took %.2f seconds'% (time.time()-t0))

    #uncomment the following dill loading when out_format='h5' above
    print('loading train features h5 file...')
    train_feats = dill.load(open(train_features_filename))

    #check whether the group file exists and hack it into the params dict. This is a quick hack. I should refactor this later on.
    f, e = os.path.splitext(train_filelist)
    gf = f + '_groups' + e
    print('checking for %s...' % gf)
    if os.path.isfile(gf):
        print('Found a groups file for %s. Adding it to the grid search parameters.' % train_filelist)
        params['single_split_gs']['groups_file'] = gf
    else:
        print('%s not found!', gf)

    print (params['single_split_gs'])

    print('Training model...')
    if params['exp_gtzan_selframes']['classifier'] == 'svm_anova':
        model = SvmAnova(params)
    elif params['exp_gtzan_selframes']['classifier'] == 'random_forest':
        model = RandomForest(params)
    elif params['exp_gtzan_selframes']['classifier'] == 'knn':
        model = KNN(params)
    elif params['exp_gtzan_selframes']['classifier'] == 'attnet':
        model = AttentionNetClassifier(params)
    else:
        print('please set the classifier to one of svm_anova, random_forest, knn or attnet')

    model = model.fit(train_feats, train_filelist)

    model_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes']['model_file']

    dill.dump(model, open( model_filename, 'w'))

    #uncomment the following dill loading when out_format='h5' above
    if hasattr(train_feats, 'h5file'):
        train_feats.close_h5()

def train(params):

    train_model(params)

def test(params):

    features_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes_norm_feats']['low_level_feature_filename']
    
    test_filelist = params['exp_gtzan_selframes']['test_filelist']
    test_features_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes_norm_feats']['test_features_filename']
    feature_scaler_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes_norm_feats']['feature_scaler_filename']
    extraction_workers = params['exp_gtzan_selframes']['extraction_workers']

    load_absolute_ss = True

    if params['exp_gtzan_selframes_norm_feats']['smart_feature_scaler_loading']:
        #build the path to the standard scaler
        fold = params['exp_gtzan_selframes']['test_filelist'].split('/')[-1].split('_')[0]
        ss_filename = params['general']['scratch_directory'] + params['general']['dataset_name'] + '_' + fold + '.ss.dill'

        if os.path.isfile(ss_filename):
            print('Standard Scaler found! Loading from file %s...' % (ss_filename))
            ss = dill.load(open(ss_filename))
            load_absolute_ss = False
        else:
            print('Standard Scaler not found: %s. Trying to load Standard Scaler from the parameter file.' % (ss_filename))

    if load_absolute_ss:
        ss = dill.load(open(feature_scaler_filename))

    #select frames!
    select_frames_params = {'_select_frames' : {'ss': ss, 'pf_filename': features_filename} }
    select_frames_params.update(params)
    print('Scaling test data and selecting frames...')
    test_feats = from_filelist(test_filelist, 100, 3, extraction_workers, _select_frames, select_frames_params, 
        test_features_filename, out_format='h5', inscript=params['general']['in_script'])

    print('Testing model...')

    model_filename = params['general']['scratch_directory'] + params['exp_gtzan_selframes']['model_file']

    tester = SklearnLike(params)

    voting_model = False
    if params['exp_gtzan_selframes']['classifier'] == 'attnet':
        voting_model = True

    test_feats = test_features_filename

    if voting_model:
        preds = tester.predict(model_filename, test_feats, test_filelist, voting_model=True)
    else:
        preds = tester.predict(model_filename, test_feats, test_filelist)

    dill.dump(preds, open(params['exp_gtzan_selframes']['frame_predictions'], 'w'))

    dill.dump(preds, open(params['exp_gtzan_selframes']['final_prediction'] + '.frames.dill', 'w'))

def vote(params):

    print('Voting...')

    preds = dill.load( open(params['exp_gtzan_selframes']['frame_predictions']) )

    if params['exp_gtzan_selframes']['voting'] == 'max':
        voter = MaxVoting(params)

    if params['exp_gtzan_selframes']['voting'] == 'sum':
        voter = SumVoting(params)
        
    res = voter.vote(preds, output_fmt='text')

    out = open(params['exp_gtzan_selframes']['final_prediction'], 'w')
    out.write(res)
    out.close()

def run(params):

    make_keras_pickable()

    if params['general']['dataset_name'] is None:
        print("You must set general.dataset_name! i.e. add -ss general.dataset_name=gtzan to the command-line!")
        exit(1)

    if params['exp_gtzan_selframes']['extract_features']:
        feature_extraction(params)

    if params['exp_gtzan_selframes']['train']:
        train(params)

    if params['exp_gtzan_selframes']['test']:
        test(params)

    if params['exp_gtzan_selframes']['vote']:
        vote(params)

if __name__ == "__main__":
    run(sys.argv)
    
def parse_commandline(argv):
    def switch_extract(argv):
        ov = [("exp_gtzan_selframes.extract_features", True),
                ("exp_gtzan_selframes.train", False),
                ("exp_gtzan_selframes.test", False)]

        ex_pos = argv.index('-extract')
        if len(argv) - ex_pos < 3:
            print ("wrong number of arguments for \'extract\'. usage: %s -extract path_to_scratch_folder path_to_extract_filelist" % \
                    (argv[0]))
            exit(1)
        ov.append(("general.scratch_directory", argv[ex_pos+1]))
        ov.append(("exp_gtzan_selframes.feature_extraction_filelist", argv[ex_pos+2]))
        return ov

    def switch_train(argv):
        ov = [("exp_gtzan_selframes.extract_features", False),
                ("exp_gtzan_selframes.train", True),
                ("exp_gtzan_selframes.test", False)]

        ex_pos = argv.index('-train')
        if len(argv) - ex_pos < 3:
            print ("wrong number of arguments for \'train\'. usage: %s -train path_to_scratch_folder path_to_train_filelist" % \
                  (argv[0]))
            exit(1)
        ov.append(("general.scratch_directory", argv[ex_pos+1]))
        ov.append(("exp_gtzan_selframes.train_filelist", argv[ex_pos+2]))

        return ov

    def switch_test(argv):
        ov = [("exp_gtzan_selframes.extract_features", False),
                ("exp_gtzan_selframes.train", False),
                ("exp_gtzan_selframes.test", True),
                ("exp_gtzan_selframes.vote", True)]

        ex_pos = argv.index('-test')
    
        if len(argv) - ex_pos < 4:
            print ("wrong number of arguments for \'test\'. usage: %s -test path_to_scratch_folder path_to_test_filelist path_to_predict_file" % \
                  (argv[0]))
            exit(1)
        ov.append(("general.scratch_directory", argv[ex_pos + 1]))
        ov.append(("exp_gtzan_selframes.test_filelist", argv[ex_pos + 2]))
        ov.append(("exp_gtzan_selframes.frame_predictions", argv[ex_pos + 1] + '/frame_predictions.dill'))
        ov.append(("exp_gtzan_selframes.final_prediction", argv[ex_pos + 3]))
        

        return ov

    ovw = []

    if '-extract' in argv:
        ovw.extend(switch_extract(argv))

    elif '-train' in argv:
        ovw.extend(switch_train(argv))
    
    elif '-test' in argv:
        ovw.extend(switch_test(argv)) 

    return ovw
