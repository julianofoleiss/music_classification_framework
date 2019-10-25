#!/bin/bash
DATASET_NAME=homburg
FSELECTORS=(random)
FOLDS=(1 2 3 4 5 6 7 8 9 10)
N_FRAMES=(5 20 40)
N_FRAMES_SEL_P=1
FOLDSFOLDER=homburg_folds_brava/
SCRATCH_FOLDER=`pwd`/scratch9/
METAFILE=homburg_folds_brava/homburg_labels.txt
OUTFOLDER=res_ae_selframes_homburg_svm-class_nxn-desc_1-sel_linspace-downsampler_5s25s-dsettings_999-seed/
TIME=/usr/bin/time
EXPERIMENT_SCRIPT=exp_gtzan_selframes/exp_gtzan_selframes_norm_feats.py
PARAMETER_FILE=exp_gtzan_selframes/parameters.yaml
LABEL_DICT=homburg_folds_brava/homburg_dict.py
#HIDDEN_SIZE=(16 32 64 128 256)

# HIDDEN_SIZE=(16 32) #(16 32 64 128 256)
# FEATURES_FOLDER=ae_features_lmd_128-mel_16,32-hs/

# HIDDEN_SIZE=(64 128 256)
# FEATURES_FOLDER=ae_features_lmd_128-mel/

HIDDEN_SIZE=(16 32 64 128)
FEATURES_FOLDER=ae_features_homburg_128-mel/

DELTAS=True
CLASSIFIER=svm_anova #svm_anova
SORT_CENTROIDS=asc
NO_ANOVA=True
RANDOM_SEL_SEED=999

export OPENBLAS_NUM_THREADS=1

for hs in ${HIDDEN_SIZE[@]}
do
	echo "@@@ HIDDEN_SIZE $hs"

	for nf in ${N_FRAMES[@]}
	do
		echo "### N_FRAMES $nf"
		n_frames_tt=`echo "$nf*$N_FRAMES_SEL_P" | bc -l | awk '{print ($0-int($0)>0)?int($0)+1:int($0)}'`
		for fs in ${FSELECTORS[@]}
		do
			echo "*** FSELECTOR $fs"

			for fold in ${FOLDS[@]}
			do
				echo "!!! FOLD $fold"
				$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -extract $SCRATCH_FOLDER $METAFILE -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes_norm_feats.feature_extractor=identity -ss exp_gtzan_selframes_norm_feats.low_level_feature_filename=f${fold}_${hs}-feats_ll_features.dill -ss identity_features.predicted_features_input=$FEATURES_FOLDER/f${fold}_features_${hs}-feats.dill -sb identity_features.deltas=$DELTAS -sb identity_features.delta_deltas=$DELTAS -ss general.dataset_name=$DATASET_NAME 
				$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_train.txt -ss exp_gtzan_selframes.frame_selector=$fs -si frame_selector.n_frames=$nf -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes_norm_feats.low_level_feature_filename=f${fold}_${hs}-feats_ll_features.dill -ss exp_gtzan_selframes.classifier=$CLASSIFIER -ss kmeansc_selector.sort_centroids_by_common=$SORT_CENTROIDS -si frame_selector.subset_size=$nf -sb svm_anova.no_anova=$NO_ANOVA -sb knn.no_anova=$NO_ANOVA -ss general.dataset_name=$DATASET_NAME -si random_selector.seed=$RANDOM_SEL_SEED 
				$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -test $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_test.txt $OUTFOLDER/${nf}-nf_${fs}-fs_${hs}-hs_${fold}-f.predict -ss exp_gtzan_selframes.frame_selector=$fs -si frame_selector.n_frames=$nf -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes_norm_feats.low_level_feature_filename=f${fold}_${hs}-feats_ll_features.dill -ss kmeansc_selector.sort_centroids_by_common=$SORT_CENTROIDS -si frame_selector.subset_size=$nf -ss general.dataset_name=$DATASET_NAME -si random_selector.seed=$RANDOM_SEL_SEED
				$TIME python eval_classification.py $OUTFOLDER/${nf}-nf_${fs}-fs_${hs}-hs_${fold}-f.predict $METAFILE > $OUTFOLDER/${nf}-nf_${fs}-fs_${hs}-hs_${fold}-f.eval
				rm $SCRATCH_FOLDER/*
			done
		done
	done

	# for fold in ${FOLDS[@]}
	# do
	# 	echo "!!! FOLD $fold"
	# 	$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -extract $SCRATCH_FOLDER $METAFILE -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes_norm_feats.feature_extractor=identity -ss exp_gtzan_selframes_norm_feats.low_level_feature_filename=f${fold}_${hs}-feats_ll_features.dill -ss identity_features.predicted_features_input=$FEATURES_FOLDER/f${fold}_features_${hs}-feats.dill -sb exp_gtzan_selframes_norm_feats.ivector=True -ss general.dataset_name=$DATASET_NAME -sb exp_gtzan_selframes_norm_feats.linspace_downsampling=False

	# 	$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_train.txt -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes_norm_feats.low_level_feature_filename=f${fold}_${hs}-feats_ll_features.dill -ss exp_gtzan_selframes.classifier=$CLASSIFIER -ss general.dataset_name=$DATASET_NAME -sb svm_anova.no_anova=$NO_ANOVA -sb knn.no_anova=$NO_ANOVA
		
	# 	$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -test $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_test.txt $OUTFOLDER/ivector-fs_${hs}-hs_${fold}-f.predict -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes_norm_feats.low_level_feature_filename=f${fold}_${hs}-feats_ll_features.dill -ss general.dataset_name=$DATASET_NAME

	# 	$TIME python eval_classification.py $OUTFOLDER/ivector-fs_${hs}-hs_${fold}-f.predict $METAFILE > $OUTFOLDER/ivector-fs_${hs}-hs_${fold}-f.eval
	# 	rm $SCRATCH_FOLDER/*
	# done

done

python headsup.py `hostname`:$OUTFOLDER app.txt


