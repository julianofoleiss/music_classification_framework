#!/bin/bash
DATASET_NAME=lmd10s
FSELECTORS=(linspace kmeansc)
FOLDS=(1 2 3)
N_FRAMES=(5 20 40)
N_FRAMES_SEL_P=1
FOLDSFOLDER=lmd_folds_10s/
SCRATCH_FOLDER=`pwd`/scratch3/
METAFILE=lmd_folds_10s/lmd_labels.txt
OUTFOLDER=res_random_selframes_lmd10s_svm-class_nxn-desc_1-sel_linspace-downsampler_5s25s-dsettings_linspace,kmeansc-fs/
TIME=/usr/bin/time
EXPERIMENT_SCRIPT=exp_gtzan_selframes/exp_gtzan_selframes_norm_feats.py
PARAMETER_FILE=exp_gtzan_selframes/parameters.yaml
LABEL_DICT=lmd_folds_10s/lmd_dict.py
NON_LINEARITY=(none)
DELTAS=(True)
N_FEATURES=(8 26 51 75 100) #(5 8 16 32 51 75 100) #(8 26 51 75 100) #(26 51 75 100) # (8
NO_ANOVA=(True)
#PROJ_MATRIX_PATH=exp_gtzan_selframes/random_proj_128-mel_ortho/proj_matrix_FEAT-f_gauss-tm.dill
PROJ_MATRIX_PATH=exp_gtzan_selframes/random_proj_128-mel/proj_matrix_FEAT-f_gauss-tm.dill
#MEL_BINS = 128 HARDCODED ABAIXO
CLASSIFIER=svm_anova #svm_anova
#EXTRACT_EXTRA_PARAMS="-si exp_gtzan_selframes.extraction_workers=3"
EXTRACT_EXTRA_PARAMS=
TRAIN_EXTRA_PARAMS=
SORT_CENTROIDS=asc
RANDOM_SEL_SEED=0

#To do the nx1 or 1xn experiment, change frame_selector.subset_size of the appropriate stage(train or test). 
	# set kmeansc_selector.sort_centroids_by_common to select the appropriate "one".
	# linspace + asc does not make sense, it would be the same as linspace + desc. (asc and desc does nothing to linspace)
export OPENBLAS_NUM_THREADS=1

for delta in ${DELTAS[@]}
do
    echo "@@@ DELTAS $delta"
    for anova in ${NO_ANOVA[@]}
    do
        echo "??? NO_ANOVA $anova"
        for nl in ${NON_LINEARITY[@]}
        do
            echo "&&& NON_LINEARITY $nl"
            for n_feats in ${N_FEATURES[@]}
            do
                echo "%%% N_FEATURES $n_feats"

            rm $SCRATCH_FOLDER/*h5

            $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -extract $SCRATCH_FOLDER $METAFILE -ss general.label_dict_file=$LABEL_DICT -sb random_projection.deltas=$delta -sb random_projection.delta_deltas=$delta -ss exp_gtzan_selframes_norm_feats.rp_projection_matrix=${PROJ_MATRIX_PATH/FEAT/$n_feats} -ss random_projection.non_linearity=$nl -si random_projection.mel_bins=128 $EXTRACT_EXTRA_PARAMS -ss general.dataset_name=$DATASET_NAME           

                #Does the frame selection experiment
                for nf in ${N_FRAMES[@]}
                do
                    echo "### N_FRAMES $nf"
                    for fs in ${FSELECTORS[@]}
                    do
						n_frames_tt=`echo "$nf*$N_FRAMES_SEL_P" | bc -l | awk '{print ($0-int($0)>0)?int($0)+1:int($0)}'`
                        echo "*** FSELECTOR $fs"

                        for fold in ${FOLDS[@]}
                        do
                            echo "!!! FOLD $fold"
                            $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_train.txt -ss exp_gtzan_selframes.frame_selector=$fs -si frame_selector.n_frames=$nf -ss general.label_dict_file=$LABEL_DICT -sb svm_anova.no_anova=$anova -sb knn.no_anova=$anova -ss exp_gtzan_selframes.classifier=$CLASSIFIER $EXTRACT_EXTRA_PARAMS $TRAIN_EXTRA_PARAMS -ss kmeansc_selector.sort_centroids_by_common=$SORT_CENTROIDS -si frame_selector.subset_size=$nf -ss general.dataset_name=$DATASET_NAME -si random_selector.seed=$RANDOM_SEL_SEED
                            $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -test $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_test.txt $OUTFOLDER/${nf}-nf_${fs}-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.predict -ss exp_gtzan_selframes.frame_selector=$fs -si frame_selector.n_frames=$nf -ss general.label_dict_file=$LABEL_DICT -sb svm_anova.no_anova=$anova -sb knn.no_anova=$anova $EXTRACT_EXTRA_PARAMS -ss kmeansc_selector.sort_centroids_by_common=$SORT_CENTROIDS -si frame_selector.subset_size=$nf -ss general.dataset_name=$DATASET_NAME -si random_selector.seed=$RANDOM_SEL_SEED
                            $TIME python eval_classification.py $OUTFOLDER/${nf}-nf_${fs}-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.predict $METAFILE > $OUTFOLDER/${nf}-nf_${fs}-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.eval
                            rm $SCRATCH_FOLDER/train*
                            rm $SCRATCH_FOLDER/test*
                        done
                    done
                done

                echo "^^^ ivector"
                $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -extract $SCRATCH_FOLDER $METAFILE -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT -sb random_projection.deltas=$delta -sb random_projection.delta_deltas=$delta -ss exp_gtzan_selframes_norm_feats.rp_projection_matrix=${PROJ_MATRIX_PATH/FEAT/$n_feats} -ss random_projection.non_linearity=$nl -sb exp_gtzan_selframes_norm_feats.ivector=True -si random_projection.mel_bins=128  $EXTRACT_EXTRA_PARAMS -ss general.dataset_name=$DATASET_NAME -sb exp_gtzan_selframes_norm_feats.linspace_downsampling=False

                for fold in ${FOLDS[@]}
                do
                    echo "!!! FOLD $fold"
                    $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_train.txt -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT -sb svm_anova.no_anova=$anova -sb knn.no_anova=$anova -ss exp_gtzan_selframes.classifier=$CLASSIFIER $EXTRACT_EXTRA_PARAMS $TRAIN_EXTRA_PARAMS -ss general.dataset_name=$DATASET_NAME
                    $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -test $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_test.txt $OUTFOLDER/ivector-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.predict -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT -sb svm_anova.no_anova=$anova -sb knn.no_anova=$anova $EXTRACT_EXTRA_PARAMS -ss general.dataset_name=$DATASET_NAME
                    $TIME python eval_classification.py $OUTFOLDER/ivector-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.predict $METAFILE > $OUTFOLDER/ivector-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.eval
                done

            done
        done
    done
done

python headsup.py `hostname`:$OUTFOLDER app.txt


