#!/bin/bash
FSELECTORS=(kmeansc linspace)
FOLDS=(1 2 3 4 5 6 7 8 9 10)
N_FRAMES=(5 20 40) # 5) # 40)
N_FRAMES_SEL_P=1
FOLDSFOLDER=gtzan_folds_10fs
SCRATCH_FOLDER=`pwd`/scratch9/
METAFILE=gtzan_folds_10fs/gtzan_labels.txt
OUTFOLDER=res_mel_selframes_gtzan_mel-128_knn-class_nxn-desc_1-sel_10fs-filter/
TIME=/usr/bin/time
EXPERIMENT_SCRIPT=exp_gtzan_selframes/exp_gtzan_selframes_norm_feats.py
PARAMETER_FILE=exp_gtzan_selframes/parameters.yaml
LABEL_DICT=gtzan_folds_10fs/gtzan_dict.py
NON_LINEARITY=(none)
DELTAS=(True)
N_FEATURES=(128) #(26 51 75 100) # (8
NO_ANOVA=(True)
#MEL_BINS = 128 HARDCODED ABAIXO
CLASSIFIER=knn
SORT_CENTROIDS=desc

#$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -extract $SCRATCH_FOLDER $METAFILE -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes_norm_feats.feature_extractor=stft -sb stft_features.deltas=True -sb stft_features.delta_deltas=True -si stft_features.mel_bins=128

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
                
                #Does the frame selection experiment
#                for nf in ${N_FRAMES[@]}
#                do
#                    echo "### N_FRAMES $nf"
#               	    n_frames_tt=`echo "$nf*$N_FRAMES_SEL_P" | bc -l | awk '{print ($0-int($0)>0)?int($0)+1:int($0)}'`
#                    for fs in ${FSELECTORS[@]}
#                    do
#                        echo "*** FSELECTOR $fs"

#                        for fold in ${FOLDS[@]}
#                        do
#                            echo "!!! FOLD $fold"
#                            $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_train.txt -ss exp_gtzan_selframes.frame_selector=$fs -si frame_selector.n_frames=$nf -ss general.label_dict_file=$LABEL_DICT -sb svm_anova.no_anova=$anova -ss exp_gtzan_selframes.classifier=$CLASSIFIER -ss kmeansc_selector.sort_centroids_by_common=$SORT_CENTROIDS -si frame_selector.subset_size=$nf
#                            $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -test $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_test.txt $OUTFOLDER/${nf}-nf_${fs}-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.predict -ss exp_gtzan_selframes.frame_selector=$fs -si frame_selector.n_frames=$nf -ss general.label_dict_file=$LABEL_DICT -sb svm_anova.no_anova=$anova -ss kmeansc_selector.sort_centroids_by_common=$SORT_CENTROIDS -si frame_selector.subset_size=$nf
#                            $TIME python eval_classification.py $OUTFOLDER/${nf}-nf_${fs}-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.predict $METAFILE > $OUTFOLDER/${nf}-nf_${fs}-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.eval 
#                        done
#                    done
#                done

                echo "^^^ ivector"
                $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -extract $SCRATCH_FOLDER $METAFILE -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT exp_gtzan_selframes_norm_feats.ivector=True -ss exp_gtzan_selframes_norm_feats.feature_extractor=stft -sb stft_features.deltas=$delta -sb stft_features.delta_deltas=$delta -si stft_features.mel_bins=128

                for fold in ${FOLDS[@]}
                do
                    echo "!!! FOLD $fold"
                    $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_train.txt -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT -sb svm_anova.no_anova=$anova -ss exp_gtzan_selframes.classifier=$CLASSIFIER
                    $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -test $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_test.txt $OUTFOLDER/ivector-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.predict -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT -sb svm_anova.no_anova=$anova
                    $TIME python eval_classification.py $OUTFOLDER/ivector-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.predict $METAFILE > $OUTFOLDER/ivector-fs_${n_feats}-nfeats_${nl}-nl_${anova}-noanova_${delta}-delta_${fold}-f.eval
                done

            done
        done
    done
done

python headsup.py `hostname`:$OUTFOLDER app.txt


