#!/bin/bash
FSELECTORS=(kmeansc linspace)
FOLDS=(1 2 3 4 5 6 7 8 9 10)
N_FRAMES=(5 20 40)
N_FRAMES_SEL_P=1
FOLDSFOLDER=gtzan_folds_10fs
SCRATCH_FOLDER=`pwd`/scratch9/
METAFILE=gtzan_folds_10fs/gtzan_labels.txt
OUTFOLDER=res_gtzan_selframes_gtzan_rosa_svm-class_nxn-desc_1-sel_10fs-filter/
TIME=/usr/bin/time
EXPERIMENT_SCRIPT=exp_gtzan_selframes/exp_gtzan_selframes_norm_feats.py
PARAMETER_FILE=exp_gtzan_selframes/parameters_gtzan_feats.yaml
LABEL_DICT=gtzan_folds_10fs/gtzan_dict.py
IVECTOR=False
CLASSIFIER=attnet
SORT_CENTROIDS=asc
EXTRACTION_OPTIONS=

#To do the nx1 or 1xn experiment, change frame_selector.subset_size of the appropriate stage(train or test). 
	# set kmeansc_selector.sort_centroids_by_common to select the appropriate "one".
	# linspace + asc does not make sense, it would be the same as linspace + desc. (asc and desc does nothing to linspace)

echo "*** FSELECTOR $fs"
$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -extract $SCRATCH_FOLDER $METAFILE -ss general.label_dict_file=$LABEL_DICT

#for nf in ${N_FRAMES[@]}
#do
#	echo "### N_FRAMES $nf"
#	for fs in ${FSELECTORS[@]}
#	do
#		n_frames_tt=`echo "$nf*$N_FRAMES_SEL_P" | bc -l | awk '{print ($0-int($0)>0)?int($0)+1:int($0)}'`
#		echo "*** FSELECTOR $fs"

#		for fold in ${FOLDS[@]}
#		do
#			echo "!!! FOLD $fold"
#			$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_train.txt -ss exp_gtzan_selframes.frame_selector=$fs -si frame_selector.n_frames=$nf -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes.classifier=$CLASSIFIER -ss kmeansc_selector.sort_centroids_by_common=$SORT_CENTROIDS -si frame_selector.subset_size=$nf
#			$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -test $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_test.txt $OUTFOLDER/${nf}-nf_${fs}-fs_${fold}-f.predict -ss exp_gtzan_selframes.frame_selector=$fs -si frame_selector.n_frames=$nf -ss general.label_dict_file=$LABEL_DICT -ss kmeansc_selector.sort_centroids_by_common=$SORT_CENTROIDS -si frame_selector.subset_size=$nf
#			$TIME python eval_classification.py $OUTFOLDER/${nf}-nf_${fs}-fs_${fold}-f.predict $METAFILE > $OUTFOLDER/${nf}-nf_${fs}-fs_${fold}-f.eval
#		done
#	done
#done


echo "^^^ ivector"
$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -extract $SCRATCH_FOLDER $METAFILE general.label_dict_file=$LABEL_DICT -sb exp_gtzan_selframes_norm_feats.ivector=True
for fold in ${FOLDS[@]}
do
    echo "!!! FOLD $fold"
    $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_train.txt -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT -ss exp_gtzan_selframes.classifier=$CLASSIFIER $EXTRACT_EXTRA_PARAMS $TRAIN_EXTRA_PARAMS
    $TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE  -test $SCRATCH_FOLDER $FOLDSFOLDER/f${fold}_test.txt $OUTFOLDER/ivector-fs_${fold}-f.predict -ss exp_gtzan_selframes.frame_selector=linspace -si frame_selector.n_frames=1 -ss general.label_dict_file=$LABEL_DICT $EXTRACT_EXTRA_PARAMS
    $TIME python eval_classification.py $OUTFOLDER/ivector-fs_${fold}-f.predict $METAFILE > $OUTFOLDER/ivector-fs_${fold}-f.eval
done

python headsup.py `hostname`:$OUTFOLDER app.txt

