#!/bin/bash
HIDDEN_SIZE=(16 32 64 128 256) #(64 128 256 512)
#HIDDEN_SIZE=(128 256 512 1024 2048 3000)
FOLDS=(1 2 3 4 5 6 7 8 9 10)
FOLDSFOLDER=gtzan_folds_10fs
SCRATCH_FOLDER=`pwd`/scratch/
METAFILE=gtzan_folds_10fs/gtzan44_labels.txt
OUTFOLDER=ae_features_gtzan_128-mel_10fs-filter/
TIME=/usr/bin/time
EXPERIMENT_SCRIPT=scripts/train_autoencoder.py
PARAMETER_FILE=scripts/train_autoencoder.yaml
LABEL_DICT=gtzan_folds_10fs/gtzan_dict.py

rm -rf stfts/

for hs in ${HIDDEN_SIZE[@]}
do
	echo "!!! HIDDEN_SIZE $hs"
	for fold in ${FOLDS[@]}
	do
		echo "!!! FOLD $fold"
		$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -train $FOLDSFOLDER/f${fold}_train.txt $OUTFOLDER/f${fold}_model_${hs}-feats.dill -ss train_autoencoder.label_dict_filename=$LABEL_DICT -si train_autoencoder.hidden_layer_sizes=$hs -ss train_autoencoder.standard_scaler_filename=$OUTFOLDER/f${fold}_scaler_${hs}-feats.dill -ss train_autoencoder.standard_scaler_mode=run
		$TIME python mirex_sub.py $EXPERIMENT_SCRIPT $PARAMETER_FILE -extract $METAFILE $OUTFOLDER/f${fold}_model_${hs}-feats.dill $LABEL_DICT $OUTFOLDER/f${fold}_features_${hs}-feats.dill -ss train_autoencoder.standard_scaler_filename=$OUTFOLDER/f${fold}_scaler_${hs}-feats.dill
	done
done
