#!/bin/bash
CENTERK=-1
FOLDS=(1)
DATASET=ismir_10s

SCRATCH_DIR=scratch_all_knn_3/

HS16=(16 32)
HS64=(64 128 256)
AE16_PATH=ae_features_ismir10s_128-mel/
AE64_PATH=ae_features_ismir10s_128-mel/

RP_TARGETDIMS=(8 26 51 75 100)
RP_PATH_PATTERN="/home/DSPCOM/juliano/dados/juliano/music_classification_framework/exp_gtzan_selframes/random_proj_128-mel/proj_matrix_%d-f_gauss-tm.dill"

export OPENBLAS_NUM_THREADS=1

##EXPERIMENTO MARSYAS
#echo "*** EXPERIMENT MARSYAS"
##python baseline_allframes_knn.py --extract-feats --centerk=$CENTERK --feature-set=marsyas --dataset=$DATASET 1 $SCRATCH_DIR
#python baseline_allframes_knn.py --extract-only --extract-feats --centerk=$CENTERK --feature-set=marsyas --dataset=$DATASET 1 $SCRATCH_DIR
##FOLDS=(3)
#for fold in ${FOLDS[@]}
#do
    #python baseline_allframes_knn.py --feature-set=marsyas --dataset=$DATASET $fold $SCRATCH_DIR
#done

#rm $SCRATCH_DIR/*

## #EXPERIMENTO MEL
#echo "*** EXPERIMENT MEL"
##python baseline_allframes_knn.py --extract-feats --centerk=$CENTERK --feature-set=mel --dataset=$DATASET 1 $SCRATCH_DIR
#python baseline_allframes_knn.py --extract-only --extract-feats --centerk=$CENTERK --feature-set=mel --dataset=$DATASET 1 $SCRATCH_DIR
##FOLDS=(3)
#for fold in ${FOLDS[@]}
#do
    #python baseline_allframes_knn.py --feature-set=mel --dataset=$DATASET $fold $SCRATCH_DIR
#done

rm $SCRATCH_DIR/*

# FOLDS=(1)

 #EXPERIMENTO AE
 echo "*** EXPERIMENT AE"
 for hs in ${HS16[@]}
 do
 	echo "### HS $hs"
 	for fold in ${FOLDS[@]}
 	do
 		echo "%%% FOLD $fold"
 		python baseline_allframes_knn.py --extract-feats --centerk=$CENTERK --feature-set=ae --ae_hs=$hs --dataset=$DATASET --ae_path=$AE16_PATH $fold  $SCRATCH_DIR
 	done
 done

 rm $SCRATCH_DIR/*

 for hs in ${HS64[@]}
 do
 	echo "### HS $hs"
  	for fold in ${FOLDS[@]}
  	do
  		echo "%%% FOLD $fold"
  		python baseline_allframes_knn.py --extract-feats --centerk=$CENTERK --feature-set=ae --ae_hs=$hs --dataset=$DATASET --ae_path=$AE64_PATH $fold $SCRATCH_DIR 
  	done
  done


##EXPERIMENTO RP
#for rpt in ${RP_TARGETDIMS[@]}
#do
	#echo "### RPT $rpt"

	#python baseline_allframes_knn.py --extract-only --extract-feats --centerk=$CENTERK --feature-set=rp --rp-target=$rpt --rp-path-pattern=$RP_PATH_PATTERN --dataset=$DATASET 1 $SCRATCH_DIR
	
 	#for fold in ${FOLDS[@]}
 	#do
 		##echo "%%% FOLD $fold"
 		#python baseline_allframes_knn.py --centerk=$CENTERK --feature-set=rp --dataset=$DATASET $fold $SCRATCH_DIR 
 	#done
 #done
