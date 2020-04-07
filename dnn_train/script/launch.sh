#!/bin/sh


# Usage:
# $: bash launch.sh <model-name> <checkpoints-path>
# Ex: bash launch.sh test-model /home/asr/prj_SlotTagger/dnn_train/checkpoints/


# The name of the ambiguity must be its folder name
#AMBIGUITY=è_e #36
AMBIGUITY=ha_a # v48
#AMBIGUITY=e_i # v6
#AMBIGUITY=hai_ai # v12

#AMBIGUITY=ho_o #v33
#AMBIGUITY=sei_6
#AMBIGUITY=sarà_sara #v9
#AMBIGUITY=hanno_anno #v22
#AMBIGUITY=ho_o #v28
#AMBIGUITY=dove_dovè  #v1

# Choose if activate pos-tagging layer
# 0 = False (NOT ACTIVE), 1 = True (ACTIVE)
POS_TAG=0


mkdir -p ../checkpoints

echo $'\n-----------------------------------------------------\n'
echo $'                  AMBIGUITY: '$AMBIGUITY
echo $'\n-----------------------------------------------------\n'

MODEL_NAME=$1
CHECKPOINTS_PATH=$2

ROOT='../'

if [ $POS_TAG = 0 ]; then
  echo $'\n-----------------------------------------------------\n'
  echo $'                  POS-TAG LAYER: NOT ACTIVE'
  echo $'\n-----------------------------------------------------\n'
  TRAIN_DATA_PATH=$ROOT'datasets/'$AMBIGUITY'/train_data_v48.tsv'
  VALID_DATA_PATH=$ROOT'datasets/'$AMBIGUITY'/valid_data_v48.tsv'
  VOCAB_PATH=$ROOT'datasets/'$AMBIGUITY'/vocab_15k_v48.tsv'
  NUM_WORDS_PATH=$ROOT'datasets/'$AMBIGUITY'/n_words_15k_v48.tsv'
else
  echo $'\n-----------------------------------------------------\n'
  echo $'                  POS-TAG LAYER: ACTIVE'
  echo $'\n-----------------------------------------------------\n'
  TRAIN_DATA_PATH=$ROOT'datasets/'$AMBIGUITY'/train_data_nph_400k.tsv'
  VALID_DATA_PATH=$ROOT'datasets/'$AMBIGUITY'/valid_data_nph_400k.tsv'
  VOCAB_PATH=$ROOT'datasets/'$AMBIGUITY'/vocab_5k_nph_400k.tsv'
  NUM_WORDS_PATH=$ROOT'datasets/'$AMBIGUITY'/n_words_5k_nph_400k.tsv'
fi


TRAINING_SIZE=$(cat $TRAIN_DATA_PATH | wc -l )
VALIDATION_SIZE=$(cat $VALID_DATA_PATH | wc -l )

echo $'\n-----------------------------------------------------\n'
echo $'                  TRAINING STARTING                    \n'
echo 'Taraining datset: ' $TRAIN_DATA_PATH
echo $'                                                       \n'
echo 'Taraining size: ' $TRAINING_SIZE
echo 'Validation size: ' $VALIDATION_SIZE

# for num in '1 2 4 8 16 32';
# do
# for num in 1 2 4 8 16 32;
# do
# for lrate in 0.05;# 0.005 0.001 0.0005 0.0001;
# do
#LRATE=$(($num * 8))
python3 main.py ${MODEL_NAME} ${CHECKPOINTS_PATH} ${TRAIN_DATA_PATH} ${VALID_DATA_PATH} ${VOCAB_PATH} ${NUM_WORDS_PATH} $TRAINING_SIZE ${AMBIGUITY} ${POS_TAG}
#done

echo $'\n-----------------------------------------------------\n'
echo $'                  TRAINING FINISHED                    \n'

echo 'Model saved in: ' $CHECKPOINTS_PATH$MODEL_NAME
echo $'                                                       '
echo $'\n-----------------------------------------------------\n'
