#!/bin/sh


# Usage:
# $: bash launch.sh <model-name> <checkpoints-path>
# Ex: bash launch.sh test-sei /home/asr/prj_SlotTagger/dnn_train/checkpoints/


# The name of the ambiguity must be its folder name
AMBIGUITY=è_e

mkdir -p ../checkpoints

echo $'\n-----------------------------------------------------\n'
echo $'                  AMBIGUITY: '$AMBIGUITY
echo $'\n-----------------------------------------------------\n'

MODEL_NAME=$1
CHECKPOINTS_PATH=$2

ROOT='/home/asr/prj_SlotTagger/dnn_train/'
TRAIN_DATA_PATH=$ROOT'datasets/'$AMBIGUITY'/train_data_v5.tsv'
VALID_DATA_PATH=$ROOT'datasets/'$AMBIGUITY'/valid_data_v5.tsv'
VOCAB_PATH=$ROOT'datasets/'$AMBIGUITY'/vocab_10k_swin_v4.tsv'
NUM_WORDS_PATH=$ROOT'datasets/'$AMBIGUITY'/n_words_10k_swin_v4.tsv'
TRAINING_SIZE=$(cat $TRAIN_DATA_PATH | wc -l )

echo $'\n-----------------------------------------------------\n'
echo $'                  TRAINING STARTING                    \n'
echo 'Taraining datset: ' $TRAIN_DATA_PATH
echo $'                                                       \n'
echo 'Taraining size: ' $TRAINING_SIZE

# for num in '1 2 4 8 16 32';
# do
# for num in 1 2 4 8 16 32;
# do
# for lrate in 0.05;# 0.005 0.001 0.0005 0.0001;
# do
#LRATE=$(($num * 8))
python3 main.py ${MODEL_NAME} ${CHECKPOINTS_PATH} ${TRAIN_DATA_PATH} ${VALID_DATA_PATH} ${VOCAB_PATH} ${NUM_WORDS_PATH} $TRAINING_SIZE
#done

echo $'\n-----------------------------------------------------\n'
echo $'                  TRAINING FINISHED                    \n'

echo 'Model saved in: ' $CHECKPOINTS_PATH$MODEL_NAME
echo $'                                                       '
echo $'\n-----------------------------------------------------\n'
