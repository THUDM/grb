#!/bin/bash

GRB_HOME=$(
  cd "$(dirname "$0")/.."
  pwd
)

unset DATASET SAVE_DIR
DATA_DIR="$GRB_HOME/data"
MODEL_DIR="$GRB_HOME/saved_models"
RESULT_DIR="$GRB_HOME/results"
ATTACK_NUM=0
GPU=-1

usage() {
  echo "Usage: $0 [-d <string>] [-g <int>] [-s <string>] [-n <int>]"
  echo "Pipeline for reproducing leaderboard on the chosen dataset."
  echo "    -h      Display help message."
  echo "    -d      Choose a dataset."
  echo "    -s      Set a directory to save leaderboard files."
  echo "    -n      Choose the number of an attack from 0 to 9."
  echo "    -g      Choose a GPU device. -1 for CPU."
  exit 1
}

while getopts "srd:d:g:s:?h" o; do
  case "$o" in
  d)
    DATASET=$OPTARG
    ;;
  g)
    GPU=$OPTARG
    ;;
  s)
    SAVE_DIR=$OPTARG
    ;;
  h | ?)
    usage
    ;;
  esac
done

python $GRB_HOME/pipeline/leaderboard_pipeline.py \
  --dataset $DATASET \
  --feat_norm arctan \
  --data_dir $DATA_DIR/$DATASET \
  --model_dir $MODEL_DIR/$DATASET/ \
  --model_file 0/checkpoint.pt \
  --config_dir $GRB_HOME/pipeline/$DATASET/ \
  --attack_dir $RESULT_DIR/$DATASET/ \
  --attack_adj_name $ATTACK_NUM/adj.pkl \
  --attack_feat_name $ATTACK_NUM/features.npy \
  --save_dir $SAVE_DIR/$DATASET/ \
  --weight_type polynomial \
  --gpu $GPU