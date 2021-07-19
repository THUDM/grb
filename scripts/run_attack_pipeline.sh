#!/bin/bash

GRB_HOME=$(
  cd "$(dirname "$0")/.."
  pwd
)
unset DATASET MODEL_DIR SAVE_DIR
GPU=-1
DATASET_MODE="full"
MODEL_TYPE="gcn"

usage() {
  echo "Usage: $0 [-d <string>] [-g <int>] [-l <string>] [-m <string>] [-t <string>] [-s <string>]"
  echo "Pipeline for attacking GNNs on the chosen dataset."
  echo "    -h      Display help message."
  echo "    -d      Choose a dataset."
  echo "    -l      Choose a dataset mode."
  echo "    -m      The directory of the saved surrogate model."
  echo "    -t      The type of the surrogate (attacked) model."
  echo "    -s      Set a directory to save attack results."
  echo "    -g      Choose a GPU device. Default: -1 for CPU."
  exit 1
}

while getopts "srd:d:g:l:m:t:s:?h" o; do
  case "$o" in
  d)
    DATASET=$OPTARG
    ;;
  g)
    GPU=$OPTARG
    ;;
  l)
    DATASET_MODE=$OPTARG
    ;;
  m)
    MODEL_DIR=$OPTARG
    ;;
  t)
    MODEL_TYPE=$OPTARG
    ;;
  s)
    SAVE_DIR=$OPTARG
    ;;
  h | ?)
    usage
    ;;
  esac
done

if [ "$DATASET_MODE" = "full" ]; then
    if [ -n "$DATASET" ]; then
      case $DATASET in
      grb-cora)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 60 \
          --n_edge_max 20 \
          --early_stop \
          --gpu $GPU
        ;;
      grb-citeseer)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 90 \
          --n_edge_max 20 \
          --early_stop \
          --gpu $GPU
        ;;
      grb-flickr)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 600 \
          --n_edge_max 100 \
          --early_stop \
          --gpu $GPU
        ;;
      grb-reddit)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 1500 \
          --n_edge_max 200 \
          --early_stop \
          --gpu $GPU
        ;;
      grb-aminer)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 1500 \
          --n_edge_max 100 \
          --early_stop \
          --gpu $GPU
        ;;
      esac
    fi
else
    if [ -n "$DATASET" ]; then
      case $DATASET in
      grb-cora)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 20 \
          --n_edge_max 20 \
          --early_stop \
          --gpu $GPU
        ;;
      grb-citeseer)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 20 \
          --n_edge_max 20 \
          --early_stop \
          --gpu $GPU
        ;;
      grb-flickr)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 30 \
          --n_edge_max 20 \
          --early_stop \
          --gpu $GPU
        ;;
      grb-reddit)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 500 \
          --n_edge_max 200 \
          --early_stop \
          --gpu $GPU
        ;;
      grb-aminer)
        python $GRB_HOME/pipeline/attack_pipeline.py \
          --n_epoch 2000 \
          --dataset $DATASET \
          --feat_norm arctan \
          --dataset_mode $DATASET_MODE \
          --data_dir $GRB_HOME/data/$DATASET \
          --model $MODEL_TYPE \
          --model_dir $GRB_HOME/$MODEL_DIR/$DATASET/ \
          --config_dir $GRB_HOME/pipeline/$DATASET/ \
          --save_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
          --n_attack 1 \
          --n_inject 500 \
          --n_edge_max 100 \
          --early_stop \
          --gpu $GPU
        ;;
      esac
    fi
fi