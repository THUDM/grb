#!/bin/bash

GRB_HOME=$(
  cd "$(dirname "$0")/.."
  pwd
)
unset DATASET SAVE_DIR
GPU=-1

usage() {
  echo "Usage: $0 [-d <string>] [-g <int>] [-s <string>]"
  echo "Pipeline for training GNNs on the chosen dataset."
  echo "    -h      Display help message."
  echo "    -d      Choose a dataset."
  echo "    -s      Set a directory to save models."
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

if [ -n "$DATASET" ]; then
  case $DATASET in
  grb-cora)
    python $GRB_HOME/pipeline/train_pipeline.py \
      --n_epoch 8000 \
      --dataset $DATASET \
      --feat_norm arctan \
      --data_dir $GRB_HOME/data/$DATASET \
      --model_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
      --config_dir $GRB_HOME/pipeline/$DATASET/ \
      --dropout 0.5 \
      --eval_every 1 \
      --save_after 0 \
      --early_stop \
      --n_train 1 \
      --train_mode inductive \
      --gpu $GPU
    ;;
  grb-citeseer)
    python $GRB_HOME/pipeline/train_pipeline.py \
      --n_epoch 8000 \
      --dataset $DATASET \
      --feat_norm arctan \
      --data_dir $GRB_HOME/data/$DATASET \
      --model_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
      --config_dir $GRB_HOME/pipeline/$DATASET/ \
      --dropout 0.5 \
      --eval_every 1 \
      --save_after 0 \
      --early_stop \
      --n_train 1 \
      --train_mode inductive \
      --gpu $GPU
    ;;
  grb-flickr)
    python $GRB_HOME/pipeline/train_pipeline.py \
      --n_epoch 12000 \
      --dataset $DATASET \
      --feat_norm arctan \
      --data_dir $GRB_HOME/data/$DATASET \
      --model_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
      --config_dir $GRB_HOME/pipeline/$DATASET/ \
      --dropout 0.5 \
      --eval_every 5 \
      --save_after 0 \
      --early_stop \
      --n_train 1 \
      --train_mode inductive \
      --gpu $GPU
    ;;
  grb-reddit)
    python $GRB_HOME/pipeline/train_pipeline.py \
      --n_epoch 12000 \
      --dataset $DATASET \
      --feat_norm arctan \
      --data_dir $GRB_HOME/data/$DATASET \
      --model_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
      --config_dir $GRB_HOME/pipeline/$DATASET/ \
      --dropout 0.5 \
      --eval_every 10 \
      --save_after 100 \
      --early_stop \
      --n_train 1 \
      --train_mode inductive \
      --gpu $GPU
    ;;
  grb-aminer)
    python $GRB_HOME/pipeline/train_pipeline.py \
      --n_epoch 2000 \
      --lr 0.01 \
      --dataset_mode easy \
      --dataset $DATASET \
      --feat_norm arctan \
      --data_dir $GRB_HOME/data/$DATASET \
      --model_dir $GRB_HOME/$SAVE_DIR/$DATASET/ \
      --config_dir $GRB_HOME/pipeline/$DATASET/ \
      --dropout 0.5 \
      --eval_every 10 \
      --save_after 100 \
      --early_stop \
      --n_train 1 \
      --train_mode inductive \
      --gpu $GPU
    ;;
  esac
fi
