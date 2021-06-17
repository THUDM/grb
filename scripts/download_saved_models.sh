#!/bin/bash

GRB_HOME=$(cd "$(dirname "$0")/.."; pwd)
MODEL_DIR="$GRB_HOME/saved_models"

# Download GRB saved models

printf "Downloading GRB saved models......\n"
if [ ! -x "$MODEL_DIR" ]; then
  mkdir $MODEL_DIR
  wget https://cloud.tsinghua.edu.cn/f/53cf0626b73148859e28/?dl=1 -O ./models.zip
  unzip models.zip -d $MODEL_DIR
  rm models.zip
  printf "Models downloaded to %s.\n" $MODEL_DIR
else
  printf "Directory: %s already exists.\n" $MODEL_DIR
fi
