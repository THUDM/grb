#!/bin/bash

GRB_HOME=$(cd "$(dirname "$0")/.."; pwd)
DATA_DIR="$GRB_HOME/data"

# Download GRB dataset

printf "Downloading GRB datasets......\n"
if [ ! -x "$DATA_DIR" ]; then
  mkdir $DATA_DIR
  wget https://cloud.tsinghua.edu.cn/f/34d5a4cfc52c42d0ad95/?dl=1 -O ./datasets.zip
  unzip datasets.zip -d $DATA_DIR
  rm datasets.zip
  printf "GRB datasets downloaded to %s.\n" $DATA_DIR
else
  printf "Directory: %s already exists.\n" $DATA_DIR
fi
