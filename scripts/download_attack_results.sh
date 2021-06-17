#!/bin/bash

GRB_HOME=$(cd "$(dirname "$0")/.."; pwd)
RESULT_DIR="$GRB_HOME/results"

# Download attack results

printf "Downloading attack results......\n"
if [ ! -x "$RESULT_DIR" ]; then
  mkdir $RESULT_DIR
  wget https://cloud.tsinghua.edu.cn/f/e1e68fe24f324866a475/?dl=1 -O ./results.zip
  unzip results.zip -d $RESULT_DIR
  rm results.zip
  printf "Attack results downloaded to %s.\n" $RESULT_DIR
else
  printf "Directory: %s already exists.\n" $RESULT_DIR
fi
