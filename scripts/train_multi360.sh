#!/bin/bash

SCENE=bicycle
EXPERIMENT=360_v2_multiscale/"$SCENE"
DATA_ROOT=/SSD_DISK/datasets/360_v2
DATA_DIR="$DATA_ROOT"/"$SCENE"

rm exp/"$EXPERIMENT"/*
python3 train.py --gin_configs=configs/multi360.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 4"
