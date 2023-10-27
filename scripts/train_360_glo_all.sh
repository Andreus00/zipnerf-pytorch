#!/bin/bash

# outdoor
EXPERIMENT_PREFIX=360_v2_glo
SCENE=("bicycle" "garden" "stump" )
DATA_ROOT=/SSD_DISK/datasets/360_v2

len=${#SCENE[@]}
for((i=0; i<$len; i++ ))
do
  EXPERIMENT=$EXPERIMENT_PREFIX/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"

  python3 train.py \
    --gin_configs=configs/360_glo.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
      --gin_bindings="Config.factor = 4"

  python3 eval.py \
  --gin_configs=configs/360_glo.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 4"

  python3 render.py \
  --gin_configs=configs/360_glo.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 120" \
  --gin_bindings="Config.render_video_fps = 30" \
  --gin_bindings="Config.factor = 4"

  python3 extract.py \
  --gin_configs=configs/360_glo.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 4" \
  --gin_bindings="Config.vertex_projection = True"
done

# indoor "Config.factor = 2"
SCENE=("room" "counter" "kitchen" "bonsai")
len=${#SCENE[@]}
for((i=0; i<$len; i++ ))
do
  EXPERIMENT=$EXPERIMENT_PREFIX/"${SCENE[i]}"
  DATA_DIR="$DATA_ROOT"/"${SCENE[i]}"

  python3 train.py \
    --gin_configs=configs/360_glo.gin \
    --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
    --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
      --gin_bindings="Config.factor = 2"

  python3 eval.py \
  --gin_configs=configs/360_glo.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 2"

  python3 render.py \
  --gin_configs=configs/360_glo.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.render_path = True" \
  --gin_bindings="Config.render_path_frames = 120" \
  --gin_bindings="Config.render_video_fps = 30" \
  --gin_bindings="Config.factor = 2"

  python3 extract.py \
  --gin_configs=configs/360_glo.gin \
  --gin_bindings="Config.data_dir = '${DATA_DIR}'" \
  --gin_bindings="Config.exp_name = '${EXPERIMENT}'" \
  --gin_bindings="Config.factor = 2" \
  --gin_bindings="Config.vertex_projection = True"
done