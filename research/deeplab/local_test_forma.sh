#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Set up the working environment.
WORK_DIR=$(pwd)

# Update PYTHONPATH.
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd ${WORK_DIR}

# ========================== SETTINGS (DATASET) ==========================
# http://hellodfan.com/2018/07/06/DeepLabv3-with-own-dataset/

# DATASET_NAME="forma_1k"
# DATASET_SIZE=1000
# DATASET_SEG_ENCODING_TYPE='seg_name_to_label'
# NUM_CLASSES=9
# EVAL_CROP_SIZE="1601,3313"
# NUM_ITERATIONS=30000
# SAVE_INTERVAL_SECS=120
# SAVE_SUMMARIES_SECS=120

# DATASET_NAME="forma_1k_3"
# DATASET_SIZE=1000
# DATASET_SEG_ENCODING_TYPE='seg_name_to_label_3'
# NUM_CLASSES=3
# EVAL_CROP_SIZE="1601,3313"
# NUM_ITERATIONS=30000
# SAVE_INTERVAL_SECS=120
# SAVE_SUMMARIES_SECS=120

# DATASET_NAME="forma_1k_7"
# DATASET_SIZE=1000
# DATASET_SEG_ENCODING_TYPE='seg_name_to_label_7'
# NUM_CLASSES=7
# EVAL_CROP_SIZE="1601,3313"
# NUM_ITERATIONS=30000
# SAVE_INTERVAL_SECS=120
# SAVE_SUMMARIES_SECS=120

DATASET_NAME="forma_37k"
DATASET_SIZE=-1
DATASET_SEG_ENCODING_TYPE='seg_name_to_label'
NUM_CLASSES=9
EVAL_CROP_SIZE="1601,3783"
NUM_ITERATIONS=740000
SAVE_INTERVAL_SECS=1200
SAVE_SUMMARIES_SECS=600

# DATASET_NAME="forma_37k_3"
# DATASET_SIZE=-1
# DATASET_SEG_ENCODING_TYPE='seg_name_to_label_3'
# NUM_CLASSES=3
# EVAL_CROP_SIZE="1601,3783"
# NUM_ITERATIONS=740000
# SAVE_INTERVAL_SECS=1200
# SAVE_SUMMARIES_SECS=600

# DATASET_NAME="forma_37k_7"
# DATASET_SIZE=-1
# DATASET_SEG_ENCODING_TYPE='seg_name_to_label_7'
# NUM_CLASSES=7
# EVAL_CROP_SIZE="1601,3783"
# NUM_ITERATIONS=740000
# SAVE_INTERVAL_SECS=1200
# SAVE_SUMMARIES_SECS=600

# ========================== SETTINGS (WORKSTATION) ==========================

# Dennis Workstation Settings
export TF_FORCE_GPU_ALLOW_GROWTH=true   # Workaround cuDNN bug with RTX GPUS
TRAIN_BATCH_SIZE=1
FINE_TUNE_BATCH_NORM=false

# # GPU 1 + GPU 2 Workstation Settings
# TRAIN_BATCH_SIZE=12
# FINE_TUNE_BATCH_NORM=true

# ========================== SETTINGS (PATHS) ==========================

# Dataset Paths
DATASET_FOLDER="${HOME}/storage/shared/datasets/deeplab_experiments"
FORMA_DATASET_DIR="${DATASET_FOLDER}/${DATASET_NAME}"
FORMA_DATASET_DATA="${FORMA_DATASET_DIR}/tfrecord"
FORMA_DATASET_LIST="${FORMA_DATASET_DIR}/dataset_split"
mkdir -p "${FORMA_DATASET_DIR}"
mkdir -p "${FORMA_DATASET_DATA}"
mkdir -p "${FORMA_DATASET_LIST}"

# Experiment Paths
EXP_FOLDER="exp/train_on_train_set"
INIT_FOLDER="${FORMA_DATASET_DIR}/init_models"
TRAIN_LOGDIR="${FORMA_DATASET_DIR}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${FORMA_DATASET_DIR}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${FORMA_DATASET_DIR}/${EXP_FOLDER}/vis"
EXPORT_DIR="${FORMA_DATASET_DIR}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

TF_INITIAL_CHECKPOINT="${DATASET_FOLDER}/forma_37k_ckpt/model.ckpt-740000"
INITIALIZE_LAST_LAYERS=true
LAST_LAYERS_CONTAINS_LOGITS_ONLY=true # irrelevant

# TF_INITIAL_CHECKPOINT="${WORK_DIR}/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt"
# INITIALIZE_LAST_LAYERS=false
# LAST_LAYERS_CONTAINS_LOGITS_ONLY=true

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

# ========================== SETTINGS (END) ==========================
# TODO: make data input dir a TF flag
# Build form data (from ~/storage/shared/datasets/json/train_json/)
python3 ./datasets/build_forma_data.py \
    --dataset_size=${DATASET_SIZE} \
    --dataset_seg_encoding_type="${DATASET_SEG_ENCODING_TYPE}" \
    --output_dir="${FORMA_DATASET_DATA}" \
    --list_folder="${FORMA_DATASET_LIST}"

# Train.
python3 "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="513,513" \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=${FINE_TUNE_BATCH_NORM} \
  --tf_initial_checkpoint="${TF_INITIAL_CHECKPOINT}" \
  --initialize_last_layer=${INITIALIZE_LAST_LAYERS} \
  --last_layers_contain_logits_only=${LAST_LAYERS_CONTAINS_LOGITS_ONLY} \
  --dataset="${DATASET_NAME}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${FORMA_DATASET_DATA}" \
  --save_interaval_secs="${SAVE_INTERVAL_SECS}" \
  --save_summary_secs = "${SAVE_SUMMARIES_SECS}" \
  --save_summaries_images=true

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
python3 "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="trainval" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size="${EVAL_CROP_SIZE}" \
  --dataset="${DATASET_NAME}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${FORMA_DATASET_DATA}" \
  --max_number_of_evaluations=1

# Visualize the results.
python3 "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="trainval" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size="${EVAL_CROP_SIZE}" \
  --dataset="${DATASET_NAME}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${FORMA_DATASET_DATA}" \
  --max_number_of_iterations=1 \
  --colormap_type="forma"

# Export the trained checkpoint.
python3 "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes="${NUM_CLASSES}" \
  --crop_size=1001 \
  --crop_size=1001 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
