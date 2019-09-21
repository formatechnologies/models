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

# Update PYTHONPATH.
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd deeplab

# ========================== SETTINGS (WORKSTATION) ==========================

# # Dennis Workstation Settings
# export TF_FORCE_GPU_ALLOW_GROWTH=true   # Workaround cuDNN bug with RTX GPUS
# NUM_CLONES=1
# TRAIN_BATCH_SIZE=1
# FINE_TUNE_BATCH_NORM=false

# GPU 1 + GPU 2 Workstation Settings
# NUM_CLONES=8
# TRAIN_BATCH_SIZE=8
# FINE_TUNE_BATCH_NORM=false

NUM_CLONES=4  # Don't use 8, draws too much power
TRAIN_BATCH_SIZE=16
FINE_TUNE_BATCH_NORM=true

# ========================== SETTINGS (DATASET) ==========================
# http://hellodfan.com/2018/07/06/DeepLabv3-with-own-dataset/

DATASET_NAME="imaterialist37k"
DEEPLAB_NAME="forma_37k"
NUM_CLASSES=9
EVAL_CROP_SIZE="1601,3783"
DATASET_TRAIN_SIZE=29606
NUM_EPOCHS=20
NUM_EXAMPLES=`expr $DATASET_TRAIN_SIZE \* $NUM_EPOCHS`
NUM_ITERATIONS=`expr $NUM_EXAMPLES / $TRAIN_BATCH_SIZE`

SAVE_INTERVAL_SECS=1200
SAVE_SUMMARIES_SECS=600

OUTPUT_STRIDE=16
ATROUS_RATE_1=6
ATROUS_RATE_2=12
ATROUS_RATE_3=18

# OUTPUT_STRIDE=8
# ATROUS_RATE_1=12
# ATROUS_RATE_2=24
# ATROUS_RATE_3=36

EVAL_OUTPUT_STRIDE=8
MODEL_VARIANT="xception_65"
DECODER_OUTPUT_STRIDE=4

BASE_LEARNING_RATE=0.007
# BASE_LEARNING_RATE=0.001
# BASE_LEARNING_RATE=0.0001

# ========================== SETTINGS (PATHS) ==========================

# Deeplab Path
DEEPLAB_DIR="${HOME}/storage/shared/deeplab"
INIT_MODELS_DIR="${DEEPLAB_DIR}/init_models"
DATASETS_DIR="${DEEPLAB_DIR}/datasets"
EXPERIMENTS_DIR="${DEEPLAB_DIR}/experiments"

# Init Models Paths
# TF_INITIAL_CHECKPOINT="${WORK_DINIT_MODELS_DIRIR}/deeplabv3_pascal_train_aug/model.ckpt"
# INITIALIZE_LAST_LAYERS=false
# LAST_LAYERS_CONTAINS_LOGITS_ONLY=true

# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/imaterialist37k/model.ckpt-740000"
# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/imaterialist37k_augmented/model.ckpt-366824"
TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/imaterialist37k_augmented/model.ckpt-693543"
INITIALIZE_LAST_LAYERS=true
LAST_LAYERS_CONTAINS_LOGITS_ONLY=true # irrelevant

# Dataset Paths
DATASET_DIR="${DATASETS_DIR}/${DATASET_NAME}"
DATASET_TFRECORD="${DATASET_DIR}/tfrecord"
DATASET_SPLIT="${DATASET_DIR}/dataset_split"
mkdir -p "${DATASET_DIR}"
mkdir -p "${DATASET_TFRECORD}"
mkdir -p "${DATASET_SPLIT}"

DATASET_NAME2='humanparsing17k'
DATASET_DIR2="${DATASETS_DIR}/${DATASET_NAME2}"
DATASET_TFRECORD2="${DATASET_DIR2}/tfrecord"
DATASET_SPLIT2="${DATASET_DIR2}/dataset_split"
mkdir -p "${DATASET_DIR2}"
mkdir -p "${DATASET_TFRECORD2}"
mkdir -p "${DATASET_SPLIT2}"

# Experiment Paths
DATE=`date +"%Y-%m-%d_%H-%M-%S"`
HOSTNAME=`hostname`
USER=`whoami`
EXPERIMENT_DESCRIPTION="augmented_learn_rate_007_output_stride_${OUTPUT_STRIDE}_batch_size_${TRAIN_BATCH_SIZE}"

EXPERIMENT_NAME="${HOSTNAME}_${USER}_${DATASET_NAME}_${EXPERIMENT_DESCRIPTION}"
EXPERIMENT_FOLDER="${EXPERIMENTS_DIR}/${EXPERIMENT_NAME}"
TRAIN_LOGDIR="${EXPERIMENT_FOLDER}/train"
EVAL_LOGDIR="${EXPERIMENT_FOLDER}/eval"
VIS_LOGDIR="${EXPERIMENT_FOLDER}/vis"
EXPORT_DIR="${EXPERIMENT_FOLDER}/export"
mkdir -p "${EXPERIMENT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

# ========================== SETTINGS (END) ==========================

# TODO: make data input dir a TF flag
# Build form data (from ~/storage/shared/datasets/json/train_json/)
python3 ./datasets/build_forma_data.py \
    --output_dir="${DATASET_TFRECORD}" \
    --list_folder="${DATASET_SPLIT}"

# TODO: make data input dir a TF flag
# Build form data (from ~/storage/shared/datasets/HumanParsing-Dataset/humanparsing/)
python3 ./datasets/build_human_parsing_data.py \
    --output_dir="${DATASET_TFRECORD2}" \
    --list_folder="${DATASET_SPLIT2}"

# Train.
python3 ./train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant=${MODEL_VARIANT} \
  --atrous_rates=${ATROUS_RATE_1} \
  --atrous_rates=${ATROUS_RATE_2} \
  --atrous_rates=${ATROUS_RATE_3} \
  --output_stride=${OUTPUT_STRIDE} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE} \
  --train_crop_size="513,513" \
  --num_clones=${NUM_CLONES} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --base_learning_rate=${BASE_LEARNING_RATE} \
  --fine_tune_batch_norm=${FINE_TUNE_BATCH_NORM} \
  --tf_initial_checkpoint="${TF_INITIAL_CHECKPOINT}" \
  --initialize_last_layer=${INITIALIZE_LAST_LAYERS} \
  --last_layers_contain_logits_only=${LAST_LAYERS_CONTAINS_LOGITS_ONLY} \
  --dataset="${DEEPLAB_NAME}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET_TFRECORD}" \
  --save_interaval_secs="${SAVE_INTERVAL_SECS}" \
  --save_summary_secs = "${SAVE_SUMMARIES_SECS}" \
  --save_summaries_images=true

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
python3 ./eval.py \
  --logtostderr \
  --eval_split="trainval" \
  --model_variant=${MODEL_VARIANT} \
  --atrous_rates=${ATROUS_RATE_1} \
  --atrous_rates=${ATROUS_RATE_2} \
  --atrous_rates=${ATROUS_RATE_3} \
  --output_stride=${EVAL_OUTPUT_STRIDE} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE} \
  --eval_crop_size="${EVAL_CROP_SIZE}" \
  --dataset="${DEEPLAB_NAME}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${DATASET_TFRECORD}" \
  --max_number_of_evaluations=1

# Visualize the results.
python3 ./vis.py \
  --logtostderr \
  --vis_split="trainval" \
  --model_variant=${MODEL_VARIANT} \
  --atrous_rates=${ATROUS_RATE_1} \
  --atrous_rates=${ATROUS_RATE_2} \
  --atrous_rates=${ATROUS_RATE_3} \
  --output_stride=${EVAL_OUTPUT_STRIDE} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE} \
  --vis_crop_size="${EVAL_CROP_SIZE}" \
  --dataset="${DEEPLAB_NAME}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${DATASET_TFRECORD}" \
  --max_number_of_iterations=1 \
  --colormap_type="forma"

# Export the trained checkpoint.
python3 ./export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant=${MODEL_VARIANT} \
  --atrous_rates=${ATROUS_RATE_1} \
  --atrous_rates=${ATROUS_RATE_2} \
  --atrous_rates=${ATROUS_RATE_3} \
  --output_stride=${EVAL_OUTPUT_STRIDE} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE} \
  --num_classes="${NUM_CLASSES}" \
  --crop_size=1001 \
  --crop_size=1001 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
