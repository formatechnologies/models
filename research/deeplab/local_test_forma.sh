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

# Deeplab Path
DEEPLAB_DIR="${HOME}/storage/shared/deeplab"
INIT_MODELS_DIR="${DEEPLAB_DIR}/init_models"
DATASETS_DIR="${DEEPLAB_DIR}/datasets"
EXPERIMENTS_DIR="${DEEPLAB_DIR}/experiments"

# ========================== SELECT DATASET ==========================
# DATASET_NAME="imaterialist37k"
# DATASET_EVAL_CROP_SIZE="1601,3783" # Max image dimensions + 1
# DATASET_TRAIN_SIZE=29606

# DATASET_NAME='humanparsing17k'
# DATASET_EVAL_CROP_SIZE="1601,1137" # Max image dimensions + 1
# DATASET_TRAIN_SIZE=14164

# DATASET_NAME='nsfw1k'
# DATASET_EVAL_CROP_SIZE="4001,4001" # Max image dimensions + 1
# DATASET_TRAIN_SIZE=1000

# DATASET_NAME='forma54k'
# DATASET_EVAL_CROP_SIZE="1601,3783" # Max image dimensions + 1
# DATASET_TRAIN_SIZE=43770

DATASET_NAME='forma55k'
DATASET_EVAL_CROP_SIZE="4001,4001" # Max image dimensions + 1
DATASET_TRAIN_SIZE=44770

DATASET_DIR="${DATASETS_DIR}/${DATASET_NAME}"
DATASET_TFRECORD="${DATASET_DIR}/tfrecord"
DATASET_SPLIT="${DATASET_DIR}/dataset_split"

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

NUM_CLONES=8
TRAIN_BATCH_SIZE=32
# TRAIN_BATCH_SIZE=8
FINE_TUNE_BATCH_NORM=false

# ========================== SETTINGS (INITIALIZATION) ==========================
# Init Models Paths
# TF_INITIAL_CHECKPOINT_NAME="pascal"
# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/deeplabv3_pascal_train_aug/model.ckpt"
# INITIALIZE_LAST_LAYERS=false
# LAST_LAYERS_CONTAINS_LOGITS_ONLY=true

# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/dennis-Z370-HD3-OP_dennis_imaterialist37k_740000/model.ckpt-740000"
# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/dennis-Z370-HD3-OP_dennis_imaterialist37k_augmented/model.ckpt-366824"
# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/dennis-Z370-HD3-OP_dennis_imaterialist37k_augmented/model.ckpt-693543"
# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/abq-gpu-2_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.001_output_stride_16_batch_size_16/model.ckpt-27356"
# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/abq-gpu-1_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.001_output_stride_16_batch_size_4/model.ckpt-109425"
# TF_INITIAL_CHECKPOINT_NAME="forma54k_0.8383"
# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/forma54k_2019-09-27_num_epochs_40_base_learning_rate_0.01_learning_rate_decay_0.1^2_abq-gpu-2_dennis/model.ckpt-109425"
# TF_INITIAL_CHECKPOINT_NAME="forma54k_0.8428"
# TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/2019-10-01_num_epochs_100_train_batch_size_32_train_crop_size_513,513_tf_initial_checkpoint_forma54k_0.8383_abq-gpu-2_dennis_forma54k/model.ckpt-136781"
TF_INITIAL_CHECKPOINT_NAME="forma54k_0.8490"
TF_INITIAL_CHECKPOINT="${INIT_MODELS_DIR}/2019-10-01_num_epochs_25_train_batch_size_8_train_crop_size_1025,1025_tf_initial_checkpoint_forma54k_0.8383_abq-gpu-1_dennis_forma54k/model.ckpt-136781"
INITIALIZE_LAST_LAYERS=true
LAST_LAYERS_CONTAINS_LOGITS_ONLY=true # irrelevant

# ========================== SETTINGS (LEARNING POLICY) ==========================
NUM_EPOCHS=100
NUM_EXAMPLES=`expr $DATASET_TRAIN_SIZE \* $NUM_EPOCHS`
NUM_ITERATIONS=`expr $NUM_EXAMPLES / $TRAIN_BATCH_SIZE`

# Initial Training
LEARNING_POLICY="step"
# BASE_LEARNING_RATE=0.007
BASE_LEARNING_RATE=0.01
LEARNING_RATE_DECAY_FACTOR=0.1
# LEARNING_RATE_DECAY_FACTOR_POWER=1  # Decay to 0.1^1 * BASE_LEARNING_RATE
LEARNING_RATE_DECAY_FACTOR_POWER=2  # Decay to 0.1^2 * BASE_LEARNING_RATE
# LEARNING_RATE_DECAY_FACTOR_POWER=3  # Decay to 0.1^3 * BASE_LEARNING_RATE
LEARNING_RATE_DECAY_STEP=`expr $NUM_ITERATIONS / $LEARNING_RATE_DECAY_FACTOR_POWER`
LEARNING_POWER=0                # unused

# # Fine-tuning
# LEARNING_POLICY="poly"
# BASE_LEARNING_RATE=0.0001
# LEARNING_RATE_DECAY_FACTOR=0  # unused
# LEARNING_RATE_DECAY_STEP=0    # unused
# LEARNING_POWER=0.9

# ========================== SETTINGS (LOGGING) ==========================
SAVE_INTERVAL_SECS=1200
SAVE_SUMMARIES_SECS=600

# ========================== SETTINGS (MODEL) ==========================
TRAIN_CROP_SIZE="513,513"
# TRAIN_CROP_SIZE="1025,1025"

TRAIN_OUTPUT_STRIDE=16
TRAIN_ATROUS_RATE_1=6
TRAIN_ATROUS_RATE_2=12
TRAIN_ATROUS_RATE_3=18

# TRAIN_OUTPUT_STRIDE=8
# TRAIN_ATROUS_RATE_1=12
# TRAIN_ATROUS_RATE_2=24
# TRAIN_ATROUS_RATE_3=36

EVAL_OUTPUT_STRIDE=16
EVAL_ATROUS_RATE_1=6
EVAL_ATROUS_RATE_2=12
EVAL_ATROUS_RATE_3=18

# EVAL_OUTPUT_STRIDE=8
# EVAL_ATROUS_RATE_1=12
# EVAL_ATROUS_RATE_2=24
# EVAL_ATROUS_RATE_3=36

MODEL_VARIANT="xception_65"
DECODER_OUTPUT_STRIDE=4

# ========================== SETTINGS (PATHS) ==========================
# Experiment Paths
# DATE=`date +"%Y-%m-%d_%H-%M-%S"`
DATE=`date +"%Y-%m-%d"`
HOSTNAME=`hostname`
USER=`whoami`
EXPERIMENT_DESCRIPTION="num_epochs_${NUM_EPOCHS}_train_batch_size_${TRAIN_BATCH_SIZE}_train_crop_size_${TRAIN_CROP_SIZE}_tf_initial_checkpoint_${TF_INITIAL_CHECKPOINT_NAME}"

EXPERIMENT_NAME="${DATE}_${EXPERIMENT_DESCRIPTION}_${HOSTNAME}_${USER}_${DATASET_NAME}"
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

# Train.
python3 ./train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant=${MODEL_VARIANT} \
  --atrous_rates=${TRAIN_ATROUS_RATE_1} \
  --atrous_rates=${TRAIN_ATROUS_RATE_2} \
  --atrous_rates=${TRAIN_ATROUS_RATE_3} \
  --output_stride=${TRAIN_OUTPUT_STRIDE} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE} \
  --train_crop_size=${TRAIN_CROP_SIZE} \
  --num_clones=${NUM_CLONES} \
  --train_batch_size=${TRAIN_BATCH_SIZE} \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --learning_policy="${LEARNING_POLICY}" \
  --base_learning_rate=${BASE_LEARNING_RATE} \
  --learning_rate_decay_factor=${LEARNING_RATE_DECAY_FACTOR} \
  --learning_rate_decay_step=${LEARNING_RATE_DECAY_STEP} \
  --learning_power=${LEARNING_POWER} \
  --fine_tune_batch_norm=${FINE_TUNE_BATCH_NORM} \
  --tf_initial_checkpoint="${TF_INITIAL_CHECKPOINT}" \
  --initialize_last_layer=${INITIALIZE_LAST_LAYERS} \
  --last_layers_contain_logits_only=${LAST_LAYERS_CONTAINS_LOGITS_ONLY} \
  --dataset="${DATASET_NAME}" \
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
  --atrous_rates=${EVAL_ATROUS_RATE_1} \
  --atrous_rates=${EVAL_ATROUS_RATE_2} \
  --atrous_rates=${EVAL_ATROUS_RATE_3} \
  --output_stride=${EVAL_OUTPUT_STRIDE} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE} \
  --eval_crop_size="${DATASET_EVAL_CROP_SIZE}" \
  --dataset="${DATASET_NAME}" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${DATASET_TFRECORD}" \
  --max_number_of_evaluations=1

# Visualize the results.
python3 ./vis.py \
  --logtostderr \
  --vis_split="trainval" \
  --model_variant=${MODEL_VARIANT} \
  --atrous_rates=${EVAL_ATROUS_RATE_1} \
  --atrous_rates=${EVAL_ATROUS_RATE_2} \
  --atrous_rates=${EVAL_ATROUS_RATE_3} \
  --output_stride=${EVAL_OUTPUT_STRIDE} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE} \
  --vis_crop_size="${DATASET_EVAL_CROP_SIZE}" \
  --dataset="${DATASET_NAME}" \
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
  --atrous_rates=${EVAL_ATROUS_RATE_1} \
  --atrous_rates=${EVAL_ATROUS_RATE_2} \
  --atrous_rates=${EVAL_ATROUS_RATE_3} \
  --output_stride=${EVAL_OUTPUT_STRIDE} \
  --decoder_output_stride=${DECODER_OUTPUT_STRIDE} \
  --num_classes="9" \
  --crop_size=1001 \
  --crop_size=1001 \
  --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
