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

"""Converts PASCAL VOC 2012 data to TFRecord file format with Example protos.

PASCAL VOC 2012 dataset is expected to have the following directory structure:

  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation

This script converts data into sharded data files and save at tfrecord folder.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import math
import os.path
import sys
import build_data
import tensorflow as tf

import glob
import json
from tqdm import tqdm

import numpy as np
import cv2

from iris.utility.paths import STORAGE_DIR
from iris.utility.json_tools import load_dict_from_json


_NUM_PER_SHARD = 500

DATASETS_DIR = os.path.join(STORAGE_DIR, 'shared/datasets')
DATA_DIR = os.path.join(DATASETS_DIR, 'json/train_json/')
LABELS_FILE = os.path.join(DATASETS_DIR, 'imat-fashion/label_descriptions.json')

DATASET_NAME = 'imaterialist37k_landmarks'
DATASETS_DIR = os.path.join(STORAGE_DIR, 'shared/deeplab/datasets')
DATASET_DIR = os.path.join(DATASETS_DIR, DATASET_NAME)
DATASET_TFRECORD_DIR = os.path.join(DATASET_DIR, 'tfrecord')
DATASET_SPLIT_DIR = os.path.join(DATASET_DIR, 'dataset_split')
for dir_ in [DATASET_DIR, DATASET_TFRECORD_DIR, DATASET_SPLIT_DIR]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)

seg_name_to_label = {
  'seg_background': 0,
  'seg_body': 1,
  'seg_garment': 2,
  'seg_skin': 3,
  'seg_hair': 4,
  'seg_arms': 5,
  'seg_shoe': 6,
  'seg_sleeves': 7,
  'seg_pants': 8,
}

with open(LABELS_FILE, 'r') as f:
  label_descriptions = json.load(f)
fashion_names_to_bits = {item['name']: item['id'] for item in label_descriptions['categories']}

# max_h, max_w = 0, 0
# filenames = sorted(os.listdir(DATA_DIR))
# for filename in tqdm(filenames):
#   json_filename = os.path.join(DATA_DIR, filename)
#   example = load_dict_from_json(json_filename)
#   h, w, _ = example['image'].shape
#   max_h = max(max_h, h)
#   max_w = max(max_w, w)
# print(max_h, max_w)

def _create_dataset_splits(data_dir, dataset_split_dir):
  filenames = sorted(os.listdir(data_dir))

  train_split = int(0.8 * len(filenames))
  valid_split = int(0.9 * len(filenames))

  if not os.path.exists(dataset_split_dir):
    os.mkdir(dataset_split_dir)
  with open(os.path.join(dataset_split_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(filenames[:train_split]))
  with open(os.path.join(dataset_split_dir, 'trainval.txt'), 'w') as f:
    f.write('\n'.join(filenames[train_split:valid_split]))
  with open(os.path.join(dataset_split_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join(filenames[valid_split:]))

def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('\nProcessing ' + dataset + '\n')
  filenames = sorted([x.strip('\n') for x in open(dataset_split, 'r')])
  num_images = len(filenames)
  num_shards = int(math.ceil(num_images / float(_NUM_PER_SHARD)))

  shard_id_start = -1
  while True:
    shard_id_start += 1
    output_filename = os.path.join(DATASET_TFRECORD_DIR, f'{dataset}-{shard_id_start:05d}-of-{num_shards:05d}.tfrecord')
    if not os.path.exists(output_filename):
      break
  # shard_id_start = 0
  if shard_id_start == num_shards:
    return
  shard_id_start = max(0, shard_id_start - 1)

  for shard_id in tqdm(range(num_shards)):
    if shard_id < shard_id_start:
      continue
    output_filename = os.path.join(DATASET_TFRECORD_DIR, f'{dataset}-{shard_id:05d}-of-{num_shards:05d}.tfrecord')
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * _NUM_PER_SHARD
      end_idx = min((shard_id + 1) * _NUM_PER_SHARD, num_images)
      for i in tqdm(range(start_idx, end_idx)):
        # Read the image.
        json_filename = os.path.join(DATA_DIR, filenames[i])
        example = load_dict_from_json(json_filename)
        image = example['image']
        image_data = to_image_bytestring(image, '.jpg')
        height, width = image.shape[:2]

        # Read the semantic segmentation annotation.
        fashion_dict = decode_segmentation(example['seg_fashion_parsing'], fashion_names_to_bits)
        seg_dict = {k: v for k, v in example.items() if k in seg_name_to_label}
        seg_dict['seg_sleeves'] = fashion_dict['sleeve']
        seg_dict['seg_pants'] = np.maximum(fashion_dict['pants'], fashion_dict['shorts'])
        seg_dict['seg_background'] = get_seg_background(seg_dict)
        seg = encode_segmentation_exclusive(seg_dict, seg_name_to_label)
        seg = seg[:, :, np.newaxis].repeat(3, axis=2)
        seg_data = to_image_bytestring(seg, '.png')
        seg_height, seg_width = seg.shape[:2]

        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')

        # Read the landmarks data.
        landmarks = example['pose_landmarks']
        landmarks_data = tf.io.serialize_tensor(landmarks.astype(np.float32))

        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data, landmarks_data=landmarks_data)
        tfrecord_writer.write(example.SerializeToString())

def encode_segmentation(seg_dict, encode_dict):
  # Encode up to 8 segmentation images via bits of np.uint8
  image = np.zeros(next(iter(seg_dict.values())).shape, np.uint8)
  for seg_name, bits in encode_dict.items():
    seg = seg_dict[seg_name]
    seg = (seg > 128).astype(np.uint8)
    seg_bits = np.left_shift(seg, bits)
    image += seg_bits
  return image

def encode_segmentation_exclusive(seg_dict, encode_dict):
  # Encode up to 8 segmentation images via bits of np.uint8
  image = np.zeros(next(iter(seg_dict.values())).shape, np.uint8)
  for seg_name, bits in encode_dict.items():
    seg = seg_dict[seg_name]
    image[seg > 128] = bits
  return image

def decode_segmentation(image, decode_dict):
  # Assumes image has enough channels to support different bit channels for decode_dict
  seg_dict = {}
  bit_mask = np.ones(image.shape[:2], image.dtype)
  for seg_name, bits in decode_dict.items():
    seg_dict[seg_name] = (image & (np.left_shift(bit_mask, bits)) > 0).astype(np.uint8) * 255
  return seg_dict

def decode_segmentation_exclusive(image, decode_dict):
  # Assumes image has enough channels to support different bit channels for decode_dict
  seg_dict = {}
  bit_mask = np.ones(image.shape[:2], image.dtype)
  for seg_name, bits in decode_dict.items():
    seg_dict[seg_name] = (image == bits).astype(np.uint8) * 255
  return seg_dict

def to_image_bytestring(arr, ext='.png'):
  """
  Args:
    arr: a numpy array

  Returns:
    a bytestring of the array encoded as a jpg
  """
  success, arr_jpg = cv2.imencode(ext, arr)
  return arr_jpg.tostring()

def get_seg_background(seg_dict):
  image = 255*np.ones(next(iter(seg_dict.values())).shape, np.uint8)
  for seg in seg_dict.values():
    image[seg > 128] = 0
  return image

def main(unused_argv):
  print(f'Building {DATASET_NAME}')
  _create_dataset_splits(DATA_DIR, DATASET_SPLIT_DIR)
  dataset_splits = sorted(tf.io.gfile.glob(os.path.join(DATASET_SPLIT_DIR, '*.txt')))
  for dataset_split in dataset_splits:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  tf.compat.v1.app.run()
