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

"""Prepares the data used for DeepLab training/evaluation."""
import numpy as np
import tensorflow as tf
from deeplab.core import feature_extractor
from deeplab.core import preprocess_utils


# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5


def preprocess_image_and_label(image,
                               label,
                               landmarks,
                               crop_height,
                               crop_width,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               is_training=True,
                               model_variant=None):
  """Preprocesses the image and label.

  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.

  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.

  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  original_image = image

  processed_image = tf.cast(image, tf.float32)

  if label is not None:
    label = tf.cast(label, tf.int32)

  # Crop image and label using landmarks, padding cropped areas with mean_pixel and ignore_label
  mean_pixel = tf.reshape(
      feature_extractor.mean_pixel(model_variant), [1, 1, 3])
  if is_training:
    processed_image, label = tf.cond(
        tf.random.uniform([], 0, 1) < 0.2,
        lambda: preprocess_utils.random_crop_legs(processed_image, label,
            landmarks, mean_pixel, ignore_label),
        lambda: (processed_image, label))

  # Resize image and label to the desired range.
  if min_resize_value or max_resize_value:
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))
    # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)

  # Data augmentation by randomly scaling the inputs.
  if is_training:
    scale = preprocess_utils.get_random_scale(
        min_scale_factor, max_scale_factor, scale_factor_step_size)
    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, scale)
    processed_image.set_shape([None, None, 3])

  # Pad image and label to have dimensions >= [crop_height, crop_width]
  image_shape = tf.shape(processed_image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  target_height = image_height + tf.maximum(crop_height - image_height, 0)
  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  # Pad image with mean pixel value.
  processed_image = preprocess_utils.pad_to_bounding_box(
      processed_image, 0, 0, target_height, target_width, mean_pixel)

  if label is not None:
    label = preprocess_utils.pad_to_bounding_box(
        label, 0, 0, target_height, target_width, ignore_label)

  # Randomly crop the image and label.
  if is_training and label is not None:
    processed_image, label = preprocess_utils.random_crop(
        [processed_image, label], crop_height, crop_width,
        do_affine_perturbation=True)

  processed_image.set_shape([crop_height, crop_width, 3])

  if label is not None:
    label.set_shape([crop_height, crop_width, 1])

  if is_training:
    # Randomly left-right flip the image and label.
    processed_image, label, _ = preprocess_utils.flip_dim(
        [processed_image, label], _PROB_OF_FLIP, dim=1)

  if is_training:
    # TODO: expose as function parameters
    # TODO: random order of perturbations
    # TODO: Perspective, Smart crop based on OpenPose, GAN / Neural Style Transfer

    processed_image = color(processed_image)
    processed_image = tf.cond(tf.random.uniform([], 0, 1) < 0.1, lambda: blur(processed_image), lambda: processed_image)
    processed_image = noise(processed_image)
    processed_image = tf.clip_by_value(processed_image, 0, 255)

  return original_image, processed_image, label


def color(x: tf.Tensor,
          hue_max_delta=0.08,
          saturation_facter_lb=0.5,
          saturation_facter_ub=1.5,
          brightness_max_delta=0.2,
          contract_factor_lb=0.7,
          contrast_factor_ub=1.3) -> tf.Tensor:
  # https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/
  """Color augmentation

  Args:
      x: Image

  Returns:
      Augmented image
  """
  x = tf.image.random_hue(x, hue_max_delta)
  x = tf.image.random_saturation(x, saturation_facter_lb, saturation_facter_ub)
  x = tf.image.random_brightness(x, brightness_max_delta)
  x = tf.image.random_contrast(x, contract_factor_lb, contrast_factor_ub)
  return x


def blur(x, mean=0.0, stddev=1.0):
  """
  Resize to smaller size (AREA) and then resize to original size (BILINEAR)
  """
  size = tf.shape(x)[:2]
  downsample_factor = 1 + tf.math.abs(tf.random.normal([], mean=mean, stddev=stddev))
  small_size = tf.to_int32(tf.to_float(size)/downsample_factor)
  x = tf.image.resize_images(x, small_size, method=tf.image.ResizeMethod.AREA)
  x = tf.image.resize_images(x, size, method=tf.image.ResizeMethod.BILINEAR)
  return x


def noise(x, mean=0.0, stddev=0.01):
  x += tf.random.normal([], mean=mean, stddev=stddev)
  return x
