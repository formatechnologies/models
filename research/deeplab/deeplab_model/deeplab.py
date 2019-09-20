''' Trained DeepLabV3+ model loading '''
import sys
import os
import time
import logging
import numpy as np
import tensorflow as tf
import yaml

# default path for Docker
# MODEL_PATH = 'deeplab_model/frozen_inference_graph.pb'
# CLASS_NAMES_PATH = 'deeplab_model/classes.yml'
MODEL_PATH = '/home/dennis/tensorflow/models/research/deeplab/deeplab_model/frozen_inference_graph.pb'
CLASS_NAMES_PATH = '/home/dennis/tensorflow/models/research/deeplab/deeplab_model/class.yml'


def load_class_names(filepath):
    ''' load YAML file with segmentation class names in it'''
    with open(filepath, 'r') as ymlfile:
        loaded = yaml.load(ymlfile)
    return loaded['classes']


class DeepLabModel(object):  # pylint: disable=too-few-public-methods
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

    def __init__(self, model_path):
        """Creates and loads pretrained deeplab model."""
        with tf.gfile.GFile(model_path, "rb") as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph')

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        inf_start = time.time()
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [image]})
        logging.debug("Inference takes %s sec", time.time() - inf_start)
        seg_map = batch_seg_map[0]
        return seg_map


class HumanParsingMachine:
    def __init__(self):
        self.model = DeepLabModel(MODEL_PATH)
        self.class_names = load_class_names(CLASS_NAMES_PATH)

    def run(self, image):
        """BGR uint8 image input"""
        seg_map = self.model.run(image[:, :, ::-1])
        one_hot = tf.keras.utils.to_categorical(
            seg_map, num_classes=len(self.class_names))
        pred = one_hot * 255
        pred = pred.astype(np.uint8)
        masks = dict()
        for idx, name in enumerate(self.class_names):
            masks[name] = pred[:, :, idx]

        logging.debug('Merging into a foreground mask...')
        fg_mask = 255 - masks['background']

        logging.debug('Merging masks...')
        masks['skin'] = np.maximum(masks['skin'], masks['arms'])
        masks['garment'] = np.maximum(masks['garment'], masks['sleeves'])
        masks['garment'] = np.maximum(masks['garment'], masks['pants'])

        result = {
            'seg_body': fg_mask
        }
        for name in ['hair', 'skin', 'arms', 'garment', 'shoe', 'sleeves', 'pants']:
            result['seg_'+name] = masks[name]
        return result
