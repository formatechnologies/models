# Usage

## Install

```
cd ~
mkdir tensorflow
cd tensorflow
git clone git@github.com:formatechnologies/models.git

```

Test to see if the Pascal VOC pretrained model runs:
```
cd ~/tensorflow/models/research/deeplab
sh local_test.sh
```

## Dataset Creation
Build existing Forma datasets (requires `tensorflow-gpu==2.0.0`, like `iris`):
```
python3 ./datasets/build_imaterialist37k_data.py
python3 ./datasets/build_humanparsing17k_data.py
python3 ./datasets/build_nsfw1k_data.py
python3 ./datasets/build_tryon1k_data.py
python3 ./datasets/build_forma54k_data.py
python3 ./datasets/build_forma54k_nsfw1k_data.py
python3 ./datasets/build_forma54k_tryon1k_data.py
```

This will use the `~/storage/shared/deeplab/datasets` folder, which contains the following:
- `~/storage/shared/deeplab/datasets/DATASET_NAME/dataset_split`: describes train, trainval, val splits
- `~/storage/shared/deeplab/datasets/DATASET_NAME/tfrecord`: stores datasets as TFRecord

If creating a custom dataset:
1. Read http://hellodfan.com/2018/07/06/DeepLabv3-with-own-dataset/
1. Copy a `datasets/build_XXX_data.py` script to create TFRecords
1. Add a `DatasetDescriptor` in `datasets/data_generator.py`
1. Modify `local_test_forma.sh` with the correct `DATASET_NAME` and other parameters for training
1. (Optional) Add a label colormap in `utils/get_dataset_colormap.py`
1. (Optional) Add this dataset to `Forma54k` in `datasets/build_forma_data.py`

## Training
Create and activate a venv for `tensorflow-gpu==1.15`:
```
cd ~/tensorflow/models/research/deeplab
virtualenv venv
source venv/bin/activate
python3 -m pip install tensorflow-gpu==1.15
```
Train:
```
cd ~/tensorflow/models/research/deeplab
source venv/bin/activate
sh local_test_forma.sh

source venv/bin/activate
tensorboard --logdir=~/storage/shared/deeplab/experiments
```

This will use the `~/storage/shared/deeplab` folder, which contains the following:

- `~/storage/shared/deeplab/datasets`: stores datasets as TFRecord
- `~/storage/shared/deeplab/init_models`: stores some pretrained models for initialization
- `~/storage/shared/deeplab/experiments`: stores all experiment output

The settings in `local_test_forma.sh` are by default for GPU 1 and GPU 2:
```
ssh -XC dennis@192.168.16.140 #gpu1-workstation
ssh -XC dennis@192.168.16.150 #gpu2-workstation
```

If running on a local machine, be sure to change to using only a single GPU in `local_test_forma.sh`:
```
# ========================== SETTINGS (WORKSTATION) ==========================

# Dennis Workstation Settings
export TF_FORCE_GPU_ALLOW_GROWTH=true   # Workaround cuDNN bug with RTX GPUS
NUM_CLONES=1
TRAIN_BATCH_SIZE=1
FINE_TUNE_BATCH_NORM=false

# GPU 1 + GPU 2 Workstation Settings
# NUM_CLONES=8
# TRAIN_BATCH_SIZE=8
# FINE_TUNE_BATCH_NORM=false

# NUM_CLONES=4  # Don't use 8, draws too much power
# TRAIN_BATCH_SIZE=16
# FINE_TUNE_BATCH_NORM=true
```

Please see all DeepLab V3+ Hyperparameters described below, in `local_test_forma.sh`, and in the code
(`train.py`, `eval.py`, `vis.py`, `export_model.py`) before training.

# DeepLab V3+ Hyperparameters

See hyperparameters tradeoffs here:

https://arxiv.org/pdf/1706.05587.pdf

See some training history here:

https://docs.google.com/spreadsheets/d/19kLXbGjNFdv_5w-_VDh-6GdO6R-RJBzglAc73SaYcjs/edit?usp=sharing

## Build Forma Data

* TODO

## Data Generator

* TODO
* affine transformation (not exposed as parameter, in code only)
* color (HSB + contrast) (not exposed as parameter, in code only)
* blur + noise (not exposed as parameter, in code only)

## Train

### Dataset

* dataset: name of the segmentation dataset, this affects settings deep in the code
  * if creating a custom dataset, make sure to investigate everywhere this is used
* dataset_dir
* train_split: train, trainval, val, test
  * TODO: retrain on all data (train, trainval, val)

### Data Augmentation

* min_scale_factor
* max_scale_factor
* scale_factor_step_size

### Model Initialization

* tf_initial_checkpoint
* initialize_last_layer
  * true: reuse the entire model
  * false: don't reuse the last layer (defined by last_layers_contain_logits_only)
* last_layers_contain_logits_only (assumes initialize_last_layer=false)
  * true: the last layer contains the logits only (use this if you have a different number of classes)
  * false: the last layer contains logits, image pooling, aspp, concat_projection, decoder, meta_architecture
    * TODO: train from scratch (backbone weights only) with learn rate = 0.007 -> fine tune learn rate 0.0001
* last_layer_gradient_multiplier
* slow_start_step
* slow_start_learning_rate

### Settings for multi-GPUs/multi-replicas training.

* num_clones
  * Choose maximal number of clones to maximize batch size
  * Use 4 clones at most (8 clones will draw all the power in the gpu server room)
* clone_on_cpu
* num_replicas
* startup_delay_steps
* num_ps_tasks
* master
* task

### GPU Memory Constrained Settings
* output_stride
  * use output_stride = 16 by default
  * use output_stride = 8 for slightly better performance but longer training
  * finetune an output_stride=16 network by continuing training with output_stride=8
* atrous_rates
  * if output_stride = 8, use [12, 24, 36]
  * if output_stride = 16, use [6, 12, 18]
* train_batch_size
  * use maximal batch size for each clone:
    * 12GB memory = batch size 4 + output stride 16
    * 12GB memory = batch size 1 + output stride 8
  * use maximal batch size across all clones
    * 4 clones * batch size 4 / clone = total batch size 16
  * batch size 32 > batch size 16 > batch size 8 > batch size 4
* fine_tune_batch_norm: set to true if batch_size >= 12, else false
* train_crop_size: [513, 513]
  * TODO: test [1024, 1024] crop size (is higher resolution image better?)

### Learning Rate

* training_number_of_steps
* learning_policy
  * use 'step' (exponential decay) for training
  * use 'poly' (polynomial decay) for finetuning
* base_learning_rate
  * use 0.007 for training
  * use 0.0001 for finetuning
* learning_rate_decay_factor (for 'step' learning policy)
* learning_rate_decay_step (for 'step' learning policy)
* learning_power (for 'polynomial' learning policy)
* momentum
* weight_decay

### Settings for logging.

* train_logdir
* log_steps
* save_interval_secs
* save_summaries_secs
* save_summaries_images

### Miscellaneous

* upsample_logits (or downsample labels when computing loss)
* drop_path_keep_prob (NAS training strategy)
* hard_example_mining_step (hard example mining)
* top_k_percent_pixels (hard example mining)
* quantize_delay_step (quanitization)
* profile_logdir (profiling)

## Eval / Vis / Export

### Dataset

* dataset: name of the segmentation dataset, this affects settings deep in the code
  * if creating a custom dataset, make sure to investigate everywhere this is used
* dataset_dir
* train_split: train, trainval, val, test

### Model Initialization

* checkpoint_dir

### GPU Memory Constrained Settings

* output_stride
  * use output_stride = 16 by default
  * use output_stride = 8 for slightly better performance but longer eval
  * TODO: test
* atrous_rates
  * if output_stride = 8, use [12, 24, 36]
  * if output_stride = 16, use [6, 12, 18]
* eval_batch_size / vis_batch_size: 1
* eval_crop_size / vis_crop_size: set to size of largest image in dataset

### Other Evaluation Settings

* eval_scales / inference_scales
  * default: [1.0]
  * slightly better performance but slower: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
  * TODO: test
* add_flipped_images

### Logging / Visualization

* eval_logdir / vis_logdir
* eval_interval_secs
* max_number_of_evaluations
* colormap_type
  * if creating a custom dataset, make sure to investigate everywhere this is used
* also_save_raw_predictions

### Export

* checkpoint_path
* export_path
* num_classes
* crop_size: [1001, 10001] because we crop to 1000 x 1000 images
* save_inference_graph

### Miscellaneous

* master
* quantize_delay_step

# DeepLab: Deep Labelling for Semantic Image Segmentation

DeepLab is a state-of-art deep learning model for semantic image segmentation,
where the goal is to assign semantic labels (e.g., person, dog, cat and so on)
to every pixel in the input image. Current implementation includes the following
features:

1.  DeepLabv1 [1]: We use *atrous convolution* to explicitly control the
    resolution at which feature responses are computed within Deep Convolutional
    Neural Networks.

2.  DeepLabv2 [2]: We use *atrous spatial pyramid pooling* (ASPP) to robustly
    segment objects at multiple scales with filters at multiple sampling rates
    and effective fields-of-views.

3.  DeepLabv3 [3]: We augment the ASPP module with *image-level feature* [5, 6]
    to capture longer range information. We also include *batch normalization*
    [7] parameters to facilitate the training. In particular, we applying atrous
    convolution to extract output features at different output strides during
    training and evaluation, which efficiently enables training BN at output
    stride = 16 and attains a high performance at output stride = 8 during
    evaluation.

4.  DeepLabv3+ [4]: We extend DeepLabv3 to include a simple yet effective
    decoder module to refine the segmentation results especially along object
    boundaries. Furthermore, in this encoder-decoder structure one can
    arbitrarily control the resolution of extracted encoder features by atrous
    convolution to trade-off precision and runtime.

If you find the code useful for your research, please consider citing our latest
works:

*   DeepLabv3+:

```
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```

*   MobileNetv2:

```
@inproceedings{mobilenetv22018,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Mark Sandler and Andrew Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
  booktitle={CVPR},
  year={2018}
}
```

*  Architecture search for dense prediction cell:

```
@inproceedings{dpc2018,
  title={Searching for Efficient Multi-Scale Architectures for Dense Image Prediction},
  author={Liang-Chieh Chen and Maxwell D. Collins and Yukun Zhu and George Papandreou and Barret Zoph and Florian Schroff and Hartwig Adam and Jonathon Shlens},
  booktitle={NIPS},
  year={2018}
}

```

*  Auto-DeepLab (also called hnasnet in core/nas_network.py):

```
@inproceedings{autodeeplab2019,
  title={Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic
Image Segmentation},
  author={Chenxi Liu and Liang-Chieh Chen and Florian Schroff and Hartwig Adam
  and Wei Hua and Alan Yuille and Li Fei-Fei},
  booktitle={CVPR},
  year={2019}
}

```


In the current implementation, we support adopting the following network
backbones:

1.  MobileNetv2 [8]: A fast network structure designed for mobile devices.

2.  Xception [9, 10]: A powerful network structure intended for server-side
    deployment.

3.  ResNet-v1-{50,101} [14]: We provide both the original ResNet-v1 and its
    'beta' variant where the 'stem' is modified for semantic segmentation.

4.  PNASNet [15]: A Powerful network structure found by neural architecture
    search.

5.  Auto-DeepLab (called HNASNet in the code): A segmentation-specific network
    backbone found by neural architecture search.

This directory contains our TensorFlow [11] implementation. We provide codes
allowing users to train the model, evaluate results in terms of mIOU (mean
intersection-over-union), and visualize segmentation results. We use PASCAL VOC
2012 [12] and Cityscapes [13] semantic segmentation benchmarks as an example in
the code.

Some segmentation results on Flickr images:
<p align="center">
    <img src="g3doc/img/vis1.png" width=600></br>
    <img src="g3doc/img/vis2.png" width=600></br>
    <img src="g3doc/img/vis3.png" width=600></br>
</p>

## Contacts (Maintainers)

*   Liang-Chieh Chen, github: [aquariusjay](https://github.com/aquariusjay)
*   YuKun Zhu, github: [yknzhu](https://github.com/YknZhu)
*   George Papandreou, github: [gpapan](https://github.com/gpapan)
*   Hui Hui, github: [huihui-personal](https://github.com/huihui-personal)
*   Maxwell D. Collins, github: [mcollinswisc](https://github.com/mcollinswisc)
*   Ting Liu: github: [tingliu](https://github.com/tingliu)

## Tables of Contents

Demo:

*   <a href='https://colab.sandbox.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb'>Colab notebook for off-the-shelf inference.</a><br>

Running:

*   <a href='g3doc/installation.md'>Installation.</a><br>
*   <a href='g3doc/pascal.md'>Running DeepLab on PASCAL VOC 2012 semantic segmentation dataset.</a><br>
*   <a href='g3doc/cityscapes.md'>Running DeepLab on Cityscapes semantic segmentation dataset.</a><br>
*   <a href='g3doc/ade20k.md'>Running DeepLab on ADE20K semantic segmentation dataset.</a><br>

Models:

*   <a href='g3doc/model_zoo.md'>Checkpoints and frozen inference graphs.</a><br>

Misc:

*   Please check <a href='g3doc/faq.md'>FAQ</a> if you have some questions before reporting the issues.<br>

## Getting Help

To get help with issues you may encounter while using the DeepLab Tensorflow
implementation, create a new question on
[StackOverflow](https://stackoverflow.com/) with the tag "tensorflow".

Please report bugs (i.e., broken code, not usage questions) to the
tensorflow/models GitHub [issue
tracker](https://github.com/tensorflow/models/issues), prefixing the issue name
with "deeplab".

## License

All the codes in deeplab folder is covered by the [LICENSE](https://github.com/tensorflow/models/blob/master/LICENSE)
under tensorflow/models. Please refer to the LICENSE for details.

## Change Logs

### March 6, 2019

* Released the evaluation code (under the `evaluation` folder) for image
parsing, a.k.a. panoptic segmentation. In particular, the released code supports
evaluating the parsing results in terms of both the parsing covering and
panoptic quality metrics. **Contributors**: Maxwell Collins and Ting Liu.


### February 6, 2019

* Updated decoder module to exploit multiple low-level features with different
output_strides.

### December 3, 2018

* Released the MobileNet-v2 checkpoint on ADE20K.


### November 19, 2018

* Supported NAS architecture for feature extraction. **Contributor**: Chenxi Liu.

* Supported hard pixel mining during training.


### October 1, 2018

* Released MobileNet-v2 depth-multiplier = 0.5 COCO-pretrained checkpoints on
PASCAL VOC 2012, and Xception-65 COCO pretrained checkpoint (i.e., no PASCAL
pretrained).


### September 5, 2018

* Released Cityscapes pretrained checkpoints with found best dense prediction cell.


### May 26, 2018

* Updated ADE20K pretrained checkpoint.


### May 18, 2018
* Added builders for ResNet-v1 and Xception model variants.
* Added ADE20K support, including colormap and pretrained Xception_65 checkpoint.
* Fixed a bug on using non-default depth_multiplier for MobileNet-v2.


### March 22, 2018

* Released checkpoints using MobileNet-V2 as network backbone and pretrained on
PASCAL VOC 2012 and Cityscapes.


### March 5, 2018

* First release of DeepLab in TensorFlow including deeper Xception network
backbone. Included chekcpoints that have been pretrained on PASCAL VOC 2012
and Cityscapes.

## References

1.  **Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs**<br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille (+ equal
    contribution). <br />
    [[link]](https://arxiv.org/abs/1412.7062). In ICLR, 2015.

2.  **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,**
    **Atrous Convolution, and Fully Connected CRFs** <br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille (+ equal
    contribution). <br />
    [[link]](http://arxiv.org/abs/1606.00915). TPAMI 2017.

3.  **Rethinking Atrous Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.

4.  **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.

5.  **ParseNet: Looking Wider to See Better**<br />
    Wei Liu, Andrew Rabinovich, Alexander C Berg<br />
    [[link]](https://arxiv.org/abs/1506.04579). arXiv:1506.04579, 2015.

6.  **Pyramid Scene Parsing Network**<br />
    Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia<br />
    [[link]](https://arxiv.org/abs/1612.01105). In CVPR, 2017.

7.  **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate shift**<br />
    Sergey Ioffe, Christian Szegedy <br />
    [[link]](https://arxiv.org/abs/1502.03167). In ICML, 2015.

8.  **MobileNetV2: Inverted Residuals and Linear Bottlenecks**<br />
    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br />
    [[link]](https://arxiv.org/abs/1801.04381). In CVPR, 2018.

9.  **Xception: Deep Learning with Depthwise Separable Convolutions**<br />
    François Chollet<br />
    [[link]](https://arxiv.org/abs/1610.02357). In CVPR, 2017.

10. **Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge 2017 Entry**<br />
    Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei, Jifeng Dai<br />
    [[link]](http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf). ICCV COCO Challenge
    Workshop, 2017.

11. **Tensorflow: Large-Scale Machine Learning on Heterogeneous Distributed Systems**<br />
    M. Abadi, A. Agarwal, et al. <br />
    [[link]](https://arxiv.org/abs/1603.04467). arXiv:1603.04467, 2016.

12. **The Pascal Visual Object Classes Challenge – A Retrospective,** <br />
    Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John
    Winn, and Andrew Zisserma. <br />
    [[link]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). IJCV, 2014.

13. **The Cityscapes Dataset for Semantic Urban Scene Understanding**<br />
    Cordts, Marius, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele. <br />
    [[link]](https://www.cityscapes-dataset.com/). In CVPR, 2016.

14. **Deep Residual Learning for Image Recognition**<br />
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. <br />
    [[link]](https://arxiv.org/abs/1512.03385). In CVPR, 2016.

15. **Progressive Neural Architecture Search**<br />
    Chenxi Liu, Barret Zoph, Maxim Neumann, Jonathon Shlens, Wei Hua, Li-Jia Li, Li Fei-Fei, Alan Yuille, Jonathan Huang, Kevin Murphy. <br />
    [[link]](https://arxiv.org/abs/1712.00559). In ECCV, 2018.
