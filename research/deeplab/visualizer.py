import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.ion()

from utility.paths import STORAGE_DIR


EXPERIMENT_DIR = os.path.join(STORAGE_DIR, 'shared/deeplab/experiments')
# experiments = os.listdir(EXPERIMENT_DIR)
experiments = [
    # 'dennis-Z370-HD3-OP_dennis_imaterialist37k_augmented',
    # 'abq-gpu-1_dennis_imaterialist37k_augmented_learn_rate_001_output_stride_8_batch_size_4',
    # 'abq-gpu-1_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.001_output_stride_16_batch_size_16',
    # 'abq-gpu-2_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.007_output_stride_16_batch_size_16_init_model_pascal',
    # 'abq-gpu-1_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.001_output_stride_16_batch_size_4',
    # 'abq-gpu-2_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.001_output_stride_16_batch_size_16',
    'forma54k_2019-09-26_base_learn_rate_0.01_train_batch_size_16_train_crop_size_513,513_abq-gpu-2_dennis',
    'forma54k_2019-09-26_base_learn_rate_0.001_train_batch_size_4_train_crop_size_1025,1025_abq-gpu-1_dennis',
    'forma54k_2019-09-27_num_epochs_40_base_learning_rate_0.01_learning_rate_decay_0.1^2_abq-gpu-2_dennis'
]

fig, axs = plt.subplots(1, len(experiments) + 1, figsize=(48, 48))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

for i in range(len(axs)):
    if i == 0:
        axs[i].set_title('image')
    else:
        axs[i].set_title(experiments[i-1])
    axs[i].axis('off')

for i in range(500):
    print(i)
    fig.suptitle(f'image {i:06d}')
    image_path = os.path.join(EXPERIMENT_DIR, f'{experiments[0]}/vis/segmentation_results/{i:06d}_image.png')
    image = cv2.imread(image_path)
    axs[0].imshow(image[:, :, ::-1])
    for j in range(len(experiments)):
        image_path = os.path.join(EXPERIMENT_DIR, f'{experiments[j]}/vis/segmentation_results/{i:06d}_prediction.png')
        image = cv2.imread(image_path)
        axs[j+1].imshow(image[:, :, ::-1])

    import pdb; pdb.set_trace()

# def create_forma_label_colormap():
#   return np.asarray([
#       [0, 0, 0],        # seg_background    black
#       [255, 0, 0],      # seg_body          red
#       [0, 255, 0],      # seg_garment       green
#       [0, 0, 255],      # seg_skin          blue
#       [255, 255, 0],    # seg_hair          yellow
#       [255, 0, 255],    # seg_arms          magenta
#       [0, 255, 255],    # seg_shoe          cyan
#       [255, 255, 255],  # seg_sleeves       white
#       [128, 128, 128],  # seg_pants         gray
#       ])
