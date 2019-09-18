import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.ion()

from utility.paths import STORAGE_DIR


EXPERIMENT_DIR = os.path.join(STORAGE_DIR, 'shared/deeplab/experiments')
experiments = os.listdir(EXPERIMENT_DIR)

fig, axs = plt.subplots(1, len(experiments) + 1, figsize=(48, 48))
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

for i in range(len(axs)):
    if i == 0:
        axs[i].set_title('image')
    else:
        axs[i].set_title(experiments[i-1])
    axs[i].axis('off')

for i in range(500):
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
