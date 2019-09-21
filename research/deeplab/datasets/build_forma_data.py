import os
import glob
import shutil

from tqdm import tqdm

from utility.paths import STORAGE_DIR

DEEPLAB_DIR = os.path.join(STORAGE_DIR, 'shared/deeplab')
DATASETS_DIR = os.path.join(DEEPLAB_DIR, 'datasets')

dataset_name_list = ['imaterialist37k', 'humanparsing17k']
dataset_name_forma = 'forma54k'
dataset_splits = ['train', 'trainval', 'val']

DATASET_FORMA_DIR = os.path.join(DATASETS_DIR, dataset_name_forma)
DATASET_FORMA_TFRECORD_DIR = os.path.join(DATASET_FORMA_DIR, 'tfrecord')
if not os.path.exists(DATASET_FORMA_DIR):
    os.mkdir(DATASET_FORMA_DIR)
if not os.path.exists(DATASET_FORMA_TFRECORD_DIR):
    os.mkdir(DATASET_FORMA_TFRECORD_DIR)

print(f'Building {dataset_name_forma}')
for dataset_split in dataset_splits:
    print(f'\nProcessing {dataset_split}\n')
    tfrecord_files = []
    for dataset_name in dataset_name_list:
        tfrecord_files += glob.glob(f'{DATASETS_DIR}/{dataset_name}/tfrecord/{dataset_split}-*')
    tfrecord_files = sorted(tfrecord_files)
    num_files = len(tfrecord_files)
    for i, tfrecord_file in enumerate(tqdm(tfrecord_files)):
        output_file = os.path.join(DATASET_FORMA_TFRECORD_DIR, f'{dataset_split}-{i:05d}-of-{num_files:05d}.tfrecord')
        if not os.path.exists(output_file):
            shutil.copy2(tfrecord_file, output_file)
