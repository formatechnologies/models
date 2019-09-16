import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

def plot_images(dataset, n_images, samples_per_image):
    output = np.zeros((32 * n_images, 32 * samples_per_image, 3))

    row = 0
    for images in dataset.repeat(samples_per_image).batch(n_images):
        output[:, row*32:(row+1)*32] = np.vstack(images.numpy())
        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

iterator = dataset.get_one_shot_iterator()
next_element = iterator.get_next()

fig, axs = plt.subplots(1, 3)

with tf.Session() as sess:
    while True:
        sample = sess.run(next_element)
        original_image = sample['original_image'].astype(np.uint8).squeeze()
        image = sample['image'].astype(np.uint8).squeeze()
        label = sample['label'].squeeze()
        label[0, 0] = 10       # consistent cmap label colors
        label[label == 255] = 10

        axs[0].imshow(original_image)
        axs[1].imshow(image)
        axs[2].imshow(label, cmap='gray')

        import pdb; pdb.set_trace()
