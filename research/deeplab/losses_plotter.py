import os

import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

if __name__ == '__main__':
    HOME_DIR = os.path.expanduser('~')
    EXPERIMENTS_DIR = os.path.join(HOME_DIR, 'storage/shared/deeplab/experiments')

    experiments = [
        {
            'name': 'abq-gpu-1_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.001_output_stride_16_batch_size_16',
            'batch_size': 16,
            'steps_per_log': 10,
            'color': 'red',
        },
        {
            'name': 'abq-gpu-2_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.007_output_stride_16_batch_size_16_init_model_pascal',
            'batch_size': 16,
            'steps_per_log': 10,
            'color': 'green',
        },
        {
            'name': 'forma54k_2019-09-26_base_learn_rate_0.01_train_batch_size_16_train_crop_size_513,513_abq-gpu-2_dennis',
            'batch_size': 16,
            'steps_per_log': 10,
            'color': 'blue',
        },
    ]

    plt.figure()
    plt.title('DeepLab V3+ Training Losses')
    plt.xlabel('examples')
    plt.ylabel('loss')
    for experiment in experiments:

        with open(os.path.join(EXPERIMENTS_DIR, experiment['name'], 'log.txt')) as f:
            lines = f.readlines()

        losses = np.array([float(line.strip()[16:-1]) for line in lines if line.startswith('Total')])/experiment['batch_size']
        examples = experiment['batch_size']*experiment['steps_per_log']*np.arange(losses.shape[0])
        losses_ewma = numpy_ewma_vectorized_v2(losses, 100)

        plt.plot(examples, losses, color=experiment['color'], alpha=0.2)
        plt.plot(examples, losses_ewma, color=experiment['color'], label=experiment['name'])
    plt.legend()
    plt.show()
