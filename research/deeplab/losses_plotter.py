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
    BATCH_SIZE = 16
    STEPS_PER_LOG = 10

    with open('abq-gpu-1-losses.txt') as f:
        lines = f.readlines()

    losses = np.array([float(line.strip()[16:-1]) for line in lines if line.startswith('Total')])/BATCH_SIZE
    examples = BATCH_SIZE*STEPS_PER_LOG*np.arange(losses.shape[0])
    losses_ewma = numpy_ewma_vectorized_v2(losses, 100)

    with open('abq-gpu-2-losses.txt') as f:
        lines = f.readlines()

    losses2 = np.array([float(line.strip()[16:-1]) for line in lines if line.startswith('Total')])/BATCH_SIZE
    examples2 = BATCH_SIZE*STEPS_PER_LOG*np.arange(losses2.shape[0])
    losses_ewma2 = numpy_ewma_vectorized_v2(losses2, 100)

    plt.figure()
    plt.title('DeepLab V3+ Training Losses')
    plt.xlabel('examples')
    plt.ylabel('loss')
    plt.plot(examples, losses, color='red', alpha=0.2)
    plt.plot(examples, losses_ewma, color='red', label='abq-gpu-1_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.001_output_stride_16_batch_size_16')
    plt.plot(examples2, losses2, color='green', alpha=0.2)
    plt.plot(examples2, losses_ewma2, color='green', label='abq-gpu-2_dennis_forma54k_augmented_learn_policy_step_learn_rate_0.007_output_stride_16_batch_size_16_init_model_pascal')
    plt.legend()
    plt.show()
