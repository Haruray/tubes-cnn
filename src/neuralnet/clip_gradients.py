import numpy as np


def clip_gradients(gradients, threshold=1.0):
    return np.clip(gradients, -threshold, threshold)
