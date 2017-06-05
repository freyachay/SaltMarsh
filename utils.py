import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

# Test comment
# Define metrics to be used by Keras fit()
import keras.backend as K


# Turns an array of patches (N, W, H) into a single tiled image (WW, HH)
def unpatch(y, num_patches_wide, num_patches_high):
    N, W, H = y.shape
    # N must be equal to num_patches_wide * num_patches_high
    assert(N == num_patches_wide * num_patches_high)
    y = y.transpose(1, 0, 2)
    y = y.reshape(W, num_patches_wide, num_patches_high, H)
    y = y.transpose(1, 0, 2, 3)
    return y.reshape(num_patches_wide * W, num_patches_high * H)

# Credit to nshaud 
# (https://github.com/nshaud/DeepNetsForEO/blob/master/notebooks/Image%20extraction.ipynb) for general outline of making patches.
def sliding_window(image, patch_size, step_size):
    """Extract chips using a sliding window
    
    Args:
        image (numpy array): The image to be processed, must be 3 dimensional.
        stride (int): The sliding window stride.
        patch_size(int): The patch size.

    Returns:
        list: list of patches with patch_size dimensions
    """
    patches = []
    for i in range(0, image.shape[0], step_size):
        for j in range(0, image.shape[1], step_size):
            new_patch = image[i:i+patch_size, j:j+patch_size, :]
            if (new_patch.shape[0], new_patch.shape[1]) == (patch_size, patch_size):
                patches.append(new_patch)
    return patches


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(p, r):
    return 2 * (p * r) /(p + r)

## Load the data: expects suffix string like '_p256_s256_50'
def load_data(suffix):
    # Get all saved patches data
    print("Loading data start.")

    X_train = np.load("train_patches_data" + suffix + ".npy")
    y_train = np.load("train_patches_labels" + suffix + ".npy")

    X_val = np.load("val_patches_data" + suffix + ".npy")
    y_val = np.load("val_patches_labels" + suffix + ".npy")

    print("Loading data finished.")
    return X_train, y_train, X_val, y_val

def percent_saltmarsh(labels):
    return (np.sum(labels)/float((labels).size))

def report_history(hist):
    # list all data in history
    print('Final F1: ' + str(f1(0.7009, 0.6123)))
    # summarize history for accuracy
    plt.plot(hist.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def display_tile(image, labels, fig_size=(15,15), save_plot_path=None, dpi=None):
    """
    image: raw, unormalized
    labels: 0 or 1
    """
    # image only
    fig = plt.figure(figsize = fig_size)
    sub1 = fig.add_subplot(1, 3, 1)
    image = image.astype('float64')
    image *= 255.0/np.max(image)  # normalize
    image = image.astype('uint8')
    sub1.imshow(image)
    
    # labels only
    sub2 = fig.add_subplot(1, 3, 2)
    sub2.imshow(labels, cmap="gray_r")
    
    # both
    sub3 = fig.add_subplot(1, 3, 3)
#     plt.imshow(image)
#     plt.imshow(labels, cmap="gray_r", alpha=0.3)
    image[labels.astype("bool"), 2] = 200
    sub3.imshow(image)
    
    if save_plot_path != None:
        if dpi != None:
            plt.savefig(save_plot_path, dpi=dpi)
        else:
            plt.savefig(save_plot_path)
    plt.show()
