import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2



def show_image(location, title, img):
    plt.subplot(*location)
    plt.title(title,fontsize=8)
    plt.axis('off')
    if len(img.shape) == 3:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')

def evaluate(X_data, y_data, accuracy_operation, BATCH_SIZE, x, y, keep_prob=None):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        batch_x = tf.reshape(tf.stack(batch_x), (batch_x.shape[0], 32, 32, 1)).eval()
        if keep_prob is not None:
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        else:
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def preprocessing(X_train):
    # image preprocessing
    # grayscaling
    X_train_gray = X_train[:, :, :, 0]
    for i in range(X_train.shape[0]):
        X_train_gray[i, :, :] = cv2.cvtColor(X_train[i, :, :], cv2.COLOR_RGB2GRAY)
    # contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    X_train_clahe = X_train_gray
    for i in range(X_train.shape[0]):
        X_train_clahe[i, :, :] = clahe.apply(X_train_clahe[i, :, :])
    # normalize image intensities
    X_train_clahe = (X_train_clahe / 255) * 2 - 1  # normalize intensity
    return X_train_clahe

def MultiScaleArch(x, dropout):
    """
    See "Traffic Sign Recognition with MultiScale Convolutional Neural Networks" by Sermanet, 2011.
    """
    mu = 0
    sigma = 0.1

    # *****************
    # **** Layer 1 ****
    # *****************

    # Convolutional. Input = 32x32x1. Output = 28x28x108.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 108), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(108))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.tanh(conv1)
    # Pooling. Input = 28x28x108. Output = 14x14x108.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # *****************
    # **** Layer 2 ****
    # *****************

    # Convolutional. Input = 14x14x108. Output = 10x10x108.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 108, 108), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(108))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.tanh(conv2)
    # Pooling. Input = 10x10x108. Output = 5x5x108.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # *****************
    # **** Layer 3 ****
    # *****************

    # From Layer 2: Input = 5x5x108. Output = 3x3x108
    conv32_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 108, 108)))
    conv32_b = tf.Variable(tf.zeros(108))
    conv32 = tf.nn.conv2d(conv2, conv32_W, strides=[1, 1, 1, 1], padding='VALID') + conv32_b
    # Activation. Output = 1x972
    conv32_active = tf.nn.tanh(conv32)
    # Flattening
    conv32_active_flat = tf.contrib.layers.flatten(conv32_active)

    # From Layer 2: Input = 5x5x108. Output = 2700
    conv2_flat = tf.contrib.layers.flatten(conv2)

    # From Layer 1: Input = 14x14x108. Output = 1x21168.
    conv1_flat = tf.contrib.layers.flatten(conv1)

    # Combine from Layer 1 and from Layer 2. Output = 1x24840
    concat = tf.concat([conv32_active_flat, conv2_flat, conv1_flat], axis=1)
    print("concatenated shape = ")
    print(concat.shape)

    # Fully Connected. Input = 1x24840. Output = 1x100.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(24840, 100), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(100))
    fc1 = tf.matmul(concat, fc1_W) + fc1_b
    # Activation
    fc1 = tf.nn.tanh(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Fully Connected. Input = 1x100. Output = 1x42.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(100, 42), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(42))
    logits = tf.matmul(fc1, fc2_W) + fc2_b

    return logits