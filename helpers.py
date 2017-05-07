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
        X_train_gray[i, :, :] = cv2.cvtColor(X_train[i, :, :, :], cv2.COLOR_RGB2GRAY)
    # contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    X_train_clahe = X_train_gray
    for i in range(X_train.shape[0]):
        X_train_clahe[i, :, :] = clahe.apply(X_train_clahe[i, :, :])
    # normalize image intensities
    X_train_clahe = (X_train_clahe / 255) * 2 - 1  # normalize intensity
    return X_train_clahe.reshape((X_train_clahe.shape[0], 32, 32, 1))


def augment(X_train, y_train):
    print("first shape", X_train.shape[0])
    X_train_augmented = np.zeros((3 * X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    y_train_augmented = np.zeros((3 * y_train.shape[0]))
    X_train_augmented[0:X_train.shape[0], :, :] = X_train
    y_train_augmented[0:y_train.shape[0]] = y_train
    aug_index = X_train.shape[0]
    for i in range(X_train.shape[0]):
        image = X_train[i]
        label = y_train[i]
        #image_fx = cv2.flip(image, 0)
        #image_fy = cv2.flip(image, 1)
        #image_fxy = cv2.flip(image_fx, 1)
        image1 = rnd_scale(rnd_trans(rnd_rot(rnd_shear(image))))
        #image_fx = rnd_scale(rnd_trans(rnd_rot(rnd_shear(image_fx))))
        #image_fy = rnd_scale(rnd_trans(rnd_rot(rnd_shear(image_fy))))
        #image_fxy = rnd_scale(rnd_trans(rnd_rot(rnd_shear(image_fxy))))
        X_train_augmented[aug_index, :, :] = image1
        #X_train_augmented[aug_index + 1, :, :] = image_fx
        #X_train_augmented[aug_index + 2, :, :] = image_fy
        #X_train_augmented[aug_index + 3, :, :] = image_fxy
        y_train_augmented[aug_index] = y_train[i]
        #y_train_augmented[aug_index + 1] = y_train[i]
        #y_train_augmented[aug_index + 2] = y_train[i]
        #y_train_augmented[aug_index + 3] = y_train[i]
        image2 = rnd_scale(rnd_trans(rnd_rot(rnd_shear(image))))
        X_train_augmented[aug_index + 1, :, :] = image2
        y_train_augmented[aug_index + 1] = y_train[i]
        aug_index += 2

        '''
        plt.figure(figsize=(5, 5))
        show_image((2, 2, 1), "image", image)
        show_image((2, 2, 2), "image", image_fx)
        show_image((2, 2, 3), "image", image_fy)
        show_image((2, 2, 4), "image", image_fxy)
        plt.show()
        plt.close()
        '''
    return [X_train_augmented, y_train_augmented]

def rnd_trans(image):
    x = np.round(np.random.rand(1)[0] * 4 - 2)  # random pixel value between -2 and 2
    y = np.round(np.random.rand(1)[0] * 4 - 2)  # random pixel value between -2 and 2
    matrix = np.array([[1, 0, x], [0, 1, y]])
    new_image = cv2.warpAffine(image, matrix, dsize=image.shape)
    #show_image((1, 1, 1), "image", new_image)
    return new_image


def rnd_rot(image):
    angle = 30*np.random.rand(1)[0] - 15  # random angle between -15 and 15 degrees
    matrix = cv2.getRotationMatrix2D((16, 16), angle, 1)
    new_image = cv2.warpAffine(image, matrix, dsize=image.shape)
    #show_image((1, 1, 1), "image", new_image)
    return new_image


def rnd_scale(image):
    scale = 1.1 #np.random.rand(1)[0]*0.2 + 0.9  # random scale betw 0.9 and 1.1
    matrix = cv2.getRotationMatrix2D((16, 16), 0, scale)
    new_image = cv2.warpAffine(image, matrix, dsize=image.shape)
    #show_image((1, 1, 1), "image", new_image)
    return new_image


def rnd_shear(image):
    cx = 0.5 * np.random.rand(1)[0] - 0.25  # random val betw -0.25 and 0.25
    cy = 0.5 * np.random.rand(1)[0] - 0.25  # random val betw -0.25 and 0.25
    matrix = np.array([[1, cx, 0], [cy, 1, 0]])
    new_image = cv2.warpAffine(image, matrix, dsize=image.shape)
    #show_image((1, 1, 1), "image", new_image)
    return new_image


def MultiScaleCNNArchitecture(x, dropout):
    """
    See "Traffic Sign Recognition with MultiScale Convolutional Neural Networks" by Sermanet, 2011.
    See also https://chatbotslife.com/german-sign-classification-using-deep-learning-neural-networks-98-8-solution-d05656bf51ad.
    I re-implemented an architecture similar to the ones described by Yadav and Sermanet. 
    """
    mu = 0
    sigma = 0.05

    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    layer1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    layer1 = tf.nn.relu(layer1)  # activation
    # Layer 2: Convolutional. Input = 32x32x32. Output = 32x32x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 32), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    layer2 = tf.nn.conv2d(layer1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
    layer2 = tf.nn.relu(layer2)  # activation
    # Layer 3: Max Pooling. Input = 32x32x32. Output = 16x16x32.
    layer3 = tf.nn.max_pool(layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer 4: Dropout: Input = 16x16x32. Output = 16x16x32.
    layer4 = tf.nn.dropout(layer3, dropout)


    # Layer 5: Convolutional. Input = 16x16x32. Output = 16x16x64.
    conv5_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean=mu, stddev=sigma))
    conv5_b = tf.Variable(tf.zeros(64))
    layer5 = tf.nn.conv2d(layer4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
    layer5 = tf.nn.relu(layer5)
    # Layer 6: Convolutional. Input = 16x16x64. Output = 16x16x64.
    conv6_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 64), mean=mu, stddev=sigma))
    conv6_b = tf.Variable(tf.zeros(64))
    layer6 = tf.nn.conv2d(layer5, conv6_W, strides=[1, 1, 1, 1], padding='SAME') + conv6_b
    layer6 = tf.nn.relu(layer6)
    # Layer 7: Max Pooling. Input = 16x16x64. Output = 8x8x64.
    layer7 = tf.nn.max_pool(layer6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer 8: Dropout. Input = 8x8x64. Output = 8x8x64.
    layer8 = tf.nn.dropout(layer7, dropout)

    # Layer 9: Convolutional. Input = 8x8x64. Output = 8x8x128.
    conv9_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 128), mean=mu, stddev=sigma))
    conv9_b = tf.Variable(tf.zeros(128))
    layer9 = tf.nn.conv2d(layer8, conv9_W, strides=[1, 1, 1, 1], padding='SAME') + conv9_b
    layer9 = tf.nn.relu(layer9)
    # Layer 10: Convolutional. Input = 8x8x128. Output = 8x8x128.
    conv10_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 128, 128), mean=mu, stddev=sigma))
    conv10_b = tf.Variable(tf.zeros(128))
    layer10 = tf.nn.conv2d(layer9, conv10_W, strides=[1, 1, 1, 1], padding='SAME') + conv10_b
    layer10 = tf.nn.relu(layer10)
    # Layer 11: Max Pooling: Input = 8x8x128. Output = 4x4x128.
    layer11 = tf.nn.max_pool(layer10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer 12: Dropout. Input = 4x4x128. Output = 4x4x128.
    layer12 = tf.nn.dropout(layer11, dropout)

    # Layer 13: Combining layers 4, 8, and 12 into one flattened layer.
    # Input = 16x16x32, 8x8x64, 4x4x128. Output = 1x14336
    flat_layer4 = tf.contrib.layers.flatten(layer4)
    flat_layer8 = tf.contrib.layers.flatten(layer8)
    flat_layer12 = tf.contrib.layers.flatten(layer12)
    layer13 = tf.concat([flat_layer4, flat_layer8, flat_layer12], axis=1)
    # Layer 14: Fully Connected. Input = 1x14336. Output = 1x1024
    fc14_W = tf.Variable(tf.truncated_normal(shape=(14336, 1024), mean=mu, stddev=sigma))
    fc14_b = tf.Variable(tf.zeros(1024))
    layer14 = tf.matmul(layer13, fc14_W) + fc14_b
    layer14 = tf.nn.relu(layer14)
    # Layer 15: Dropout
    layer15 = tf.nn.dropout(layer14, dropout)
    # Layer 16: Fully Connected: Input = 1x1024. Output = 1x1024.
    fc16_W = tf.Variable(tf.truncated_normal(shape=(1024, 1024), mean=mu, stddev=sigma))
    fc16_b = tf.Variable(tf.zeros(1024))
    layer16 = tf.matmul(layer15, fc16_W) + fc16_b
    layer16 = tf.nn.relu(layer16)
    # Layer 17: Dropout
    layer17 = tf.nn.dropout(layer16, dropout)
    # Layer 18: Fully Connected: Input = 1x1024. Output = 1x42.
    fc18_W = tf.Variable(tf.truncated_normal(shape=(1024, 42), mean=mu, stddev=sigma))
    fc18_b = tf.Variable(tf.zeros(42))
    layer18 = tf.matmul(layer17, fc18_W) + fc18_b

    regularizers = (tf.nn.l2_loss(conv1_W)
                    + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(conv5_W)
                    + tf.nn.l2_loss(conv6_W) + tf.nn.l2_loss(conv9_W)
                    + tf.nn.l2_loss(conv10_W) + tf.nn.l2_loss(fc14_W)
                    + tf.nn.l2_loss(fc16_W) + tf.nn.l2_loss(fc18_W))

    return layer18, regularizers

def MultiScaleCNNArchitectureSmall(x, dropout):
    """
    Smaller version that hopefully fits in memory
    """
    mu = 0
    sigma = 0.05

    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    layer1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
    layer1 = tf.nn.relu(layer1)  # activation
    # Layer 3: Max Pooling. Input = 32x32x32. Output = 16x16x32.
    layer3 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer 4: Dropout: Input = 16x16x32. Output = 16x16x32.
    layer4 = tf.nn.dropout(layer3, dropout)


    # Layer 5: Convolutional. Input = 16x16x32. Output = 16x16x64.
    conv5_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 32, 64), mean=mu, stddev=sigma))
    conv5_b = tf.Variable(tf.zeros(64))
    layer5 = tf.nn.conv2d(layer4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b
    layer5 = tf.nn.relu(layer5)
    # Layer 7: Max Pooling. Input = 16x16x64. Output = 8x8x64.
    layer7 = tf.nn.max_pool(layer5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer 8: Dropout. Input = 8x8x64. Output = 8x8x64.
    layer8 = tf.nn.dropout(layer7, dropout)

    # Layer 9: Convolutional. Input = 8x8x64. Output = 8x8x128.
    conv9_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 128), mean=mu, stddev=sigma))
    conv9_b = tf.Variable(tf.zeros(128))
    layer9 = tf.nn.conv2d(layer8, conv9_W, strides=[1, 1, 1, 1], padding='SAME') + conv9_b
    layer9 = tf.nn.relu(layer9)
    # Layer 11: Max Pooling: Input = 8x8x128. Output = 4x4x128.
    layer11 = tf.nn.max_pool(layer9, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer 12: Dropout. Input = 4x4x128. Output = 4x4x128.
    layer12 = tf.nn.dropout(layer11, dropout)

    # Layer 13: Combining layers 4, 8, and 12 into one flattened layer.
    # Input = 16x16x32, 8x8x64, 4x4x128. Output = 1x14336
    flat_layer4 = tf.contrib.layers.flatten(layer4)
    flat_layer8 = tf.contrib.layers.flatten(layer8)
    flat_layer12 = tf.contrib.layers.flatten(layer12)
    layer13 = tf.concat([flat_layer4, flat_layer8, flat_layer12], axis=1)
    # Layer 14: Fully Connected. Input = 1x14336. Output = 1x1024
    fc14_W = tf.Variable(tf.truncated_normal(shape=(14336, 1024), mean=mu, stddev=sigma))
    fc14_b = tf.Variable(tf.zeros(1024))
    layer14 = tf.matmul(layer13, fc14_W) + fc14_b
    layer14 = tf.nn.relu(layer14)
    # Layer 15: Dropout
    layer15 = tf.nn.dropout(layer14, dropout)
    # Layer 16: Fully Connected: Input = 1x1024. Output = 1x1024.
    fc16_W = tf.Variable(tf.truncated_normal(shape=(1024, 1024), mean=mu, stddev=sigma))
    fc16_b = tf.Variable(tf.zeros(1024))
    layer16 = tf.matmul(layer15, fc16_W) + fc16_b
    layer16 = tf.nn.relu(layer16)
    # Layer 17: Dropout
    layer17 = tf.nn.dropout(layer16, dropout)
    # Layer 18: Fully Connected: Input = 1x1024. Output = 1x42.
    fc18_W = tf.Variable(tf.truncated_normal(shape=(1024, 42), mean=mu, stddev=sigma))
    fc18_b = tf.Variable(tf.zeros(42))
    layer18 = tf.matmul(layer17, fc18_W) + fc18_b

    regularizers = (tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv5_W)
                    + tf.nn.l2_loss(conv9_W)
                    + tf.nn.l2_loss(fc14_W)
                    + tf.nn.l2_loss(fc16_W) + tf.nn.l2_loss(fc18_W))

    return layer18, regularizers

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
    conv1 = tf.nn.relu(conv1)
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
    conv2 = tf.nn.relu(conv2)
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
    conv32_active = tf.nn.relu(conv32)
    # Flattening
    conv32_active_flat = tf.contrib.layers.flatten(conv32_active)

    # From Layer 2: Input = 5x5x108. Output = 2700
    conv2_flat = tf.contrib.layers.flatten(conv2)

    # From Layer 1: Input = 14x14x108. Output = 1x21168.
    conv1_flat = tf.contrib.layers.flatten(conv1)

    # Combine from Layer 1 and from Layer 2. Output = 1x24840
    concat = tf.concat([conv32_active_flat, conv2_flat, conv1_flat], axis=1)

    # Fully Connected. Input = 1x24840. Output = 1x100.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(24840, 100), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(100))
    fc1 = tf.matmul(concat, fc1_W) + fc1_b
    # Activation
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Fully Connected. Input = 1x100. Output = 1x42.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(100, 42), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(42))
    logits = tf.matmul(fc1, fc2_W) + fc2_b

    regularizers = (tf.nn.l2_loss(conv1_W)
                    + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(conv32_W)
                    + tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc2_W))

    return logits, regularizers
