# Load pickled data
import pickle
import multiprocessing
import os
import helpers
import numpy as np
import cv2
import random

def preprocessing(X_train):
    # image preprocessing
    # grayscaling
    X_train_gray = X_train[:, :, :, 0]
    for i in range(X_train.shape[0]):
        X_train_gray[i, :, :] = cv2.cvtColor(X_train[i, :, :, :], cv2.COLOR_RGB2GRAY)
    # contrast limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    X_train_preproc = X_train_gray
    for i in range(X_train.shape[0]):
        X_train_preproc[i, :, :] = clahe.apply(X_train_preproc[i, :, :])
    # normalize image intensities
    X_train_preproc = 2.0*(X_train_preproc / 255) - 1.0  # normalize intensity
    return X_train_preproc.reshape((X_train_preproc.shape[0], 32, 32, 1))


def augment(X, y, n_classes):
    def rnd_blur(image):
        kernel = random.randint(0,5)
        if kernel == 0 or kernel == 2:
            return image
        if kernel == 1:
            return cv2.GaussianBlur(image, (1, 1), 0)
        if kernel == 3:
            return cv2.GaussianBlur(image, (3, 3), 0)
        if kernel == 5:
            return cv2.GaussianBlur(image, (5, 5), 0)
        if kernel == 4:
            return cv2.GaussianBlur(image, (7, 7), 0)

    def rnd_brightness(image):
        scale = np.random.rand(1)[0] * 0.4 + 0.8  # random scale betw 0.8 and 1.2
        return image * scale

    def rnd_trans(image):
        x = np.round(np.random.rand(1)[0] * 4 - 2)  # random pixel value between -2 and 2
        y = np.round(np.random.rand(1)[0] * 4 - 2)  # random pixel value between -2 and 2
        matrix = np.array([[1, 0, x], [0, 1, y]])
        new_image = cv2.warpAffine(image, matrix, dsize=image.shape)
        # show_image((1, 1, 1), "image", new_image)
        return new_image

    def rnd_rot(image):
        angle = 40 * np.random.rand(1)[0] - 20  # random angle between -20 and 20 degrees
        matrix = cv2.getRotationMatrix2D((16, 16), angle, 1)
        new_image = cv2.warpAffine(image, matrix, dsize=image.shape)
        # show_image((1, 1, 1), "image", new_image)
        return new_image

    def rnd_scale(image):
        scale = np.random.rand(1)[0]*0.4 + 0.8  # random scale betw 0.8 and 1.2
        matrix = cv2.getRotationMatrix2D((16, 16), 0, scale)
        new_image = cv2.warpAffine(image, matrix, dsize=image.shape)
        # show_image((1, 1, 1), "image", new_image)
        return new_image

    def rnd_shear(image):
        cx = 0.5 * np.random.rand(1)[0] - 0.25  # random val betw -0.25 and 0.25
        cy = 0.5 * np.random.rand(1)[0] - 0.25  # random val betw -0.25 and 0.25
        matrix = np.array([[1, cx, 0], [cy, 1, 0]])
        new_image = cv2.warpAffine(image, matrix, dsize=image.shape)
        # show_image((1, 1, 1), "image", new_image)
        return new_image

    X_augmented = X
    y_augmented = y
    for sign_code in range(n_classes + 1):
        image_list = list(np.where(y == sign_code)[0])
        if len(image_list) < 500:  # augment the classes that have few examples
            print("Augmenting class with under 500 examples. Class #", sign_code, " of ", n_classes)
            for image_index in image_list:
                # add 4 extra randomly perturbed images to dataset
                image = X[image_index, :, :].reshape(32, 32)

                transformed_image = rnd_scale(rnd_shear(rnd_rot(rnd_trans(rnd_blur(rnd_brightness(image)))))).reshape(1,32,32,1)
                X_augmented = np.concatenate((X_augmented, transformed_image))
                y_augmented = np.concatenate((y_augmented, np.array([sign_code])))

                transformed_image = rnd_scale(rnd_shear(rnd_rot(rnd_trans(rnd_blur(rnd_brightness(image)))))).reshape(1,32,32,1)
                X_augmented = np.concatenate((X_augmented, transformed_image))
                y_augmented = np.concatenate((y_augmented, np.array([sign_code])))

                transformed_image = rnd_scale(rnd_shear(rnd_rot(rnd_trans(rnd_blur(rnd_brightness(image)))))).reshape(1,32,32,1)
                X_augmented = np.concatenate((X_augmented, transformed_image))
                y_augmented = np.concatenate((y_augmented, np.array([sign_code])))

                transformed_image = rnd_scale(rnd_shear(rnd_rot(rnd_trans(rnd_blur(rnd_brightness(image)))))).reshape(1,32,32,1)
                X_augmented = np.concatenate((X_augmented, transformed_image))
                y_augmented = np.concatenate((y_augmented, np.array([sign_code])))
        elif len(image_list) < 1000:
            print("Augmenting class with under 1000 examples. Class #", sign_code, " of ", n_classes)
            for image_index in image_list:
                # add 2 extra randomly perturbed images to dataset
                image = X[image_index, :, :].reshape(32, 32)

                transformed_image = rnd_scale(rnd_shear(rnd_rot(rnd_trans(rnd_blur(rnd_brightness(image)))))).reshape(1,32,32,1)
                X_augmented = np.concatenate((X_augmented, transformed_image))
                y_augmented = np.concatenate((y_augmented, np.array([sign_code])))

                transformed_image = rnd_scale(rnd_shear(rnd_rot(rnd_trans(rnd_blur(rnd_brightness(image)))))).reshape(1,32,32,1)
                X_augmented = np.concatenate((X_augmented, transformed_image))
                y_augmented = np.concatenate((y_augmented, np.array([sign_code])))

    return X_augmented, y_augmented

if __name__ == "__main__":
    print("LOADING RAW DATA")
    # Load data
    training_file = 'traffic-signs-data/train.p'
    validation_file = 'traffic-signs-data/valid.p'
    testing_file = 'traffic-signs-data/test.p'
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    # Show data statistics.
    print("BEFORE AUGMENTATION")
    n_train = X_train.shape[0]
    n_validation = X_valid.shape[0]
    n_test = X_test.shape[0]
    image_shape = X_train.shape[1:]
    n_classes = max(y_test) - min(y_test) + 1
    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    X_train_preproc = preprocessing(X_train)  # preprocessing
    X_valid_preproc = preprocessing(X_valid)  # preprocessing
    X_test_preproc = preprocessing(X_test)  # preprocessing
    [X_train_preproc, y_train_preproc] = augment(X_train_preproc, y_train, n_classes)  # augmentation

    # Show data statistics.
    print("AFTER AUGMENTATION")
    n_train = X_train_preproc.shape[0]
    n_validation = X_valid_preproc.shape[0]
    n_test = X_test_preproc.shape[0]
    image_shape = X_train_preproc.shape[1:]
    n_classes = max(y_test) - min(y_test) + 1
    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_validation)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    print("SAVING AUGMENTED DATA TO DISK")
    train_preproc = {'features': X_train_preproc, 'labels': y_train_preproc}
    valid_preproc = {'features': X_valid_preproc, 'labels': y_valid}
    test_preproc = {'features': X_test_preproc, 'labels': y_test}
    pickle.dump(train_preproc, open("traffic-signs-data/train_preproc_clahe_data.p", "wb"))
    pickle.dump(valid_preproc, open("traffic-signs-data/valid_preproc_clahe_data.p", "wb"))
    pickle.dump(test_preproc, open("traffic-signs-data/test_preproc_clahe_data.p", "wb"))
    print("DONE")
