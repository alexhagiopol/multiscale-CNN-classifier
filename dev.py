# Load pickled data
import helpers
import pickle
import tensorflow as tf
import cv2
import sklearn as skl
import numpy as np

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
n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]
image_shape = X_train.shape[1:]
n_classes = max(y_test) - min(y_test)
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# Hyperparameters
EPOCHS = 15
BATCH_SIZE = 128
rate = 0.0002
dropout = 0.50

# Set up TensorFlow input and output
x = tf.placeholder(tf.float32, (None, 32, 32, 1))  # floats for normalized data
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 42)
keep_prob = tf.placeholder(tf.float32)
logits, regularizers = helpers.MultiScaleCNNArchitecture(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy) + 1e-5 * regularizers
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

# model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

X_train_preproc = helpers.preprocessing(X_train)  # CLAHE = contrast limited adaptive histogram
X_valid_preproc = helpers.preprocessing(X_valid)
X_test_preproc = helpers.preprocessing(X_test)

# I experimented with augmentation. Unfortunately I did not figure out how to fit all of this
# data into memory on my hardware :(.
# [X_train_preproc, y_train] = helpers.augment(X_train_preproc, y_train)

# training
max_accuracy = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = X_train.shape[0]

    print("Training with batch size ", BATCH_SIZE)
    print()
    for i in range(EPOCHS):
        X_train_preproc, y_train = skl.utils.shuffle(X_train_preproc, y_train)
        # process each batch
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_preproc[offset:end], y_train[offset:end]
            _, loss = sess.run([training_operation, loss_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})  # execute session
        validation_accuracy = helpers.evaluate(X_valid_preproc, y_valid, accuracy_operation, BATCH_SIZE, x, y, keep_prob)
        print("EPOCH {} ...".format(i + 1), " validation accuracy = {:.3f}".format(validation_accuracy))
        if validation_accuracy > max_accuracy:  # save only highest accuracy we've achieved so far
            max_accuracy = validation_accuracy
            saver.save(sess, './best_model_save_file')
            print("Highest accuracy seen so far. Model saved.")
        print()

