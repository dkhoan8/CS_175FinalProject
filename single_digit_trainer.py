from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Data_harvest import load_local_svhn_data
from Model import classification_head
from datetime import datetime
from tensorflow.core.protobuf import saver_pb2

import time
import numpy as np
import sys
import os
import tensorflow as tf



GRAPH_SUMMARY = '/tmp/svhn_classifier_logs'
NUM_LABELS = 10
_R = 32
_C = 32
_Chnls = 3
SAVE_FILE = "classifier.ckpt"

BATCH_SIZE = 256
NUM_EPOCHS = 128

LEARN_RATE = 0.075
DECAY_RATE = 0.95
STAIRCASE = True

def clean_log():
    if tf.gfile.Exists(GRAPH_SUMMARY):
        tf.gfile.DeleteRecursively(GRAPH_SUMMARY)
    tf.gfile.MakeDirs(GRAPH_SUMMARY)


def feed_data_label(data, labels, x, y_, step):
    a = (step * BATCH_SIZE) % (labels.shape[0] - BATCH_SIZE)
    _data = data[a:(a + BATCH_SIZE), ...]
    _label = labels[a:(a + BATCH_SIZE)]
    return {x: _data, y_: _label}

def scopeOfInput(): 
    _x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, _R, _C, _Chnls], name="Images_Input")
    _y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS], name="Labels_Input")
    return _x,_y

def train_single_digit_mod(train_data, train_labels,
                         valid_data, valid_labels,
                         test_data, test_labels,
                         train_size, saved_weights_path):

    _gStep = tf.Variable(0, trainable=False)

    with tf.name_scope('input'):
        _imgStorage, _labelStorage = scopeOfInput()

    with tf.name_scope('image'):
        tf.summary.image('train_input', _imgStorage, 10)

    logits = classification_head(_imgStorage, train=True)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = _labelStorage))
        tf.summary.scalar('loss', loss)

    learning_rate = tf.train.exponential_decay(LEARN_RATE, _gStep*BATCH_SIZE, train_size, DECAY_RATE, staircase=STAIRCASE)
    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('train'):
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step =_gStep)

    train_prediction = tf.nn.softmax(classification_head(_imgStorage, train=False))
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

        if(saved_weights_path):
            saver.restore(sess, saved_weights_path)
            print("Model Found.")

        sess.run(init_op)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(train_prediction, 1),
                                              tf.argmax(_labelStorage, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

        run_options = tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            feed_dict = feed_data_label(train_data, train_labels,
                                       _imgStorage, _labelStorage,
                                       step)
            _, _, _, acc = sess.run([optimizer, loss, learning_rate, accuracy], feed_dict=feed_dict)
            if step % 100 == 0:
                valid_feed_dict = feed_data_label(valid_data, valid_labels,
                                                  _imgStorage,
                                                  _labelStorage, step)
                _, _, _, _, valid_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy], feed_dict=valid_feed_dict, options=run_options, run_metadata=run_metadata)
                print('--------------------------------------------------------------------------')
                print('Validation Accuracy: %.2f%%' % valid_acc)

                _, _, _, _, train_acc = sess.run([merged, optimizer, loss, learning_rate, accuracy],
                    feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                print('Training Accuracy: %.2f%%' % train_acc)

                if step % 1000:
                    print('Number of steps taken: ', step)
                print('--------------------------------------------------------------------------')

            elif step % 10 == 0:
                print('Mini-Batch Accuracy: %.2f%%' % acc)
                sys.stdout.flush()

        # Save the variables to disk.
        save_path = saver.save(sess, SAVE_FILE)
        print("Model saved in file: %s" % save_path)

        test_feed_dict = feed_data_label(test_data, test_labels, _imgStorage, _labelStorage, step)
        summary, acc = sess.run([merged, accuracy], feed_dict=test_feed_dict)
        print('Test Accuracy: %.5f%%' % acc)


def main(saved_weights_path):
    clean_log()
    train_data, train_labels = load_local_svhn_data("train", "cropped")
    valid_data, valid_labels = load_local_svhn_data("valid", "cropped")
    test_data, test_labels = load_local_svhn_data("test", "cropped")

    print("Training Data Dim", train_data.shape)
    print("Valid Data Dim", valid_data.shape)
    print("Test Data Dim", test_data.shape)

    train_size = train_labels.shape[0]
    saved_weights_path = None
    train_single_digit_mod(train_data, train_labels,
                         valid_data, valid_labels,
                         test_data, test_labels, train_size,
                         saved_weights_path)


if __name__ == '__main__':
    saved_weights_path = None
    if len(sys.argv) > 1:
        print("Loading Saved Checkpoints From:", sys.argv[1])
        if os.path.isfile(sys.argv[1]):
            saved_weights_path = sys.argv[1]
        else:
            raise EnvironmentError("The weights file cannot be opened.")
    else:
        print("Starting without Saved Weights.")
    main(saved_weights_path)
