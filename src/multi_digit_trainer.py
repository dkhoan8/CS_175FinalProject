from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import sys
import os
import tensorflow as tf

from Data_harvest import load_local_svhn_data
from Model import regression_head
from tensorflow.core.protobuf import saver_pb2

from datetime import datetime

# Run Options
BATCH_SIZE = 32
NUM_EPOCHS = 128
GRAPH_SUMMARY = '/tmp/svhn_regression_logs'
SAVE_FILE = 'regression.ckpt'

# Image Settings
_inputH = 64
_inputW = 64
_chnls = 3

# Label Settings
_labels = 11
_label_Length = 6

# LEARING RATE HYPER PARAMS
LEARN_RATE = 0.075
DECAY_RATE = 0.95
STAIRCASE = True


def prepare_log_dir():
    if tf.gfile.Exists(GRAPH_SUMMARY):
        tf.gfile.DeleteRecursively(GRAPH_SUMMARY)
    tf.gfile.MakeDirs(GRAPH_SUMMARY)


def fill_feed_dict(data, labels, x, y_, step):
    set_size = labels.shape[0]
    offset = (step * BATCH_SIZE) % (set_size - BATCH_SIZE)
    batch_data = data[offset:(offset + BATCH_SIZE), ...]
    batch_labels = labels[offset:(offset + BATCH_SIZE)]
    return {x: batch_data, y_: batch_labels}


def multi_digit_trainer(train_data, train_labels,
						valid_data, valid_labels,
						test_data, test_labels,
						train_size, saved_weights_path):
    _gStep = tf.Variable(0, trainable=False)

    with tf.name_scope('input'):
        images_placeholder = tf.placeholder(tf.float32,
                                            shape=(BATCH_SIZE, _inputH,
                                                   _inputW, _chnls))

    with tf.name_scope('image'):
        tf.summary.image('input', images_placeholder, 10)

    labels_placeholder = tf.placeholder(tf.int32,
                                        shape=(BATCH_SIZE, _label_Length))

    [logits_1, logits_2, logits_3, logits_4, logits_5] = regression_head(images_placeholder, True)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_1, labels = labels_placeholder[:, 1])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_2, labels = labels_placeholder[:, 2])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_3, labels = labels_placeholder[:, 3])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_4, labels = labels_placeholder[:, 4])) +\
        tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_5, labels = labels_placeholder[:, 5]))

    learning_rate = tf.train.exponential_decay(LEARN_RATE, _gStep*BATCH_SIZE, train_size, DECAY_RATE)
    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('train'):
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=_gStep)

    prediction = tf.stack([tf.nn.softmax(regression_head(images_placeholder)[0]),
                                tf.nn.softmax(regression_head(images_placeholder)[1]),
                                tf.nn.softmax(regression_head(images_placeholder)[2]),
                                tf.nn.softmax(regression_head(images_placeholder)[3]),
                                tf.nn.softmax(regression_head(images_placeholder)[4])])

    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

    start_time = time.time()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        init_op = tf.global_variables_initializer()

        if(saved_weights_path):
            saver.restore(sess, saved_weights_path)
        print("Model restored.")

        reader = tf.train.NewCheckpointReader("classifier.ckpt")
        reader.get_variable_to_shape_map()
        sess.run(init_op)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                best = tf.transpose(prediction, [1, 2, 0])
                lb = tf.cast(labels_placeholder[:, 1:6], tf.int64)
                correct_prediction = tf.equal(tf.argmax(best, 1), lb)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32)) / prediction.get_shape().as_list()[1] / prediction.get_shape().as_list()[0]
            tf.summary.scalar('accuracy', accuracy)

        final_merge = tf.summary.merge_all()
        trWriter = tf.summary.FileWriter(GRAPH_SUMMARY + '/train', sess.graph)
        valWriter = tf.summary.FileWriter(GRAPH_SUMMARY + '/validation')

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
            duration = time.time() - start_time
            examples_per_sec = BATCH_SIZE / duration
            tr_feed_dict = fill_feed_dict(train_data, train_labels, images_placeholder, labels_placeholder, step)
            _, l, lr, acc, predictions = sess.run([optimizer, loss, learning_rate,
                                                  accuracy, prediction],
                                                  feed_dict=tr_feed_dict)

            tr_batched_labels = tr_feed_dict.values()

            if step % 10 == 0:
                val_feed_dict = fill_feed_dict(valid_data, valid_labels, images_placeholder, labels_placeholder, step)
                valid_batch_labels = val_feed_dict.values()

                valSummary, _, l, lr, valid_acc = sess.run([final_merge, optimizer, loss, learning_rate, accuracy],
                feed_dict=val_feed_dict, options=run_options, run_metadata=run_metadata)
                print('---------------------------------------------------------------')
                print('Validation Accuracy: %.2f' % valid_acc)
                valWriter.add_run_metadata(run_metadata, 'step%03d' % step)
                valWriter.add_summary(valSummary, step)

                trSummary, _, l, lr, trAccuracy = sess.run([final_merge, optimizer, loss, learning_rate, accuracy],
                    feed_dict=tr_feed_dict)
                trWriter.add_run_metadata(run_metadata, 'step%03d' % step)
                trWriter.add_summary(trSummary, step)
                print('Training Set Accuracy: %.2f' % trAccuracy)
                if(step % 1000 == 0):
                    print('Current milestone is at: ', step)
                    # save_path = saver.save(sess, SAVE_FILE)
                    # print("Checkpoint saved in file: %s" % save_path)
                print('---------------------------------------------------------------')

            elif step % 2 == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                format_str = ('%s: step %d, loss = %.2f  learning rate = %.2f  (%.1f examples/sec; %.3f ''sec/batch)')
                print('Minibatch accuracy: %.2f' % acc)
                sys.stdout.flush()

        test_feed_dict = fill_feed_dict(test_data, test_labels, images_placeholder, labels_placeholder, step)
        _, l, lr, test_acc = sess.run([optimizer, loss, learning_rate, accuracy], feed_dict=test_feed_dict, options=run_options, run_metadata=run_metadata)
        print('Test accuracy: %.2f' % test_acc)
        save_path = saver.save(sess, SAVE_FILE)
        print("Model saved in file: %s" % save_path)
        trWriter.close()
        valWriter.close()


def main(saved_weights_path):
    prepare_log_dir()
    train_data, train_labels = load_local_svhn_data("train", "full")
    valid_data, valid_labels = load_local_svhn_data("valid", "full")
    test_data, test_labels = load_local_svhn_data("test", "full")

    print("Train Data Dim", train_data.shape)
    print("Valid Data Dim", valid_data.shape)
    print("Test Data Dim", test_data.shape)

    train_size = len(train_labels)
    multi_digit_trainer(train_data, train_labels, valid_data, valid_labels,
                    test_data, test_labels, train_size, saved_weights_path)


if __name__ == '__main__':
    saved_weights_path = None
    if len(sys.argv) > 1:
        print("Loading Saved Checkpoints From:", sys.argv[1])
        if os.path.isfile(sys.argv[1]):
            saved_weights_path = sys.argv[1]
        else:
            raise EnvironmentError("I'm afraid I can't load that file.")
    else:
        print("Starting without Saved Weights.")
    main(saved_weights_path)
