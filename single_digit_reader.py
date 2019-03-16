import matplotlib
matplotlib.use('Agg')

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Model import classification_head
from numpy import array
from PIL import ImageOps, Image

WEIGHTS_FILE = "classifier.ckpt"
SIZE = (32,32)

def detect(img_path, saved_model_weights):
    img = Image.open(img_path)
    
    print("-------------------Attemped resize img------------------------------")    
    fit_and_resized_image = ImageOps.fit(img, SIZE, Image.ANTIALIAS)
    plt.imshow(fit_and_resized_image)
    plt.show()

    X = tf.placeholder(tf.float32, shape=(1, 32, 32, 3))
    prediction = tf.nn.softmax(classification_head(X))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("hello")
        print("Loading Saved Checkpoints From:", WEIGHTS_FILE)

        saver.restore(sess, saved_model_weights)
        print("Model restored: ")

        pix = np.array(fit_and_resized_image)
        norm_img = (255-pix)*1.0/255.0
        exp = np.expand_dims(norm_img, axis=0)

        feed_dat = {X: exp}
        predictions = sess.run(prediction, feed_dict=feed_dat)
        print("Best Prediction is:", np.argmax(predictions))

if __name__ == "__main__":
    img_path = None
    if len(sys.argv) > 1:
        print("Reading Image file:", sys.argv[1])
        if os.path.isfile(sys.argv[1]):
            img_path = sys.argv[1]
        else:
            raise EnvironmentError("Cannot open image file.")
    else:
        raise EnvironmentError("You must pass an image file to process")

    if os.path.isfile(WEIGHTS_FILE):
        saved_model_weights = WEIGHTS_FILE
    else:
        raise IOError("Cannot find checkpoint file. Please run train_classifier.py")

    detect(img_path, saved_model_weights)