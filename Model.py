import tensorflow as tf
from pdb import set_trace as bp

CHANNEL_COUNT = 3
CL_LABELS_COUNT = 10
LABELS_COUNT = CL_LABELS_COUNT + 1

PATH_MATRIX = 5
DEPTH_LAYER_1 = 48
DEPTH_LAYER_2 = 64
DEPTH_LAYER_3 = 128
DEPTH_LAYER_4 = 160

DROPOUT_THRESHOLD = 0.85

W_conv1 = W_conv2 = W_conv3 = W_conv4 = 0
B_conv1 = B_conv2 = B_conv3 = B_conv4 = 0
W_conv1 = tf.get_variable("Weights_1", shape=[PATH_MATRIX, PATH_MATRIX,
                                CHANNEL_COUNT, DEPTH_LAYER_1])
B_conv1 = tf.Variable(tf.constant(0.0, shape=[DEPTH_LAYER_1]), name='Biases_1')

W_conv2 = tf.get_variable("Weights_2", shape=[PATH_MATRIX, PATH_MATRIX,
                                DEPTH_LAYER_1, DEPTH_LAYER_2])
B_conv2 = tf.Variable(tf.constant(0.1, shape=[DEPTH_LAYER_2]), name='Biases_2')

W_conv3 = tf.get_variable("Weights_3", shape=[PATH_MATRIX, PATH_MATRIX,
                                DEPTH_LAYER_2, DEPTH_LAYER_3])
B_conv3 = tf.Variable(tf.constant(0.1, shape=[DEPTH_LAYER_3]), name='Biases_3')

W_conv4 = tf.get_variable("Weights_4", shape=[PATH_MATRIX,
                                PATH_MATRIX, DEPTH_LAYER_3, DEPTH_LAYER_4])
B_conv4 = tf.Variable(tf.constant(0.1, shape=[DEPTH_LAYER_4]), name='Biases_4')


W_reg1 = W_reg2 = W_reg3 = W_reg4 = W_reg5 = 0
B_reg1 = B_reg2 = B_reg3 = B_reg4 = B_reg5 = 0
W_reg1 = tf.get_variable("WS1", shape=[DEPTH_LAYER_4, LABELS_COUNT])
B_reg1 = tf.Variable(tf.constant(1.0, shape=[LABELS_COUNT]), name='BS1')

W_reg2 = tf.get_variable("WS2", shape=[DEPTH_LAYER_4, LABELS_COUNT])
B_reg2 = tf.Variable(tf.constant(1.0, shape=[LABELS_COUNT]), name='BS2')

W_reg3 = tf.get_variable("WS3", shape=[DEPTH_LAYER_4, LABELS_COUNT])
B_reg3 = tf.Variable(tf.constant(1.0, shape=[LABELS_COUNT]), name='BS3')

W_reg4 = tf.get_variable("WS4", shape=[DEPTH_LAYER_4, LABELS_COUNT])
B_reg4 = tf.Variable(tf.constant(1.0, shape=[LABELS_COUNT]), name='BS4')

W_reg5 = tf.get_variable("WS5", shape=[DEPTH_LAYER_4, LABELS_COUNT])
B_reg5 = tf.Variable(tf.constant(1.0, shape=[LABELS_COUNT]), name='BS5')


cl_l3_weights = 0
cl_l3_weights = tf.get_variable("Classifer_Weights_1", shape=[DEPTH_LAYER_3, DEPTH_LAYER_4])
cl_l3_biases = 0
cl_l3_biases = tf.Variable(tf.constant(0.05, shape=[DEPTH_LAYER_4]),
                           name='Classifer_Biases_1')

cl_out_weights = 0
cl_out_weights = tf.get_variable("Classifer_Weights_3",
                                 shape=[DEPTH_LAYER_4, CL_LABELS_COUNT])

cl_out_biases = 0
cl_out_biases = tf.Variable(tf.constant(0.05, shape=[CL_LABELS_COUNT]),
                            name='Classifer_Biases_3')


def activation_summary(x):
    tensor_name = 0
    tensor_name = x.op.name


def convolution_model(data):
    with tf.variable_scope('Layer_1', reuse=True) as scope:
        con = 0
        con = tf.nn.conv2d(data, W_conv1,
                           [1, 1, 1, 1], 'VALID', name='C1')
        hid = 0
        hid = tf.nn.relu(con + B_conv1)
        activation_summary(hid)
    
    pol = 0
    pol = tf.nn.max_pool(hid,
                         [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='Pool_1')
    
    lrn = 0
    lrn = tf.nn.local_response_normalization(pol, name="Normalize_1")

    with tf.variable_scope('Layer_2') as scope:
        con = 0
        con = tf.nn.conv2d(lrn, W_conv2,
                           [1, 1, 1, 1], padding='VALID', name='C3')
        hid = 0
        hid = tf.nn.relu(con + B_conv2)
        activation_summary(hid)
    
    pol = 0
    pol = tf.nn.max_pool(hid,
                         [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='Pool_2')
    lrn = 0
    lrn = tf.nn.local_response_normalization(pol, name="Normalize_2")

    with tf.variable_scope('Layer_3') as scope:
        con = 0
        con = tf.nn.conv2d(lrn, W_conv3,
                           [1, 1, 1, 1], padding='VALID', name='C5')
        hid = 0
        hid = tf.nn.relu(con + B_conv3)
        lrn = 0
        lrn = tf.nn.local_response_normalization(hid)

        if lrn.get_shape().as_list()[1] is 1:  
            sub = 0
            sub = tf.nn.max_pool(lrn,
                                 [1, 1, 1, 1], [1, 1, 1, 1], 'SAME', name='S5')
        else:
            sub = 0
            sub = tf.nn.max_pool(lrn,
                                 [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='S5')

        activation_summary(sub)

    return sub


def classification_head(data, keep_prob=1.0, train=False):
    conv_layer = 0
    conv_layer = convolution_model(data)
    shape = 0
    shape = conv_layer.get_shape().as_list()
    dim = 0
    dim = shape[1] * shape[2] * shape[3]

    if train is True:
        print("Using drop out")
        conv_layer = tf.nn.dropout(conv_layer, DROPOUT_THRESHOLD)
    else:
        print("Not using dropout")

    with tf.variable_scope('fully_connected_1') as scope:
        fc1 = tf.reshape(conv_layer, [shape[0], -1])
        fc1 = tf.add(tf.matmul(fc1, cl_l3_weights), cl_l3_biases)
        fc_out = tf.nn.relu(fc1, name=scope.name)
        activation_summary(fc_out)

    with tf.variable_scope("softmax_linear") as scope:
        logits = tf.matmul(fc_out, cl_out_weights) + cl_out_biases
        activation_summary(logits)

    return logits


def regression_head(data, train=False):
    conv_layer = convolution_model(data)

    with tf.variable_scope('full_connected_1') as scope:
        con = tf.nn.conv2d(conv_layer, W_conv4, [1, 2, 2, 1], padding='VALID', name='C5')
        hid = tf.nn.relu(con + B_conv4)
        activation_summary(hid)
    shape = hid.get_shape().as_list()
    reshape = tf.reshape(hid, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.variable_scope('Output') as scope:
        logits_1 = tf.matmul(reshape, W_reg1) + B_reg1
        logits_2 = tf.matmul(reshape, W_reg2) + B_reg2
        logits_3 = tf.matmul(reshape, W_reg3) + B_reg3
        logits_4 = tf.matmul(reshape, W_reg4) + B_reg4
        logits_5 = tf.matmul(reshape, W_reg5) + B_reg5

    return [logits_1, logits_2, logits_3, logits_4, logits_5]