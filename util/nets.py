import tensorflow as tf
from util import layers


def build_generator(input_tensor):
    """
    To build the generative network
    :param input_tensor: D*128
    :return:
    """
    weights_initializer = tf.random_normal_initializer(stddev=0.02)
    biases_initializer = tf.constant_initializer(0.)
    # t_conv_1: N*2H*2W*128
    t_conv_1 = tf.layers.conv2d_transpose(input_tensor, 128, 3, strides=(1, 1), activation=tf.nn.relu,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    # t_conv_2: N*4H*4W*64
    t_conv_2 = tf.layers.conv2d_transpose(t_conv_1, 128, 3, strides=(2, 2), padding='SAME', activation=tf.nn.relu,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    # t_conv_3: N*8H*8W*32
    t_conv_3 = tf.layers.conv2d_transpose(t_conv_2, 64, 3, strides=(2, 2), padding='SAME', activation=tf.nn.relu,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    # t_conv_4: N*16H*16W*16
    t_conv_4 = tf.layers.conv2d_transpose(t_conv_3, 64, 3, strides=(2, 2), padding='SAME', activation=tf.nn.relu,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    # t_conv_5: N*32H*32W*3
    t_conv_5 = tf.layers.conv2d_transpose(t_conv_4, 3, 3, strides=(2, 2), padding='SAME',
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    return t_conv_5


def build_generator_new(input_tensor):
    weights_initializer = tf.random_normal_initializer(stddev=0.02)
    biases_initializer = tf.constant_initializer(0.)
    fc_1 = layers.fc_layer('fc_0', input_tensor, 4 * 4 * 256, weights_initializer=weights_initializer,
                           biases_initializer=biases_initializer)
    fc_1 = tf.reshape(fc_1, [-1, 4, 4, 256])
    fc_1 = tf.nn.relu(tf.layers.batch_normalization(fc_1, momentum=0.9, epsilon=1e-5))

    t_conv_1 = tf.layers.conv2d_transpose(fc_1, 512, 5, (2, 2), padding='SAME',
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    t_conv_1 = tf.nn.relu(tf.layers.batch_normalization(t_conv_1, momentum=0.9, epsilon=1e-5))
    t_conv_2 = tf.layers.conv2d_transpose(t_conv_1, 256, 5, (2, 2), padding='SAME',
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    t_conv_2 = tf.nn.relu(tf.layers.batch_normalization(t_conv_2, momentum=0.9, epsilon=1e-5))
    t_conv_3 = tf.layers.conv2d_transpose(t_conv_2, 128, 5, (2, 2), padding='SAME',
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    t_conv_3 = tf.nn.relu(tf.layers.batch_normalization(t_conv_3, momentum=0.9, epsilon=1e-5))

    t_conv_4 = tf.layers.conv2d_transpose(t_conv_3, 3, 5, (2, 2), padding='SAME', activation=tf.nn.tanh,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    return t_conv_4


def net_discriminator(input_tensor, output_dim=1, keep_prob=0.5):
    """
    To build the discriminative network
    :param input_tensor:
    :param keep_prob:
    :return:
    """
    conv_1 = layers.conv_relu_layer('conv_1', input_tensor, kernel_size=3, stride=1, output_dim=32)
    pool_1 = layers.pooling_layer('pool_1', conv_1, kernel_size=2, stride=1)
    conv_2 = layers.conv_relu_layer('conv_2', pool_1, kernel_size=3, stride=1, output_dim=32)
    pool_2 = layers.pooling_layer('pool_2', conv_2, kernel_size=2, stride=1)
    conv_3 = layers.conv_relu_layer('conv_3', pool_2, kernel_size=3, stride=1, output_dim=32)
    pool_3 = layers.pooling_layer('pool_3', conv_3, kernel_size=4, stride=1)
    fc_1 = layers.fc_relu_layer('fc_1', pool_3, output_dim=512)
    fc_1 = tf.nn.dropout(fc_1, keep_prob=keep_prob)
    fc_2 = layers.fc_relu_layer('fc_2', fc_1, output_dim=512)
    fc_2 = tf.nn.dropout(fc_2, keep_prob=keep_prob)
    fc_3 = layers.fc_layer('fc_3', fc_2, output_dim=output_dim)

    return fc_3


def net_discriminator_new(input_tensor, output_dim=1):
    def bn(in_tensor):
        return tf.layers.batch_normalization(in_tensor, momentum=0.9, epsilon=1e-5)

    def leaky_relu(in_tensor, leak=0.2):
        return tf.maximum(in_tensor, leak * in_tensor)

    starting_out_dim = 64
    kernel_size = 5
    stride = 2
    conv_1 = layers.conv_relu_layer('conv_1', input_tensor, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim)
    conv_1 = leaky_relu(conv_1)
    conv_2 = layers.conv_relu_layer('conv_2', conv_1, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 2)
    conv_2 = leaky_relu(bn(conv_2))
    conv_3 = layers.conv_relu_layer('conv_3', conv_2, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 4)
    conv_3 = leaky_relu(bn(conv_3))
    conv_4 = layers.conv_relu_layer('conv_4', conv_3, kernel_size=kernel_size, stride=stride,
                                    output_dim=starting_out_dim * 8)
    conv_4 = leaky_relu(bn(conv_4))
    fc_d = layers.fc_layer('fc_d', conv_4, output_dim=output_dim)
    return fc_d
