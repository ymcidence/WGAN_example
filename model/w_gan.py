import tensorflow as tf
import model.layers as layers
import numpy as np
from time import gmtime, strftime

MODE_FLAG_TRAIN = 'Train'
MODE_FLAG_TEST = 'Test'
NAME_SCOPE_GENERATIVE_NET = 'GenerativeNet'
NAME_SCOPE_DISCRIMINATIVE_NET = 'DiscriminativeNet'


def build_generator(input_tensor):
    """
    To build the generative network
    :param input_tensor: D*128
    :return:
    """
    weights_initializer = tf.random_normal_initializer(stddev=0.02)
    biases_initializer = tf.constant_initializer(0.)
    # t_conv_1: N*2H*2W*128
    # t_conv_1 = layers.deconv_relu_layer('tConv1', input_tensor, kernel_size=3, stride=2, output_dim=128)
    t_conv_1 = tf.layers.conv2d_transpose(input_tensor, 128, 3, strides=(2, 2), padding='SAME', activation=tf.nn.relu,
                                          activity_regularizer=tf.layers.batch_normalization,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    # t_conv_2: N*4H*4W*64
    t_conv_2 = tf.layers.conv2d_transpose(t_conv_1, 128, 5, strides=(2, 2), padding='SAME', activation=tf.nn.relu,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    # t_conv_3: N*8H*8W*32
    t_conv_3 = tf.layers.conv2d_transpose(t_conv_2, 64, 5, strides=(2, 2), padding='SAME', activation=tf.nn.relu,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    # t_conv_4: N*16H*16W*16
    t_conv_4 = tf.layers.conv2d_transpose(t_conv_3, 64, 5, strides=(2, 2), padding='SAME', activation=tf.nn.relu,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    # t_conv_5: N*32H*32W*3
    t_conv_5 = tf.layers.conv2d_transpose(t_conv_4, 3, 5, strides=(2, 2), padding='SAME', activation=tf.sigmoid,
                                          bias_initializer=biases_initializer, kernel_initializer=weights_initializer)
    return t_conv_5


def build_discriminator(input_tensor, keep_prob):
    """
    To build the discriminative network
    :param input_tensor:
    :param keep_prob:
    :return:
    """
    conv_1 = layers.conv_relu_layer('Conv1', input_tensor, kernel_size=3, stride=1, output_dim=32)
    pool_1 = layers.pooling_layer('Pool1', conv_1, kernel_size=2, stride=1)
    conv_2 = layers.conv_relu_layer('Conv2', pool_1, kernel_size=3, stride=1, output_dim=32)
    pool_2 = layers.pooling_layer('Pool2', conv_2, kernel_size=2, stride=1)
    conv_3 = layers.conv_relu_layer('Conv3', pool_2, kernel_size=3, stride=1, output_dim=32)
    pool_3 = layers.pooling_layer('Pool3', conv_3, kernel_size=4, stride=1)
    fc_1 = layers.fc_relu_layer('Fc1', pool_3, output_dim=256)
    fc_1 = tf.layers.batch_normalization(fc_1, training=True)
    fc_1 = tf.nn.dropout(fc_1, keep_prob=keep_prob)
    fc_2 = layers.fc_relu_layer('Fc2', fc_1, output_dim=256)
    fc_2 = tf.layers.batch_normalization(fc_2, training=True)
    fc_2 = tf.nn.dropout(fc_2, keep_prob=keep_prob)
    fc_3 = layers.fc_layer('Fc3', fc_2, output_dim=128)

    return fc_3


def build_gan(sampled_latent_variables, real_data, mode=MODE_FLAG_TRAIN, generator=None, discriminator=None):
    """
    A simple GAN
    :param sampled_latent_variables:
    :param real_data:
    :param mode:
    :param generator:
    :param discriminator:
    :return:
    """
    if mode == MODE_FLAG_TRAIN:
        drop_prob = 0.5
    else:
        drop_prob = 1
    if generator is None:
        generator = build_generator
    if discriminator is None:
        discriminator = build_discriminator

    with tf.variable_scope(NAME_SCOPE_GENERATIVE_NET):
        gen_out = generator(sampled_latent_variables)
    with tf.variable_scope(NAME_SCOPE_DISCRIMINATIVE_NET):
        dis_out_real = discriminator(real_data, keep_prob=drop_prob)
    with tf.variable_scope(NAME_SCOPE_DISCRIMINATIVE_NET, reuse=True):
        dis_out_latent = discriminator(gen_out, keep_prob=drop_prob)

    return gen_out, dis_out_latent, dis_out_real


def loss_func(dis_out_latent, dis_out_real):
    """
    To set the adversarial loss functions
    :param dis_out_latent:
    :param dis_out_real:
    :return:
    """
    loss_latent = tf.reduce_mean(dis_out_latent)
    loss_real = tf.reduce_mean(dis_out_real)
    loss_dis = (loss_real - loss_latent)
    return loss_latent, loss_dis


class WGan(object):
    def __init__(self, batch_size, sess=tf.Session(), mode=MODE_FLAG_TRAIN, restore_file=None):
        """
        A generative adversarial network class for training or test
        :param batch_size:
        :param sess: A TensorFlow Session
        :param mode: MODE_FLAG_TRAIN or MODE_FLAG_TEST
        :param restore_file: Path to the previously trained model
        """
        self.sess = sess
        self.mode = mode
        self.restore_file = restore_file
        self.batch_shape = [batch_size, 64, 64, 3]
        self.real_data = tf.placeholder(tf.float32, self.batch_shape)
        self.sampled_latent_variables = tf.random_normal([batch_size, 2, 2, 128], stddev=1.)
        self.nets = self._build_net()
        if mode == MODE_FLAG_TRAIN:
            self.loss = self._get_loss()
            self.g_step = tf.Variable(0, trainable=False)
            self.d_step = tf.Variable(0, trainable=False)
            self.ops = self._get_opt()
        else:
            assert self.restore_file is not None
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.restore_file()

    def _build_net(self):
        if self.mode == MODE_FLAG_TRAIN:
            nets = build_gan(self.sampled_latent_variables, self.real_data, mode=self.mode)
            tf.summary.image(NAME_SCOPE_GENERATIVE_NET + '/Image', nets[0])
            tf.summary.image(NAME_SCOPE_DISCRIMINATIVE_NET + '/Image', self.real_data)
            tf.summary.histogram(NAME_SCOPE_GENERATIVE_NET + '/SampledDiscrimination', nets[1])
            tf.summary.histogram(NAME_SCOPE_GENERATIVE_NET + '/GeneratedHist', nets[0] * 256.)
            tf.summary.histogram(NAME_SCOPE_DISCRIMINATIVE_NET + '/RealDiscrimination', nets[2])
            tf.summary.histogram(NAME_SCOPE_DISCRIMINATIVE_NET + '/RealHist', self.real_data)
            return nets
        else:
            with tf.variable_scope(NAME_SCOPE_GENERATIVE_NET):
                return build_generator(self.sampled_latent_variables)

    def _get_loss(self):
        losses = loss_func(self.nets[1], self.nets[2])
        tf.summary.scalar(NAME_SCOPE_GENERATIVE_NET + '/Loss', losses[0])
        tf.summary.scalar(NAME_SCOPE_DISCRIMINATIVE_NET + '/Loss', losses[1])

        return losses

    def _get_opt(self):
        opt = tf.train.RMSPropOptimizer(learning_rate=0.00001)
        train_list_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=NAME_SCOPE_GENERATIVE_NET)
        generative_step = opt.minimize(self.loss[0], global_step=self.g_step, var_list=train_list_gen,
                                       aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        train_list_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=NAME_SCOPE_DISCRIMINATIVE_NET)
        discriminative_step_provisional = opt.minimize(self.loss[1], global_step=self.d_step, var_list=train_list_dis,
                                                       aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        dis_clipping = [tf.assign(var, tf.clip_by_value(var, -0.1, 0.1)) for var in train_list_dis]
        with tf.control_dependencies([discriminative_step_provisional]):
            discriminative_step = tf.tuple(dis_clipping)
        return generative_step, discriminative_step

    def _restore(self):
        if self.mode == MODE_FLAG_TRAIN:
            save_list = tf.trainable_variables()
        else:
            save_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=NAME_SCOPE_GENERATIVE_NET)
        saver = tf.train.Saver(var_list=save_list)
        return saver.restore(self.sess, self.restore_file)

    def training_loop(self, batch_reader, k=1, max_loop=200000, summary_path='E:\\WorkSpace\\WorkSpace\\TrainingLogs',
                      snapshot_path='E:\\WorkSpace\\WorkSpace\\SavedModels\\GAN'):
        """
        The main training loop of gan
        :param batch_reader: A data reading function pointer
        :param k: The training interval of the generative net
        :param max_loop:
        :param summary_path:
        :param snapshot_path:
        :return:
        """
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        writer = tf.summary.FileWriter(summary_path + '/' + time_string + '/')
        saver = tf.train.Saver()
        summary_gen = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=NAME_SCOPE_GENERATIVE_NET))
        summary_dis = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=NAME_SCOPE_DISCRIMINATIVE_NET))
        if self.restore_file is not None:
            self._restore()
        for i in range(max_loop):
            this_batch = batch_reader(i)
            loss_d, _, summary_dis_out, d_step_out = self.sess.run(
                [self.loss[1], self.ops[1], summary_dis, self.d_step],
                feed_dict={self.real_data: this_batch['batch_image']})
            writer.add_summary(summary_dis_out, global_step=d_step_out)
            if (i + 1) % k == 0:
                loss_g, _, summary_gen_out, g_step_out = self.sess.run(
                    [self.loss[0], self.ops[0], summary_gen, self.g_step])
                writer.add_summary(summary_gen_out, global_step=g_step_out)
                print('Iteration ' + str(i) + ' d-step loss:' + str(loss_d) + ' g-step loss:' + str(loss_g))

            if (i + 1) % 10000 == 0:
                saver.save(self.sess, snapshot_path + '/YMModel', i)

    def forward(self):
        if self.mode == MODE_FLAG_TRAIN:
            to_run = self.nets[0]
        else:
            to_run = self.nets
        return self.sess.run(to_run)


if __name__ == '__main__':
    """
    This is the test routine of the GAN.
    """
    print('Now we are going to train the network.')
    import scipy.io as sio


    def reader(i):
        total_batches = 1000
        data_folder = 'E:\\WorkSpace\\Data\\Face\\Batch\\'
        file_name = data_folder + 'batch_' + str(i % total_batches + 1) + '.mat'
        mat_file = sio.loadmat(file_name)
        batch = dict()
        batch['batch_image'] = np.asarray(mat_file['batch_image'], dtype=np.float32) / 256.
        batch['batch_random'] = np.random.randn(20, 2, 2, 128)
        return batch


    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    this_session = tf.Session(config=config)
    model = WGan(20, this_session)
    model.training_loop(reader, k=1)