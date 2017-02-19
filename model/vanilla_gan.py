import tensorflow as tf


def build_generator(input_tensor):
    """
    To build the generative network
    :param input_tensor:
    :return:
    """
    return input_tensor


def build_discriminator(input_tensor):
    """
    To build the discriminative network
    :param input_tensor:
    :return:
    """
    return input_tensor


def build_gan(sampled_latent_variables, real_data, generator=None, discriminator=None):
    """
    A simple GAN
    :param sampled_latent_variables:
    :param real_data:
    :param generator:
    :param discriminator:
    :return:
    """
    if generator is None:
        generator = build_generator
    if discriminator is None:
        discriminator = build_discriminator

    with tf.variable_scope('GenerativeNet'):
        gen_out = generator(sampled_latent_variables)
    with tf.variable_scope('DiscriminativeNet', reuse=True):
        dis_out_real = discriminator(real_data)
        dis_out_latent = discriminator(gen_out)

    return gen_out, dis_out_latent, dis_out_real


def loss_func(dis_out_latent, dis_out_real):
    loss_latent = tf.reduce_mean(1. - tf.sigmoid(dis_out_latent))
    loss_real = tf.reduce_mean(tf.sigmoid(dis_out_real))
    loss_gen = tf.reduce_mean(tf.sigmoid(dis_out_latent))
    loss_dis = loss_latent + loss_real
    return loss_gen, loss_dis


class Gan(object):
    def __init__(self, sess=tf.Session()):
        self.sess = sess
        self.batch_shape = [None, 128, 128, 3]
        self.real_data = tf.placeholder(tf.float32, self.batch_shape)
        self.sampled_latent_variables = tf.random_normal(self.batch_shape, stddev=0.1)
        self.nets = self._build_net()
        self.loss = self._get_loss()
        self.g_step = tf.Variable(0, trainable=False)
        self.d_step = tf.Variable(0, trainable=False)
        self.ops = self._get_opt()

    def _build_net(self):
        return build_gan(self.sampled_latent_variables, self.real_data)

    def _get_loss(self):
        return loss_func(self.nets[1], self.nets[2])

    def _get_opt(self):
        opt = tf.train.AdamOptimizer()
        train_list_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GenerativeNet')
        generative_step = opt.minimize(self.loss[0], global_step=self.g_step, var_list=train_list_gen)

        train_list_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DiscriminativeNet')
        discriminative_step = opt.minimize(self.loss[1], global_step=self.d_step, var_list=train_list_dis)
        return generative_step, discriminative_step

    @staticmethod
    def _restore(restore_file):
        return restore_file

    def training_loop(self, batch_reader, k=1, max_loop=10000, restore_file=None):
        """
        The main training loop of gan
        :param batch_reader: A data reading function pointer
        :param k: The training interval of the generative net
        :param max_loop:
        :param restore_file:
        :return:
        """
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if restore_file is not None:
            self._restore(restore_file)
        for i in range(max_loop):
            this_batch = batch_reader(i)
            loss_d, _ = self.sess.run([self.loss[1], self.ops[1]],
                                      feed_dict={self.real_data: this_batch['batch_image']})
            if (i + 1) % k == 0:
                loss_g, _ = self.sess.run([self.loss[0], self.ops[0]])
