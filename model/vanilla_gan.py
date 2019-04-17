import tensorflow as tf
import os
import gc
from six.moves import xrange
from model import net_factory as nf

a = 1

class VanillaGan(nf.AbstractNet):
    def __init__(self, **kwargs):
        super().__init__(sess=kwargs.get('sess'))
        self.batch_size = kwargs.get('batch_size')
        self.log_path = kwargs.get('log_path')
        self.sampled_variables = tf.placeholder(tf.float32, [self.batch_size, 100])
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3])
        self.nets = self._build_net()
        self.loss = self._build_loss()
        assert self.log_path is not None

    def _build_net(self):
        """
        To build the networks
        :return: ni cai"""

        from util import nets
        with tf.variable_scope(nf.NAME_SCOPE_GENERATIVE_NET):
            generative_out = nets.build_generator_new(self.sampled_variables)
        with tf.variable_scope(nf.NAME_SCOPE_DISCRIMINATIVE_NET):
            decision_from_gen = nets.net_discriminator_new(generative_out)
        with tf.variable_scope(nf.NAME_SCOPE_DISCRIMINATIVE_NET, reuse=True):
            decision_from_real = nets.net_discriminator_new(self.real_data / 255. * 2 - 1)

        tf.summary.image(nf.NAME_SCOPE_GENERATIVE_NET + '/gen_im', generative_out)
        tf.summary.image(nf.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im', self.real_data)
        tf.summary.histogram(nf.NAME_SCOPE_GENERATIVE_NET + '/gen_dec_hist', decision_from_gen)
        tf.summary.histogram(nf.NAME_SCOPE_DISCRIMINATIVE_NET + '/dis_dec_hist', decision_from_real)
        tf.summary.histogram(nf.NAME_SCOPE_GENERATIVE_NET + '/gen_im_hist', (generative_out + 1) / 2 * 255.)
        tf.summary.histogram(nf.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im_hist', self.real_data)

        return generative_out, decision_from_gen, decision_from_real


    def _build_loss(self):
        loss_gen = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.nets[1]), logits=self.nets[1]))
        loss_dis_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.nets[2]), logits=self.nets[2]))
        loss_dis = loss_dis_fake + loss_dis_real

        tf.summary.scalar(nf.NAME_SCOPE_GENERATIVE_NET + '/loss', loss_gen)
        tf.summary.scalar(nf.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss', loss_dis)

        return loss_gen, loss_dis

    def _build_opt(self):
        train_list_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nf.NAME_SCOPE_GENERATIVE_NET)
        train_list_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nf.NAME_SCOPE_DISCRIMINATIVE_NET)
        trainer1 = tf.train.AdamOptimizer(0.0002, beta1=0.5)
        trainer2 = tf.train.AdamOptimizer(0.0002, beta1=0.5)
        op_gen = trainer1.minimize(self.loss[0], var_list=train_list_gen, global_step=self.g_step)
        op_dis = trainer2.minimize(self.loss[1], var_list=train_list_dis, global_step=self.g_step)

        return op_gen, op_dis

    def train(self, max_iter, dataset, restore_file=None):
        from time import gmtime, strftime
        time_string = strftime("%a%d%b%Y-%H%M%S", gmtime())
        ops = self._build_opt()
        initial_op = tf.global_variables_initializer()
        self.sess.run(initial_op)
        summary_path = os.path.join(self.log_path, 'log', time_string) + os.sep
        save_path = os.path.join(self.log_path, 'model') + os.sep

        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if restore_file is not None:
            self._restore(restore_file)
            print('Model restored.')

        writer = tf.summary.FileWriter(summary_path)
        summary_gen = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=nf.NAME_SCOPE_GENERATIVE_NET))
        summary_dis = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope=nf.NAME_SCOPE_DISCRIMINATIVE_NET))
        d_loss = 0
        g_loss = 0
        for i in xrange(max_iter):
            this_batch = dataset.next_batch_train()
            if i % 2 == 0:
                noise = dataset.next_batch_noise()
                g_loss, _, gen_sum = self.sess.run([self.loss[0], ops[0], summary_gen],
                                                   feed_dict={self.sampled_variables: noise})
                writer.add_summary(gen_sum, global_step=tf.train.global_step(self.sess, self.g_step))
            d_loss, _, dis_sum = self.sess.run([self.loss[1], ops[1], summary_dis],
                                               feed_dict={self.real_data: this_batch[0],
                                                          self.sampled_variables: this_batch[1]})
            step = tf.train.global_step(self.sess, self.g_step)
            writer.add_summary(dis_sum, global_step=step)
            print(
                'Batch ' + str(i) + '(Global Step: ' + str(step) + '): ' + str(
                    g_loss) + '; ' + str(d_loss))

            gc.collect()

            if i % 2000 == 0 and i > 0:
                self._save(save_path, step)
        return 0
