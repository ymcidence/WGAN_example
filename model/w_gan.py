import tensorflow as tf
from model import net_factory as nf
from model.vanilla_gan import VanillaGan


class WassersteinGan(VanillaGan):
    def __init__(self, **kwargs):
        self.decision_length = kwargs.get('decision_length')
        assert self.decision_length is not None
        super().__init__(**kwargs)

    def _build_net(self):
        from util import nets
        with tf.variable_scope(nf.NAME_SCOPE_GENERATIVE_NET):
            generative_out = nets.build_generator_new(self.sampled_variables)
        with tf.variable_scope(nf.NAME_SCOPE_DISCRIMINATIVE_NET):
            decision_from_gen = nets.net_discriminator_new(generative_out, output_dim=self.decision_length)
        with tf.variable_scope(nf.NAME_SCOPE_DISCRIMINATIVE_NET, reuse=True):
            decision_from_real = nets.net_discriminator_new(self.real_data / 255. * 2 - 1,
                                                            output_dim=self.decision_length)

        tf.summary.image(nf.NAME_SCOPE_GENERATIVE_NET + '/gen_im', generative_out)
        tf.summary.image(nf.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im', self.real_data)
        tf.summary.histogram(nf.NAME_SCOPE_GENERATIVE_NET + '/gen_dec_hist', decision_from_gen)
        tf.summary.histogram(nf.NAME_SCOPE_DISCRIMINATIVE_NET + '/dis_dec_hist', decision_from_real)
        tf.summary.histogram(nf.NAME_SCOPE_GENERATIVE_NET + '/gen_im_hist', (generative_out + 1) / 2 * 255.)
        tf.summary.histogram(nf.NAME_SCOPE_DISCRIMINATIVE_NET + '/real_im_hist', self.real_data)

        return generative_out, decision_from_gen, decision_from_real

    def _build_loss(self):
        loss_dis_fake = tf.reduce_mean(tf.nn.sigmoid(self.nets[1]))
        loss_dis_real = tf.reduce_mean(tf.nn.sigmoid(self.nets[2]))
        loss_dis = loss_dis_real - loss_dis_fake
        loss_gen = loss_dis_fake
        tf.summary.scalar(nf.NAME_SCOPE_GENERATIVE_NET + '/loss', loss_gen)
        tf.summary.scalar(nf.NAME_SCOPE_DISCRIMINATIVE_NET + '/loss', loss_dis)

        return loss_gen, loss_dis

    def _build_opt(self):
        trainer1 = tf.train.RMSPropOptimizer(learning_rate=0.0002)
        trainer2 = tf.train.RMSPropOptimizer(learning_rate=0.0002)
        train_list_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nf.NAME_SCOPE_GENERATIVE_NET)
        op_gen = trainer1.minimize(self.loss[0], global_step=self.g_step, var_list=train_list_gen,
                                   aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        train_list_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=nf.NAME_SCOPE_DISCRIMINATIVE_NET)
        op_dis_provisional = trainer2.minimize(self.loss[1], global_step=self.g_step,
                                               var_list=train_list_dis,
                                               aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        dis_clipping = [tf.assign(var, tf.clip_by_value(var, -1, 1)) for var in train_list_dis]
        with tf.control_dependencies([op_dis_provisional]):
            op_dis = tf.tuple(dis_clipping)
        return op_gen, op_dis
