import tensorflow as tf
from util.data import DatasetProto
from model.net_factory import NetFactory

if __name__ == '__main__':
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=sess_config)
    batch_size = 30
    train_size = 3000
    data_file = 'E:\\WorkSpace\\Data\\CIFAR10\\face.hdf5'
    saved_model = 'E:\\WorkSpace\\Data\\Log\\Gan\\model\\YMModel-16002'
    net_config = {
        'sess': sess,
        'batch_size': batch_size,
        'model': 'wasserstein',
        'log_path': 'E:\\WorkSpace\\Data\\Log\\Gan',
        'decision_length': 10
    }
    dataset = DatasetProto(train_size, 0, batch_size, data_file)
    model = NetFactory.get_net(**net_config)
    model.train(10000, dataset, restore_file=None)
