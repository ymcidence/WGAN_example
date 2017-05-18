import h5py
import numpy as np
from sklearn.utils import shuffle


class DatasetProto(object):
    def __init__(self, train_num, test_num, batch_size, file_path):
        self.test_num = test_num
        self.train_num = train_num
        self.batch_size = batch_size
        self.file_path = file_path
        self._where_train = 0
        self._where_test = 0
        self.data_train, self.data_test = self._get_data()

    def _get_data(self):
        this_file = h5py.File(self.file_path, 'r')
        train_data = this_file['img'][0:self.train_num]
        test_data = 0
        this_file.close()
        return train_data, test_data

    def next_batch_train(self):
        assert (self.train_num % self.batch_size) == 0
        ind_start = self._where_train
        ind_end = ind_start + self.batch_size
        batch_data = self.data_train[ind_start:ind_end]
        self._where_train = (self.batch_size + self._where_train)
        if self._where_train >= self.train_num:
            self._where_train = 0
            self.reshuffle()
        batch_noise = np.random.uniform(-1, 1, size=(self.batch_size, 100))
        return batch_data, batch_noise

    def next_batch_noise(self):
        batch_noise = np.random.uniform(-1, 1, size=(self.batch_size, 100))
        return batch_noise

    def iter_num(self):
        return int(self.train_num // self.batch_size), int(self.test_num // self.batch_size)

    def reshuffle(self):
        self.data_train = shuffle(self.data_train)
