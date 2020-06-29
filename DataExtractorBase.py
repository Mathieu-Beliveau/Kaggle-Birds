from random import shuffle
from abc import abstractmethod
import tensorflow as tf
import pickle
import re
import os


class DataExtractorBase:

    def __init__(self, meta_data, batch_size, dataset_size_ratio=1):
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.dataset_size = None
        self.padding_size = None
        self.dataset_size_ratio = dataset_size_ratio
        self.label_regex = re.compile('^[^-]*-[^-]*')
        self.__create_dataset()

    def __create_dataset(self):
        self.dataset = self.__load_file_and_labels_dataset()
        self.dataset = self.dataset.map(lambda data, label: self.load_spectrogram_data(data, label))
        return self.dataset


    def __load_file_and_labels_dataset(self):
        suffled_paths_file = self.meta_data.base_path + "suffled_paths.pkl"
        if os.path.isfile(suffled_paths_file):
            with open(suffled_paths_file, 'rb') as f:
                paths_labels_pairs = pickle.load(f)
        else:
            paths_labels_pairs = self.meta_data.get_data_paths_and_labels
            shuffle(paths_labels_pairs)
            with open(suffled_paths_file, 'wb') as f:
                pickle.dump(paths_labels_pairs, f)
        shuffled_paths = [path for path, _ in paths_labels_pairs]
        shuffled_labels = [label for _, label in paths_labels_pairs]
        self.dataset_size = len(shuffled_paths)
        return tf.data.Dataset.from_tensor_slices((shuffled_paths, shuffled_labels))

    @abstractmethod
    def load_spectrogram_data(self, data, label):
        raise NotImplementedError

    def load_spectrogram(self, file):
        spectrogram = tf.io.read_file(self.meta_data.work_data_path + file)
        spectrogram = tf.io.parse_tensor(spectrogram, tf.float32)
        spectrogram = tf.reshape(spectrogram, (128, 216, 3))
        return spectrogram

    def get_datasets(self, train_ratio, validation_ratio, test_ratio, epochs=1):
        if train_ratio + validation_ratio + test_ratio != 1:
            raise Exception("invalid train, validation or test ratios; they must sum to 1.")
        train_size = round(self.dataset_size * train_ratio)
        validation_size = round(self.dataset_size * validation_ratio)
        train_data = self.dataset.take(train_size).batch(self.batch_size)
        test_dataset = self.dataset.skip(train_size)
        validation_data = test_dataset.take(validation_size).batch(self.batch_size)
        test_data = test_dataset.skip(validation_size).batch(self.batch_size)
        return train_data, validation_data, test_data

