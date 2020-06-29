from abc import abstractmethod
import tensorflow as tf
import re


class DataExtractorBase:

    def __init__(self, meta_data, batch_size, dataset_size_ratio=1):
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.dataset_size = None
        self.padding_size = None
        self.label_vector_size = None
        self.input_shape = None
        self.dataset_size_ratio = dataset_size_ratio
        self.label_regex = re.compile('^[^-]*-[^-]*')
        self.__create_dataset()

    def __create_dataset(self):
        self.dataset = self.__load_file_and_labels_dataset()
        self.dataset = self.dataset.map(lambda data, label: self.load_spectrogram_data(data, label))
        return self.dataset

    def __load_file_and_labels_dataset(self):
        paths, labels = self.meta_data.load_file_and_labels_dataset()
        self.dataset_size = len(paths)
        file_stream = tf.io.read_file(self.meta_data.work_data_path + paths[0])
        tensor = tf.io.parse_tensor(file_stream, tf.float32)
        self.input_shape = tensor.shape
        self.label_vector_size = labels[0].shape[0]
        return tf.data.Dataset.from_tensor_slices((paths, labels))

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

