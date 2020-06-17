import pandas as pd
import tensorflow as tf
import librosa as rosa
import os

class DataExtractor:

    meta_data = None
    dataset = None
    padding_size = None
    dataset_size = None
    means = None
    variances = None

    def __init__(self, meta_data, padding_size, batch_size, means_file_path=None, variance_file_path=None):
        self.padding_size = padding_size
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.means_file_path = means_file_path
        self.variance_file_path = variance_file_path
        if means_file_path is not None and variance_file_path is not None:
            self.load_means_and_variance(means_file_path, variance_file_path)
        self.__create_dataset()

    def __create_dataset(self):
        self.dataset = self.__load_file_and_labels_dataset()
        self.dataset = self.dataset.shuffle(self.meta_data.dataset_size)
        self.dataset = self.dataset.map(lambda wav, label: self.load_wav_data(wav, label))
        return self.dataset

    def load_means_and_variance(self, means_file_path, variance_file_path):
        means_content = tf.io.read_file(means_file_path)
        self.means = tf.io.parse_tensor(means_content, tf.float32)
        variance_content = tf.io.read_file(variance_file_path)
        self.variances = tf.io.parse_tensor(variance_content, tf.float32)

    def load_wav_data_simple(self, file, label):
        wav = tf.io.read_file(file)
        wav_tensor, sample_rate = tf.audio.decode_wav(wav)
        return wav_tensor, label

    def load_wav_data(self, file, label):
        wav = tf.io.read_file(file)
        wav_tensor, sample_rate = tf.audio.decode_wav(wav)
        wav_len = tf.shape(wav_tensor)[0]
        pad_len = tf.subtract(self.padding_size, wav_len)
        wav_tensor = tf.pad(wav_tensor, [[0, pad_len], [0, 0]])
        if self.means is not None and self.variances is not None:
            wav_tensor = tf.subtract(wav_tensor, self.means)
            wav_tensor = tf.divide(wav_tensor, self.variances)
        return wav_tensor, label

    def get_datasets(self, train_ratio, validation_ratio, test_ratio, epochs=1):
        if train_ratio + validation_ratio + test_ratio != 1:
            raise Exception("invalid train, validation or test ratios; they must sum to 1.")

        train_size = round(self.meta_data.dataset_size * train_ratio)
        validation_size = round(self.meta_data.dataset_size * validation_ratio)
        train_data = self.dataset.take(train_size)
        train_data = self.__pad_dataset(train_data)
        test_dataset = self.dataset.skip(train_size)
        validation_data = test_dataset.take(validation_size)
        validation_data = self.__pad_dataset(validation_data)
        test_data = test_dataset.skip(validation_size)
        return train_data, validation_data, test_data

    def __pad_dataset(self, dataset):
        return dataset.padded_batch(self.batch_size, padded_shapes=([self.padding_size, 1], [50, ]))

    def get_max_wav_length(self):
        max_shape = 0
        for wav, one_hot in self.dataset:
            if wav.shape[0] > max_shape:
                max_shape = wav.shape[0]
        return max_shape

    def get_dataset_feature_means(self, padding_size,  means_file_name):
        padded_data = self.__pad_dataset(self.dataset)
        means = tf.zeros([padding_size, 1])
        counts = 0
        for wav, one_hot in padded_data:
            summed_wav = tf.reduce_sum(wav, 0)
            counts += wav.shape[0]
            means = tf.add(means, summed_wav)
        means = tf.divide(means, tf.cast(counts, tf.float32))
        serialized_means = tf.io.serialize_tensor(means)
        tf.io.write_file(means_file_name, serialized_means)

    def get_dataset_feature_variance(self, padding_size, means_file_name,  variance_file_name):
        variance = tf.zeros([padding_size, 1])
        content = tf.io.read_file(means_file_name)
        means = tf.io.parse_tensor(content, tf.float32)
        counts = 0
        padded_data = self.__pad_dataset(self.dataset)
        for wav, one_hot in padded_data:
            batch_diff = tf.subtract(wav, means)
            batch_diff_squared = tf.square(batch_diff)
            batch_diff_squared = tf.reduce_sum(batch_diff_squared, 0)
            variance = tf.add(variance, batch_diff_squared)
            counts += wav.shape[0]
        variance = tf.divide(means, tf.cast(counts, tf.float32))
        serialized_variance = tf.io.serialize_tensor(variance)
        tf.io.write_file(variance_file_name, serialized_variance)

    def __load_file_and_labels_dataset(self):
        species = self.meta_data.info["Species"]
        paths = [tf.constant(path) for path in self.meta_data.get_work_data_paths()]
        one_hot_labels = self.__transform_labels_to_one_hot_vector(self, species)
        return tf.data.Dataset.from_tensor_slices((paths, one_hot_labels))

    @staticmethod
    def __transform_labels_to_one_hot_vector(self, labels):
        labels_df = pd.DataFrame(labels)
        return pd.get_dummies(labels_df, dtype=float).values
