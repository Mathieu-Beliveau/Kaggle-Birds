from random import shuffle
from os import walk
import pandas as pd
import tensorflow as tf
import pickle
import os

class DataExtractor:

    meta_data = None
    dataset = None
    padding_size = None
    dataset_size = None
    means = None
    std_deviation = None

    def __init__(self, meta_data, batch_size, means_file_path=None, std_deviation_file_path=None):
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.means_file_path = means_file_path
        self.padding_size_file_path = self.meta_data.base_path + "padding_size.pkl"
        self.__get_padding_size()
        self.std_deviation_file_path = std_deviation_file_path
        if means_file_path is not None and std_deviation_file_path is not None:
            self.__load_means_and_std_deviation(means_file_path, std_deviation_file_path)
        self.__create_dataset()

    def __create_dataset(self):
        self.dataset = self.__load_file_and_labels_dataset()
        self.dataset = self.dataset.map(lambda data, label: self.__load_spectrogram_data(data, label))
        if self.padding_size is None:
            self.padding_size = self.__get_max_data_tensor_length()
            with open(self.padding_size_file_path, 'wb') as f:
                pickle.dump(self.padding_size, f)
            self.dataset = self.__load_file_and_labels_dataset()
            self.dataset = self.dataset.map(lambda data, label: self.__load_spectrogram_data(data, label))
        return self.dataset

    def __get_padding_size(self):
        if os.path.isfile(self.padding_size_file_path):
            with open(self.padding_size_file_path, 'rb') as f:
                self.padding_size = pickle.load(f)
        else:
            return None

    def __load_file_and_labels_dataset(self):
        suffled_paths_file = self.meta_data.base_path + "suffled_paths.pkl"
        if os.path.isfile(suffled_paths_file):
            with open(suffled_paths_file, 'rb') as f:
                paths_labels_pairs = pickle.load(f)
        else:
            species = []
            file_paths = []
            for (dirpath, dirnames, filenames) in walk(self.meta_data.work_data_path):
                file_paths = filenames
                for filename in filenames:
                    label = os.path.splitext(filename)[0][:-5]
                    species.append(label)
            one_hot_labels = self.__transform_labels_to_one_hot_vector(self, species)
            paths_labels_pairs = [[path, label] for path, label in zip(file_paths, one_hot_labels)]
            shuffle(paths_labels_pairs)
            with open(suffled_paths_file, 'wb') as f:
                pickle.dump(paths_labels_pairs, f)
        shuffled_paths = [path for path, _ in paths_labels_pairs]
        shuffled_labels = [label for _, label in paths_labels_pairs]
        return tf.data.Dataset.from_tensor_slices((shuffled_paths, shuffled_labels))

    def __load_means_and_std_deviation(self, means_file_path, std_deviation_file_path):
        means_content = tf.io.read_file(means_file_path)
        self.means = tf.io.parse_tensor(means_content, tf.float32)
        std_deviation_content = tf.io.read_file(std_deviation_file_path)
        self.std_deviation = tf.io.parse_tensor(std_deviation_content, tf.float32)

    def __load_wav_data(self, file, label):
        wav = tf.io.read_file(file)
        wav_tensor, sample_rate = tf.audio.decode_wav(wav)
        wav_len = tf.shape(wav_tensor)[0]
        pad_len = tf.subtract(self.padding_size, wav_len)
        wav_tensor = tf.pad(wav_tensor, [[0, pad_len], [0, 0]])
        if self.means is not None and self.std_deviation is not None:
            wav_tensor = tf.subtract(wav_tensor, self.means)
            wav_tensor = tf.divide(wav_tensor, self.std_deviation)
        return wav_tensor, label

    def __load_spectrogram_data(self, file, label):
        mel_spectrogram = tf.io.read_file(file)
        mel_spectrogram = tf.io.parse_tensor(mel_spectrogram, tf.float32)
        chroma_spectrogram = tf.io.read_file(file[:-8] + "chr_spec")
        chroma_spectrogram = tf.io.parse_tensor(chroma_spectrogram, tf.float32)
        spectrogram = tf.stack(mel_spectrogram, chroma_spectrogram)
        if self.padding_size is not None:
            spectrogram_tensor = self.__pad_dataset(spectrogram)
        return spectrogram_tensor, label

    def __pad_dataset(self, tensor):
        spectrogram_length = tf.shape(tensor)[1]
        pad_len = tf.subtract(self.padding_size, spectrogram_length)
        padded_tensor = tf.pad(tensor, [[0, 0], [0, pad_len]])
        padded_tensor = tf.reshape(padded_tensor, (128, self.padding_size, 1))
        return padded_tensor

    def __standardize_dataset(self,  tensor, label):
        if self.means is not None and self.std_deviation is not None:
            tensor = tf.subtract(tensor, self.means)
            tensor = tf.divide(tensor, self.std_deviation)
        return tensor, label

    def get_datasets(self, train_ratio, validation_ratio, test_ratio, epochs=1):
        if train_ratio + validation_ratio + test_ratio != 1:
            raise Exception("invalid train, validation or test ratios; they must sum to 1.")

        train_size = round(self.meta_data.dataset_size * train_ratio)
        validation_size = round(self.meta_data.dataset_size * validation_ratio)
        train_data = self.dataset.take(train_size).batch(self.batch_size)
        test_dataset = self.dataset.skip(train_size)
        validation_data = test_dataset.take(validation_size).batch(self.batch_size)
        test_data = test_dataset.skip(validation_size).batch(self.batch_size)
        return train_data, validation_data, test_data

    def __get_max_data_tensor_length(self):
        max_shape = 0
        for data_tensor, one_hot in self.dataset:
            if data_tensor.shape[1] > max_shape:
                max_shape = data_tensor.shape[1]
        return max_shape

    def get_dataset_mean(self,  means_file_name):
        mean = tf.constant([0.0])
        counts = 0
        for wav, one_hot in self.dataset:
            summed_wav = tf.reduce_sum(wav, 0)
            counts += wav.shape[0]
            mean = tf.add(mean, summed_wav)
        mean = tf.divide(mean, tf.cast(counts, tf.float32))
        serialized_mean = tf.io.serialize_tensor(mean)
        tf.io.write_file(means_file_name, serialized_mean)

    def get_dataset_standard_deviation(self, means_file_name,  variance_file_name):
        sd = tf.constant([0.0])
        content = tf.io.read_file(means_file_name)
        mean = tf.io.parse_tensor(content, tf.float32)
        counts = 0
        for wav, one_hot in self.dataset:
            batch_diff = tf.subtract(wav, mean)
            batch_diff_squared = tf.square(batch_diff)
            batch_diff_squared = tf.reduce_sum(batch_diff_squared, 0)
            sd = tf.add(sd, batch_diff_squared)
            counts += wav.shape[0]
        sd = tf.divide(sd, tf.cast(counts, tf.float32))
        sd = tf.sqrt(sd)
        serialized_variance = tf.io.serialize_tensor(sd)
        tf.io.write_file(variance_file_name, serialized_variance)

    @staticmethod
    def __transform_labels_to_one_hot_vector(self, labels):
        labels_df = pd.DataFrame(labels)
        return pd.get_dummies(labels_df, dtype=float).values
