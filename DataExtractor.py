import pandas as pd
import tensorflow as tf


class DataExtractor:

    meta_data = None
    dataset = None
    padding_size = None
    dataset_size = None
    means = None
    std_deviation = None

    def __init__(self, meta_data, padding_size, batch_size, means_file_path=None, std_deviation_file_path=None):
        self.padding_size = padding_size
        self.meta_data = meta_data
        self.batch_size = batch_size
        self.means_file_path = means_file_path
        self.std_deviation_file_path = std_deviation_file_path
        if means_file_path is not None and std_deviation_file_path is not None:
            self.__load_means_and_std_deviation(means_file_path, std_deviation_file_path)
        self.__create_dataset()

    def __create_dataset(self):
        self.dataset = self.__load_file_and_labels_dataset()
        self.dataset = self.dataset.shuffle(self.meta_data.dataset_size)
        self.dataset = self.dataset.map(lambda wav, label: self.__load_wav_data(wav, label))
        return self.dataset

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

    def get_max_wav_length(self):
        max_shape = 0
        for wav, one_hot in self.dataset:
            if wav.shape[0] > max_shape:
                max_shape = wav.shape[0]
        return max_shape

    def get_dataset_mean(self, padding_size,  means_file_name):
        mean = tf.constant([0.0])
        counts = 0
        for wav, one_hot in self.dataset:
            summed_wav = tf.reduce_sum(wav, 0)
            counts += wav.shape[0]
            mean = tf.add(mean, summed_wav)
        mean = tf.divide(mean, tf.cast(counts, tf.float32))
        serialized_mean = tf.io.serialize_tensor(mean)
        tf.io.write_file(means_file_name, serialized_mean)

    def get_dataset_standard_deviation(self, padding_size, means_file_name,  variance_file_name):
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

    def __load_file_and_labels_dataset(self):
        species = self.meta_data.info["Species"]
        paths = [tf.constant(path) for path in self.meta_data.get_work_data_paths()]
        one_hot_labels = self.__transform_labels_to_one_hot_vector(self, species)
        return tf.data.Dataset.from_tensor_slices((paths, one_hot_labels))

    @staticmethod
    def __transform_labels_to_one_hot_vector(self, labels):
        labels_df = pd.DataFrame(labels)
        return pd.get_dummies(labels_df, dtype=float).values
