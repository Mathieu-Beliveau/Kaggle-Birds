import os
import pandas as pd
import tensorflow as tf
import numpy as np


def load_wav_data(file, label):
    wav = tf.io.read_file(file)
    wav_tensor, sample_rate = tf.audio.decode_wav(wav)
    return wav_tensor, label


class DataExtractor:
    data_prefix_path = "../Bird_Songs/"
    dataset = None
    meta_data = None

    def __init__(self):
        self.meta_data = pd.read_csv(self.data_prefix_path + "metadata_trimmed.csv")
        self.dataset = self.__load_file_and_labels_dataset(self)
        self.dataset = self.dataset.shuffle(self.meta_data.shape[0])
        self.dataset = self.dataset.map(load_wav_data)
        #Current Largest tensor dim for wav length: 31170287
        self.dataset = self.dataset.padded_batch(20, padded_shapes=([31170287, 1], [50,]))

    @staticmethod
    def __load_file_and_labels_dataset(self):
        species = self.meta_data["Species"]
        paths = [tf.constant(self.data_prefix_path + "/Wav" + path[4:-3] + "wav") for path in self.meta_data["Path"]]
        one_hot_labels = self.__transform_labels_to_one_hot_vector(self, species)
        return tf.data.Dataset.from_tensor_slices((paths, one_hot_labels))

    @staticmethod
    def __transform_labels_to_one_hot_vector(self, labels):
        labels_df = pd.DataFrame(labels)
        return pd.get_dummies(labels_df, dtype=float).values





