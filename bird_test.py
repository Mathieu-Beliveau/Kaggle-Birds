import pathlib
import os
import io
import pandas as pd
import numpy as np
import tensorflow as tf


def transform_labels_to_one_hot_vector(labels):
    labels_df = pd.DataFrame(labels)
    return pd.get_dummies(labels_df, dtype=float).values


def load_wav_data(file_path):
    file_name = tf.strings.split(file_path, os.sep)[-1]
    label_len = tf.strings.length(file_name) - 11
    label = tf.strings.substr(file_name, 0, label_len)
    wav = tf.io.read_file(file_path)
    wav_tensor, sample_rate = tf.audio.decode_wav(wav)
    return wav_tensor, label


data_prefix_path = "../Bird_Songs/"
meta_data = pd.read_csv(data_prefix_path + "metadata.csv")
one_hot_labels = transform_labels_to_one_hot_vector(meta_data["Species"])
dataset = tf.data.Dataset.list_files(data_prefix_path + "Wav/*")
dataset = dataset.map(load_wav_data)
x = 0


