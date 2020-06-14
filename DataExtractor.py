import pathlib
import os
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf


@tf.function
def write_map_fn(tensor, _):
    return tf.io.serialize_tensor(tensor)


class DataExtractor:

    def __init__(self):
        data_prefix_path = "../Bird_Songs/"
        meta_data = pd.read_csv(data_prefix_path + "metadata.csv")
        wav_tensors = []
        labels = []
        i = 0
        for path, specie in zip(meta_data["Path"], meta_data["Species"]):
            wav_path = data_prefix_path + "Wav/" + path[5:-3] + "wav"
            with io.open(wav_path, 'rb') as wav:
                wav_audio, sample_rate = tf.audio.decode_wav(wav.read())
                wav_tensors.append(wav_audio)
                labels.append(specie)
            i += 1
            if i > 5:
                break
        labels_df = pd.DataFrame(labels)
        one_hot_labels = pd.get_dummies(labels_df, dtype=float)
        max_sample_size = max(wt.shape[0] for wt in wav_tensors)
        padded_wav_tensors = [tf.pad(wt, tf.constant([[0, max_sample_size - wt.shape[0]], [0, 0]]), "CONSTANT")
                              for wt in wav_tensors]
        dataset = tf.data.Dataset.from_tensor_slices((wav_tensors, one_hot_labels.values))
        dataset = dataset.map(write_map_fn)
        writer = tf.data.experimental.TFRecordWriter(data_prefix_path + "bird_songs.tfrecord")
        writer.write(dataset)

