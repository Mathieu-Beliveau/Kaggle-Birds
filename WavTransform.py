import librosa as rosa
import librosa.display
import os
import multiprocessing
import threading
import concurrent.futures
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class WavTransform:

    meta_data = None

    def __init__(self, meta_data):
        self.meta_data = meta_data

    def generate_mel_spectograms(self):
        paths = self.meta_data.get_source_data_paths()
        max_workers = multiprocessing.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.generate_mel_spectrogram_thread, paths)
        # for path in paths:
        #     self.generate_mel_spectrogram_thread(path)

    def generate_mel_spectrogram_thread(self, path):
        n_fft = 512
        hop_length = 256
        win_length = 512
        # target_sample_rate = 16000
        file_name = os.path.basename(path)
        y, sr = rosa.load(path, mono=True)
        rosa.effects.trim(y)
        mel_spectogram = rosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length,
                                                     win_length=win_length)
        s_db = rosa.power_to_db(mel_spectogram, ref=np.max)
        # s_db = np.clip(s_db, a_min=-40, a_max=None)
        # rosa.display.specshow(s_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        s_db_normalized = rosa.util.normalize(s_db)
        # rosa.display.specshow(s_db_normalized, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        # plt.colorbar(format='%+2.0f dB')
        # plt.show()
        serialized_mel_spectogram = tf.io.serialize_tensor(s_db_normalized)
        tf.io.write_file(self.meta_data.work_data_path + file_name[:-4] + ".mel_spec", serialized_mel_spectogram)

    def downsample_wavs(self,  target_sample_rate):
        paths = self.meta_data.get_source_data_paths()
        for path in paths:
            file_name = os.path.basename(path)
            y, s = rosa.load(path, sr=target_sample_rate, mono=True)
            rosa.output.write_wav(self.meta_data.work_data_path + file_name, y, target_sample_rate, norm=True)
