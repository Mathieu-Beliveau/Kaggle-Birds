import librosa as rosa
import librosa.display
import os
import multiprocessing
import concurrent.futures
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class WavTransform:

    meta_data = None

    def __init__(self, meta_data, display_spectrograms=False, use_clipping=True, trim_silence=True):
        self.meta_data = meta_data
        self.display_spectrograms = display_spectrograms
        self.use_clipping = use_clipping
        self.trim_silence = trim_silence

    def generate_mel_spectograms(self):
        paths = self.meta_data.get_source_data_paths()
        max_workers = multiprocessing.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.generate_mel_spectrogram_thread, paths)
        # for path in paths:
        #     self.generate_mel_spectrogram_thread(path)

    def generate_mel_spectrogram_thread(self, path):
        target_sample_rate = 16000
        n_fft = 1024
        hop_length = 512
        win_length = 1024
        file_name = os.path.basename(path)
        y, sr = rosa.load(path, mono=True, sr=target_sample_rate)
        if self.trim_silence:
            rosa.effects.trim(y)

        mel_spectogram = rosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length,
                                                     win_length=win_length)
        s_db = rosa.power_to_db(mel_spectogram, ref=np.max)
        if self.use_clipping:
            s_db = np.clip(s_db, a_min=-50, a_max=None)
        s_db_normalized = rosa.util.normalize(s_db)
        if self.display_spectrograms:
            rosa.display.specshow(s_db_normalized, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.show()
            print(s_db_normalized.shape)
        serialized_mel_spectogram = tf.io.serialize_tensor(s_db_normalized)
        tf.io.write_file(self.meta_data.work_data_path + file_name[:-4] + ".mel_spec", serialized_mel_spectogram)

    def downsample_wavs(self,  target_sample_rate):
        paths = self.meta_data.get_source_data_paths()
        for path in paths:
            file_name = os.path.basename(path)
            y, s = rosa.load(path, sr=target_sample_rate, mono=True)
            rosa.output.write_wav(self.meta_data.work_data_path + file_name, y, target_sample_rate, norm=True)
