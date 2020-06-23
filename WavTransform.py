import librosa as rosa
import librosa.display
import os
import multiprocessing
import concurrent.futures
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


class WavTransform:

    meta_data = None

    def __init__(self, meta_data, display_spectrograms=False, use_clipping=True):
        self.meta_data = meta_data
        self.display_spectrograms = display_spectrograms
        self.use_clipping = use_clipping
        self.hop_length = 256
        self.audio_segment_length_in_sec = 10

    def generate_spectrograms(self):
        paths = []
        for (dirpath, dirnames, filenames) in os.walk(self.meta_data.source_data_path):
            paths = [dirpath + filename for filename in filenames]
        max_workers = multiprocessing.cpu_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.__generate_spectrograms_thread, paths)
        # for path in paths:
        #     self.__generate_spectrograms_thread(path)

    def __get_tensor_split_segments(self, sample_rate, tensor):
        segment_samples = sample_rate * self.audio_segment_length_in_sec
        trimmed_samples_count = tensor.shape[0] % segment_samples
        trimmed_tensor = tensor[: -trimmed_samples_count]
        partitions = trimmed_tensor.shape[0] / segment_samples
        return np.array_split(trimmed_tensor, partitions)

    def __generate_spectrograms_thread(self, path):
        file_name = os.path.basename(path)
        y, sr = rosa.load(path, mono=True)
        audio_tensors = self.__get_tensor_split_segments(sr, y)
        i = 0
        pad_size = 5
        for audio_tensor in audio_tensors:
            mel_spectrogram = self.generate_mel_spectrogram_thread(audio_tensor, sr)
            tf.io.write_file(self.meta_data.work_data_path + file_name[:-4] + ('_%04d' % i) + ".mel_spec", mel_spectrogram)
            i += 1
            # chroma_spectrogram = self.generate_chroma_spectrogram_thread(audio_tensor, sr)
            # tf.io.write_file(self.meta_data.work_data_path + file_name[:-4] + ".chr_spec", chroma_spectrogram)

    def generate_mel_spectrogram_thread(self, audio_tensor, sample_rate):
        n_fft = 512
        win_length = 512
        mel_spectogram = rosa.feature.melspectrogram(audio_tensor, sample_rate, n_fft=n_fft, hop_length=self.hop_length,
                                                     win_length=win_length, window=scipy.signal.windows.hanning)
        s_db = rosa.power_to_db(mel_spectogram, ref=np.max)
        if self.use_clipping:
            s_db = np.clip(s_db, a_min=-50, a_max=None)
        s_db = rosa.util.normalize(s_db)
        if self.display_spectrograms:
            rosa.display.specshow(s_db, sr=sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.show()
            print(s_db.shape)
        return tf.io.serialize_tensor(s_db)

    def generate_chroma_spectrogram_thread(self,  audio_tensor, sample_rate):
        chroma = librosa.feature.chroma_cens(y=audio_tensor, sr=sample_rate, n_chroma=128, hop_length=self.hop_length)
        chroma = rosa.util.normalize(chroma)
        if self.display_spectrograms:
            rosa.display.specshow(chroma, sr=sample_rate, y_axis='chroma', x_axis='time')
            plt.colorbar()
            plt.show()
            print(chroma.shape)
        return tf.io.serialize_tensor(chroma)

    def downsample_wavs(self,  target_sample_rate):
        paths = self.meta_data.get_source_data_paths()
        for path in paths:
            file_name = os.path.basename(path)
            y, s = rosa.load(path, sr=target_sample_rate, mono=True)
            rosa.output.write_wav(self.meta_data.work_data_path + file_name, y, target_sample_rate, norm=True)
