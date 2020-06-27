import librosa
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
        self.hop_length = 1024
        self.win_length = 1024
        self.y_scale = 128
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
        y, sr = librosa.load(path, mono=True)
        audio_tensors = self.__get_tensor_split_segments(sr, y)
        i = 0
        for audio_tensor in audio_tensors:
            mel_spectrogram = self.generate_mel_spectrogram_thread(audio_tensor, sr)
            #chroma_spectrogram = self.generate_chroma_spectrogram_thread(audio_tensor, sr)
            magphase_spectrogram = self.generate_magphase_spectrogram_thread(audio_tensor, sr)
            # spectral_contrast = self.generate_spectral_contrast(audio_tensor, sr)
            # mfcc = self.generate_mfcc(audio_tensor, sr)
            # stacked_tensor = tf.stack([mel_spectrogram, chroma_spectrogram, magphase_spectrogram, mfcc])
            # stacked_tensor = tf.stack([mel_spectrogram, chroma_spectrogram])
            stacked_tensor = tf.io.serialize_tensor(mel_spectrogram)
            tf.io.write_file(self.meta_data.work_data_path + file_name[:-4] + ('_%04d' % i) + ".chr_spec",
                             stacked_tensor)
            i += 1

    def generate_mel_spectrogram_thread(self, audio_tensor, sample_rate):
        n_fft = self.win_length
        win_length = self.win_length
        mel_spectogram = librosa.feature.melspectrogram(audio_tensor, sample_rate, n_fft=n_fft,
                                                        hop_length=self.hop_length,
                                                        win_length=win_length, window=scipy.signal.windows.hanning,
                                                        n_mels=self.y_scale, power=4.0)

        s_db = librosa.power_to_db(mel_spectogram, ref=np.max)
        if self.use_clipping:
            s_db = np.clip(s_db, a_min=-50, a_max=None)
        s_db = librosa.util.normalize(s_db)
        if self.display_spectrograms:
            librosa.display.specshow(s_db, sr=sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.show()
            print(s_db.shape)
        return s_db

    def generate_chroma_spectrogram_thread(self,  audio_tensor, sample_rate):
        s = librosa.feature.chroma_stft(y=audio_tensor, sr=sample_rate, n_chroma=self.y_scale,
                                        hop_length=self.hop_length,
                                        win_length=self.win_length, n_fft=self.win_length)
        s = librosa.util.normalize(s)
        if self.display_spectrograms:
            librosa.display.specshow(s, sr=sample_rate, hop_length=self.hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.show()
        return s

    def generate_magphase_spectrogram_thread(self, audio_tensor, sample_rate):
        stft = librosa.stft(audio_tensor, n_fft=self.y_scale * 2, hop_length=self.hop_length)
        s, phase = librosa.magphase(stft)
        s = librosa.amplitude_to_db(s, ref=np.max)
        s = librosa.util.normalize(s)
        s = np.delete(s, 0, 0)
        s = librosa.util.normalize(s)
        if self.display_spectrograms:
            librosa.display.specshow(s, y_axis='log', x_axis='time')
            plt.show()
        return s

    def generate_spectral_contrast(self,  audio_tensor, sample_rate):
        stft = librosa.stft(audio_tensor, n_fft=256, hop_length=self.hop_length)
        s = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        if self.display_spectrograms:
            librosa.display.specshow(s, y_axis='log', x_axis='time')
            plt.show()
        return s

    def generate_mfcc(self,  audio_tensor, sample_rate):
        s = librosa.feature.mfcc(y=audio_tensor, sr=sample_rate, hop_length=self.hop_length, n_mfcc=self.y_scale,
                                 htk=True)
        s = librosa.util.normalize(s)
        if self.display_spectrograms:
            librosa.display.specshow(s, y_axis='log', x_axis='time')
            plt.show()
        return s

    def downsample_wavs(self,  target_sample_rate):
        paths = self.meta_data.get_source_data_paths()
        for path in paths:
            file_name = os.path.basename(path)
            y, s = librosa.load(path, sr=target_sample_rate, mono=True)
            librosa.output.write_wav(self.meta_data.work_data_path + file_name, y, target_sample_rate, norm=True)
