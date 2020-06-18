import librosa as rosa
import os
import tensorflow as tf

class WavTransform:

    meta_data = None

    def __init__(self, meta_data):
        self.meta_data = meta_data

    def generate_mel_spectograms(self):
        paths = self.meta_data.get_source_data_paths()
        for path in paths:
            file_name = os.path.basename(path)
            y, s = rosa.load(path, mono=True)
            mel_spectogram = rosa.feature.melspectrogram(y, s)
            serialized_mel_spectogram = tf.io.serialize_tensor(mel_spectogram)
            tf.io.write_file(self.meta_data.work_data_path + file_name[:-4] + ".mel_spec", serialized_mel_spectogram)

    def downsample_wavs(self,  target_sample_rate):
        paths = self.meta_data.get_source_data_paths()
        for path in paths:
            file_name = os.path.basename(path)
            y, s = rosa.load(path, sr=target_sample_rate, mono=True)
            rosa.output.write_wav(self.meta_data.work_data_path + file_name, y, target_sample_rate, norm=True)
