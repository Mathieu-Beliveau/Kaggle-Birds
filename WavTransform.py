import librosa as rosa
import os


class WavTransform:

    meta_data = None

    def __init__(self, meta_data):
        self.meta_data = meta_data

    def downsample_wavs(self,  target_sample_rate, target_path):
        paths = self.meta_data.get_source_data_paths()
        for path in paths:
            file_name = os.path.basename(path)
            y, s = rosa.load(path, sr=target_sample_rate, mono=True)
            rosa.output.write_wav(target_path + file_name, y, target_sample_rate, norm=True)
