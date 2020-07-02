from WavTransformBase import WavTransformBase
import tensorflow as tf


class WavTransformConvLSTM2D(WavTransformBase):
    def __init__(self, meta_data,  audio_segment_length_in_secs=10, display_spectrograms=False, use_clipping=False):
        super(WavTransformConvLSTM2D, self).__init__(meta_data, audio_segment_length_in_secs, display_spectrograms,
                                                     use_clipping)

    def generate_spectrograms_thread_specialized(self, audio_tensor, sample_rate):
        mel_spectrogram = self.generate_mel_spectrogram_thread(audio_tensor, sample_rate)
        mel_spectrogram_time_series = self.transform_spectrogram_data_into_time_series(mel_spectrogram)
        return mel_spectrogram_time_series

    @staticmethod
    def transform_spectrogram_data_into_time_series(spectrogram_tensor):
        tensors = []
        hop_size = 8
        upper_limit = 16
        for i in range(0, spectrogram_tensor.shape[1], hop_size):
            if upper_limit > spectrogram_tensor.shape[1]:
                break
            tensors.append(spectrogram_tensor[:, i:upper_limit])
            upper_limit += hop_size
        time_spectrogram = tf.stack(tensors, axis=0)
        return time_spectrogram
