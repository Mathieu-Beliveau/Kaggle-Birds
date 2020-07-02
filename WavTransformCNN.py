from WavTransformBase import WavTransformBase
import tensorflow as tf


class WavTransformCNN(WavTransformBase):
    def __init__(self, meta_data,  audio_segment_length_in_secs=10, display_spectrograms=False, use_clipping=False):
        super(WavTransformCNN, self).__init__(meta_data, audio_segment_length_in_secs, display_spectrograms,
                                              use_clipping)

    def generate_spectrograms_thread_specialized(self, audio_tensor, sample_rate):
        mel_spectrogram = self.generate_mel_spectrogram_thread(audio_tensor, sample_rate)
        chroma_spectrogram = self.generate_chroma_spectrogram_thread(audio_tensor, sample_rate)
        magphase_spectrogram = self.generate_magphase_spectrogram_thread(audio_tensor, sample_rate)
        spectral_contrast = self.generate_spectral_contrast(audio_tensor, sample_rate)
        mfcc = self.generate_mfcc(audio_tensor, sample_rate)
        stacked_tensor = tf.stack([mel_spectrogram, chroma_spectrogram, magphase_spectrogram, mfcc])
        return stacked_tensor
