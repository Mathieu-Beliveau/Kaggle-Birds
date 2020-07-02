from WavTransformBase import WavTransformBase


class WavTransformLSTM(WavTransformBase):
    def __init__(self, meta_data, audio_segment_length_in_secs=10, display_spectrograms=False, use_clipping=False):
        super(WavTransformLSTM, self).__init__(meta_data, audio_segment_length_in_secs, display_spectrograms,
                                               use_clipping)

    def generate_spectrograms_thread_specialized(self, audio_tensor, sample_rate):
        return self.generate_mel_spectrogram_thread(audio_tensor, sample_rate).T

