from DataExtractorBase import DataExtractorBase


class RNNDataExtractor(DataExtractorBase):
    def __load_spectrogram_data(self, file, label):
        spectrogram = self.load_spectrogram(file)
        return spectrogram, label
