from DataExtractorBase import DataExtractorBase


class CNNDataExtractor(DataExtractorBase):
    def __load_spectrogram_data(self, file, label):
        spectrogram = self.load_spectrogram(file)
        return spectrogram, label
