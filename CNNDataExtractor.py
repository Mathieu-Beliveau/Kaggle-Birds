from DataExtractorBase import DataExtractorBase


class CNNDataExtractor(DataExtractorBase):

    def __init__(self, meta_data, batch_size, dataset_size_ratio=1):
        super(CNNDataExtractor, self).__init__(meta_data, batch_size, dataset_size_ratio=1)

    def load_spectrogram_data(self, file, label):
        spectrogram = self.load_spectrogram(file)
        return spectrogram, label
