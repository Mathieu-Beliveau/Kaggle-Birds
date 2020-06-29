from RNNDataExtractor import RNNDataExtractor
from ClassifierBase import ClassifierBase


class LSTMBirdSongsClassifier(ClassifierBase):

    def __init__(self, meta_data):
        super(LSTMBirdSongsClassifier, self).__init__(meta_data)

    def get_data_extractor(self):
        return RNNDataExtractor(self.meta_data, self.batch_size, dataset_size_ratio=1)

    def perform_training(self):
       pass
