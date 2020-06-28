from RNNDataExtractor import RNNDataExtractor
from ClassifierBase import ClassifierBase


class LSTMBirdSongsClassifier(ClassifierBase):

    def get_data_extractor(self):
        return RNNDataExtractor(self.meta_data, self.batch_size, dataset_size_ratio=1)

    def perform_training(self):
       pass
