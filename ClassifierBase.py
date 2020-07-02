import MetaData as Mt
import WavTransformBase
import os
import datetime
from DataExtractor import DataExtractor


class ClassifierBase:

    def __init__(self, meta_data, batch_size, train_ratio=0.8,
                 validation_ratio=0.1, test_ratio=0.1, load_saved_weights=True):
        self.meta_data = meta_data
        self.weights_filepath = "../Bird_Songs/Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        self.load_saved_weights = load_saved_weights
        self.log_dir = os.path.join('..\\Bird_Songs\\logs\\fit\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.valitation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.input_shape = None
        self.label_size = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def get_data(self):
        data_extractor = DataExtractor(self.meta_data, self.batch_size)
        self.label_size = data_extractor.label_vector_size
        self.input_shape = data_extractor.input_shape
        self.train_data, self.validation_data, self.test_data = \
            data_extractor.get_datasets(train_ratio=self.train_ratio,
                                        validation_ratio=self.valitation_ratio,
                                        test_ratio=self.valitation_ratio)

    @staticmethod
    def lr_scheduler(epoch, lr):
        if epoch > 30:
            lr = 0.00001
            return lr
        return lr
