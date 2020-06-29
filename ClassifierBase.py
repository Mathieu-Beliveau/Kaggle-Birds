import MetaData as Mt
import DataCleaner
import WavTransform
import os
import datetime


class ClassifierBase:

    def __init__(self, meta_data):
        self.meta_data = meta_data
        self.weights_filepath = "../Bird_Songs/Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        self.best_weights_file_path = "../Bird_Songs/Models/weights.50-1.53.hdf5"
        self.load_saved_weights = True
        self.log_dir = os.path.join('..\\Bird_Songs\\logs\\fit\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.batch_size = 10
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def get_data_extractor(self):
        raise NotImplementedError()

    def get_data(self):
        data_extractor = self.get_data_extractor()
        self.train_data, self.validation_data, self.test_data = data_extractor.get_datasets(train_ratio=0.80,
                                                                                            validation_ratio=0.10,
                                                                                            test_ratio=0.10)

    @staticmethod
    def lr_scheduler(epoch, lr):
        if epoch > 30:
            lr = 0.00001
            return lr
        return lr
