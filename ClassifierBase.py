from DataExtractor import DataExtractor
import MetaData as Mt
import DataCleaner
import WavTransform
import os
import datetime


class ClassifierBase:

    def __init__(self):
        self.base_path = "../Bird_Songs/"
        self.source_data_path = "/test_wav/"
        self.work_data_path = "/test_spectrograms/"
        self.weights_filepath = "../Bird_Songs/Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        self.best_weights_file_path = "../Bird_Songs/Models/weights.50-1.53.hdf5"
        self.load_saved_weights = True
        self.log_dir = os.path.join('..\\Bird_Songs\\logs\\fit\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.batch_size = 10
        self.meta_data = Mt.MetaData(self.base_path, self.source_data_path, self.work_data_path)
        self.wav_transform = WavTransform.WavTransform(self.meta_data, display_spectrograms=False, use_clipping=False)
        self.meta_data_cleaner = DataCleaner.DataCleaner(self.meta_data)
        self.train_data = None
        self.validation_data = None
        self.test_data = None

    def get_data(self):
        data_extractor = DataExtractor(self.meta_data, self.batch_size, dataset_size_ratio=1)
        self.train_data, self.validation_data, self.test_data = data_extractor.get_datasets(train_ratio=0.80,
                                                                                            validation_ratio=0.10,
                                                                                            test_ratio=0.10)

    def process_wavs(self):
        self.wav_transform.generate_spectrograms()

    def clean_meta_data(self):
        self.meta_data_cleaner.clean()

    @staticmethod
    def lr_scheduler(epoch, lr):
        if epoch > 30:
            lr = 0.00001
            return lr
        return lr
