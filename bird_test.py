from CNNBirdSongsClassifier import CNNBirdSongsClassifier
from ClassifierBase import ClassifierBase
from MetaData import MetaData
from WavTransform import WavTransform
from DataCleaner import DataCleaner

base_path = "../Bird_Songs/"
source_data_path = "/test_wav/"
work_data_path = "/test_spectrograms/"
meta_data_file_path = "metadata_clean.csv"
meta_data = MetaData(base_path, source_data_path, work_data_path, meta_data_file_path)

meta_data.get_source_wavs()

# wav_transform = WavTransform.WavTransform(meta_data, display_spectrograms=False, use_clipping=False)
# meta_data_cleaner = DataCleaner.DataCleaner(meta_data)

# CNN Classification
cnn_classifier = CNNBirdSongsClassifier()
cnn_classifier.perform_training()



