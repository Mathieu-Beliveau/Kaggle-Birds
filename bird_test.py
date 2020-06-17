import DataExtractor as De
import tensorflow as tf
import MetaData as Mt
import DataCleaner as Dc
import WavTransform as Wt

# Current Largest tensor dim for wav length: 31170287
# padding_size = 31170287

padding_size = 5240277
source_data_path = "../Bird_Songs/Wav/"
work_data_path = "../Bird_Songs/downsampled_16bits/"
feature_means_file_path = "../Bird_Songs/means.tf"
feature_variance_file_path = "../Bird_Songs/variance.tf"
source_meta_data_file_path = "../Bird_Songs/metadata.csv"
work_meta_data_file_path = "../Bird_Songs/metadata_trimmed.csv"

# data_cleaner = Dc.DataCleaner(source_meta_data_file_path, work_meta_data_file_path,
#                               "../Bird_Songs/Too_Large_Files")
# data_cleaner.clean()

meta_data = Mt.MetaData(source_data_path, work_data_path, work_meta_data_file_path)
dataExtractor = De.DataExtractor(meta_data, padding_size)
dataExtractor.get_dataset_feature_means()
dataExtractor.get_dataset_feature_variance()
#
# train_data, validation_data, test_data = dataExtractor.get_datasets(train_ratio=0.70, validation_ratio=0.15,
#                                                                     test_ratio=0.15)


# dataset = dataExtractor.create_dataset(epochs=1, batch_size=10)
# dataExtractor.get_dataset_feature_means(max_wav_tensor_size, feature_means_file_path)
# dataExtractor.get_dataset_feature_variance(max_wav_tensor_size, feature_means_file_path,
#                                            feature_variance_file_path)
# dataExtractor.downsample_wav(8000, dataset_location_prefix + "downsampled_wavs/")
