from CNNBirdSongsClassifier import CNNBirdSongsClassifier
from MetaData import MetaData
from WavTransform import WavTransform

base_path = "../Bird_Songs/"
source_data_path = "/Source_Wav/"
work_data_path = "/test_spectrograms/"
meta_data_file_path = "metadata_clean.csv"
meta_data = MetaData(base_path, source_data_path, work_data_path, meta_data_file_path)

# wav_transform = WavTransform(meta_data, display_spectrograms=False, use_clipping=False)
# wav_transform.generate_spectrograms()

# CNN Classification
cnn_classifier = CNNBirdSongsClassifier(meta_data)
cnn_classifier.perform_training()



