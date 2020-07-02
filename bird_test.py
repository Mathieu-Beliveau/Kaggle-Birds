from MetaData import MetaData
from CNNBirdSongsClassifier import CNNBirdSongsClassifier
from WavTransformCNN import WavTransformCNN
from ConvLSTM2DBirdSongsClassifier import ConvLSTM2DBirdSongsClassifier
from WavTransformConvLSTM2D import WavTransformConvLSTM2D
from LSTMBirdSongsClassifier import LSTMBirdSongsClassifier
from WavTransformLSTM import WavTransformLSTM

base_path = "../Bird_Songs/"
source_data_path = "/Source_Wav/"
work_data_path = "/test_spectrograms/"
meta_data_file_path = "metadata_clean.csv"
meta_data = MetaData(base_path, source_data_path, work_data_path, meta_data_file_path)

# CNN Classification
# wav_transform = WavTransformCNN(meta_data)
# wav_transform.generate_spectrograms()

# cnn_classifier = CNNBirdSongsClassifier(meta_data)
# cnn_classifier.perform_training()


# ConvLSTM2D Classification
# wav_transform = WavTransformConvLSTM2D(meta_data)
# wav_transform.generate_spectrograms()

# lstm_classifier = ConvLSTM2DBirdSongsClassifier(meta_data)
# lstm_classifier.perform_training()

# Simple LSTM Classification
# wav_transform = WavTransformLSTM(meta_data)
# wav_transform.generate_spectrograms()
lstm_classifier = LSTMBirdSongsClassifier(meta_data)
lstm_classifier.perform_training()




