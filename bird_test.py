from CNNBirdSongsClassifier import CNNBirdSongsClassifier
from ClassifierBase import ClassifierBase

classifier_base = ClassifierBase()
# classifier_base.process_wavs()

# CNN Classification
cnn_classifier = CNNBirdSongsClassifier()
cnn_classifier.perform_training()



