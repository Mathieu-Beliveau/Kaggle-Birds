import DataExtractor as De
import MetaData as Mt
import DataCleaner
import WavTransform
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os
import datetime


class BirdSongsClassifier:

    def __init__(self):
        self.base_path = "../Bird_Songs/"
        self.source_data_path = "/Temp_Wav/"
        self.work_data_path = "/spectrograms/"
        self.source_meta_data_file_path = "metadata.csv"
        self.work_meta_data_file_path = "metadata_trimmed.csv"
        self.weights_filepath = "../Bird_Songs/Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        self.best_weights_file_path = "../Bird_Songs/Models/weights.20-1.87.hdf5"
        self.load_saved_weights = True
        self.log_dir = os.path.join('..\\Bird_Songs\\logs\\fit\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.batch_size = 5
        self.meta_data = Mt.MetaData(self.base_path, self.source_data_path, self.work_data_path,
                                     self.source_meta_data_file_path, self.work_meta_data_file_path)
        self.wav_transform = WavTransform.WavTransform(self.meta_data)
        self.dataExtractor = De.DataExtractor(self.meta_data, self.batch_size)
        self.meta_data_cleaner = DataCleaner.DataCleaner(self.meta_data)

    def process_wavs(self):
        self.wav_transform.generate_mel_spectograms()

    def clean_meta_data(self):
        self.meta_data_cleaner.clean()

    def perform_training(self):
        train_data, validation_data, test_data = self.dataExtractor.get_datasets(train_ratio=0.70,
                                                                                 validation_ratio=0.15,
                                                                                 test_ratio=0.15)

        model = models.Sequential()
        model.add(layers.Conv2D(8, (10, 10), strides=(2, 2), activation='relu', data_format='channels_last',
                                input_shape=(128, self.dataExtractor.padding_size, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(8, (10, 10), strides=(2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(8, activation='softmax'))
        model.summary()
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.weights_filepath, save_weights_only=True)
        callbacks_list = [checkpoint, tensorboard_callback]

        if self.load_saved_weights and os.path.isfile(self.best_weights_file_path):
            model.load_weights(self.best_weights_file_path)

        history = model.fit(x=train_data, epochs=20, validation_data=validation_data, callbacks=callbacks_list)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        print("Evaluate")
        model.evaluate(test_data)
