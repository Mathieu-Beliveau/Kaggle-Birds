import DataExtractor as De
import MetaData as Mt
import DataCleaner
import WavTransform
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import os
import datetime


class BirdSongsClassifier:

    def __init__(self):
        self.base_path = "../Bird_Songs/"
        self.source_data_path = "/test_wav/"
        self.work_data_path = "/test_spectrograms/"
        self.weights_filepath = "../Bird_Songs/Models/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        self.best_weights_file_path = "../Bird_Songs/Models/weights.50-1.68.hdf5"
        self.load_saved_weights = True
        self.log_dir = os.path.join('..\\Bird_Songs\\logs\\fit\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.batch_size = 10
        self.meta_data = Mt.MetaData(self.base_path, self.source_data_path, self.work_data_path)
        self.wav_transform = WavTransform.WavTransform(self.meta_data, display_spectrograms=False, use_clipping=False)
        self.meta_data_cleaner = DataCleaner.DataCleaner(self.meta_data)

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

    def perform_training(self):
        data_extractor = De.DataExtractor(self.meta_data, self.batch_size, dataset_size_ratio=1)
        train_data, validation_data, test_data = data_extractor.get_datasets(train_ratio=0.80,
                                                                             validation_ratio=0.10,
                                                                             test_ratio=0.10)

        model = models.Sequential()
        model.add(layers.Conv2D(16, (10, 10), strides=(2, 2), input_shape=(128, 216, 1), activation='relu',
                                data_format='channels_last'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (10, 10), strides=(2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(9, activation='softmax'))
        model.summary()
        optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.weights_filepath, save_weights_only=True)
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(BirdSongsClassifier.lr_scheduler)
        callbacks_list = [learning_rate_scheduler, checkpoint, tensorboard_callback]

        if self.load_saved_weights and os.path.isfile(self.best_weights_file_path):
            model.load_weights(self.best_weights_file_path)

        history = model.fit(x=train_data, epochs=50, shuffle=True, validation_data=validation_data,
                            callbacks=callbacks_list)

        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        print("Evaluate")
        model.evaluate(test_data)
