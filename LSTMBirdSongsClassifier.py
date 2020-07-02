from ClassifierBase import ClassifierBase
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os


class LSTMBirdSongsClassifier(ClassifierBase):

    def __init__(self, meta_data):
        self.best_weights_file_path = "Models/weights.50-1.60.hdf5"
        super(LSTMBirdSongsClassifier, self).__init__(meta_data, batch_size=10, train_ratio=0.8,
                                                      validation_ratio=0.1, test_ratio=0.1,
                                                      load_saved_weights=False)

    def perform_training(self):
        self.get_data()
        model = models.Sequential()
        model.add(layers.LSTM(units=128, return_sequences=True,
                              input_shape=(216, 128)))
        model.add(layers.Dropout(0.1))
        model.add(layers.BatchNormalization())
        model.add(layers.LSTM(units=128, return_sequences=False))
        model.add(layers.Dropout(0.1))
        model.add(layers.BatchNormalization())
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(9, activation='softmax'))
        model.summary()
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=['accuracy'])

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.weights_filepath, save_weights_only=True)
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(ClassifierBase.lr_scheduler)
        callbacks_list = [learning_rate_scheduler, checkpoint, tensorboard_callback]

        if self.load_saved_weights and os.path.isfile(self.meta_data.base_path + self.best_weights_file_path):
            model.load_weights(self.meta_data.base_path + self.best_weights_file_path)

        model.fit(x=self.train_data, epochs=50, shuffle=True, validation_data=self.validation_data,
                  callbacks=callbacks_list)
        print("Evaluate")
        model.evaluate(self.test_data)
