from ClassifierBase import ClassifierBase
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os


class LSTMBirdSongsClassifier(ClassifierBase):

    def perform_training(self):
        self.get_data()
        model = models.Sequential()
        model.add(layers.Conv2D(16, (10, 10), strides=(2, 2), input_shape=(128, 216, 3), activation='relu',
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
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(self.lr_scheduler)
        callbacks_list = [learning_rate_scheduler, checkpoint, tensorboard_callback]

        if self.load_saved_weights and os.path.isfile(self.best_weights_file_path):
            model.load_weights(self.best_weights_file_path)

        model.fit(x=self.train_data, epochs=50, shuffle=True, validation_data=self.validation_data,
                  callbacks=callbacks_list)

        print("Evaluate")
        model.evaluate(self.test_data)
