import DataExtractor as De
import MetaData as Mt
import DataCleaner as Dc
import WavTransform as Wt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Current Largest tensor dim for wav length: 31170287
# padding_size = 31170287
# padding_size = 2875046 (downsampled wav)
batch_size = 10
padding_size = 15478
source_data_path = "../Bird_Songs/Wav/"
work_data_path = "../Bird_Songs/spectograms/"
source_meta_data_file_path = "../Bird_Songs/metadata.csv"
work_meta_data_file_path = "../Bird_Songs/metadata_trimmed.csv"
dataset_means_file_path = "../Bird_Songs/means.tf"
dataset_standard_deviation_file_path = "../Bird_Songs/standard_deviation.tf"

meta_data = Mt.MetaData(source_data_path, work_data_path, work_meta_data_file_path)

# Produce down sampled data from original source
# wt = Wt.WavTransform(meta_data)
# wt.generate_mel_spectograms()

dataExtractor = De.DataExtractor(meta_data, padding_size, batch_size, None, None)

train_data, validation_data, test_data = dataExtractor.get_datasets(train_ratio=0.70, validation_ratio=0.15,
                                                                    test_ratio=0.15)
x = 1
#
model = models.Sequential()
model.add(layers.Conv2D(8, (128, 5), strides=(1, 1), activation='relu', data_format='channels_last'))
model.add(layers.MaxPooling2D((1, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(50, activation='softmax'))
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x=train_data, epochs=10, validation_data=validation_data)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


# Utility code

# Extract means and variance for every feature of the dataset
# dataExtractor.get_dataset_mean(padding_size, dataset_means_file_path)
# dataExtractor.get_dataset_standard_deviation(padding_size, dataset_means_file_path,
#                                              dataset_standard_deviation_file_path)



# Create new meta data from filtering out excluded files
# data_cleaner = Dc.DataCleaner(source_meta_data_file_path, work_meta_data_file_path,
#                               "../Bird_Songs/Too_Large_Files")
# data_cleaner.clean()
