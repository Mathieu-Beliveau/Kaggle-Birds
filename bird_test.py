import DataExtractor as de
import DataCleaner as dc

# data_cleaner = dc.DataCleaner()
# data_cleaner.clean()


x = 0
dataExtractor = de.DataExtractor()
max_shape = 0
for wav, one_hot in dataExtractor.dataset:
    if wav.shape[0] > max_shape:
        max_shape = wav.shape[0]
        print(max_shape)
