import DataExtractor as De
import DataCleaner as Dc

# data_cleaner = Dc.DataCleaner("../Bird_Songs/metadata.csv", "../Bird_Songs/metadata_trimmed.csv",
#                               "../Bird_Songs/Too_Large_Files")
# data_cleaner.clean()

# Current Largest tensor dim for wav length: 31170287
dataExtractor = De.DataExtractor("../Bird_Songs/", "metadata_trimmed.csv", 31170287)


