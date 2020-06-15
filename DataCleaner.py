import csv
from os import walk


class DataCleaner:
    original_meta_data = '../Bird_Songs/metadata.csv'
    cleaned_meta_data = '../Bird_Songs/metadata_trimmed.csv'
    folder_of_files_to_exclude = "../Bird_Songs/Too_Large_Files"
    files_to_clean = []

    def __init__(self):
        pass

    def clean(self):
        self.__get_files_to_clean()
        self.__produce_cleaned_meta_data()

    def __get_files_to_clean(self):
        f = []
        for (dirpath, dirnames, filenames) in walk(self.folder_of_files_to_exclude):
            f.extend(filenames)
            break
        self.files_to_clean = ["mp3//" + fname[:-3] + "mp3" for fname in f]

    def __produce_cleaned_meta_data(self):
        with open(self.original_meta_data, encoding='utf-8') as csvfile:
            rdr = csv.reader(csvfile, delimiter=',', quotechar='"')
            with open(self.cleaned_meta_data, mode='w+', encoding='utf-8', newline='') as csvtrimmed:
                wrt = csv.writer(csvtrimmed, delimiter=',', quotechar='"',)
                i = 0
                for row in rdr:
                    if any(filename == row[-1] for filename in self.files_to_clean):
                        continue
                    else:
                        wrt.writerow(row)
