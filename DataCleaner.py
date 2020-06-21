import csv
from os import walk
import os


class DataCleaner:

    def __init__(self, meta_data):
        self.meta_data = meta_data

    def clean(self):
        self.__get_files_to_keep()
        self.__produce_cleaned_meta_data()

    def __get_files_to_keep(self):
        f = []
        for (dirpath, dirnames, filenames) in walk(self.meta_data.work_data_path):
            f.extend(filenames)
            break
        self.files_to_keep = ["mp3//" + fname[:-(len(os.path.splitext(fname)[1]) - 1)] + "mp3" for fname in f]

    def __produce_cleaned_meta_data(self):
        with open(self.meta_data.source_meta_data_path, encoding='utf-8') as csvfile:
            rdr = csv.reader(csvfile, delimiter=',', quotechar='"')
            with open(self.meta_data.work_meta_data_path, mode='w+', encoding='utf-8', newline='') as csvtrimmed:
                wrt = csv.writer(csvtrimmed, delimiter=',', quotechar='"',)
                first_iter = True
                for row in rdr:

                    if first_iter or any(filename == row[-1] for filename in self.files_to_keep):
                        wrt.writerow(row)
                        first_iter = False
                    else:
                        continue
