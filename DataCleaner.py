import csv
from os import walk


class DataCleaner:

    original_meta_data_path = None
    cleaned_meta_data_path = None
    path_for_files_to_exclude = None
    files_to_clean = []

    def __init__(self, original_meta_data_path, cleaned_meta_data_path, path_for_files_to_exclude):
        self.original_meta_data_path = original_meta_data_path
        self.cleaned_meta_data_path = cleaned_meta_data_path
        self.path_for_files_to_exclude = path_for_files_to_exclude

    def clean(self):
        self.__get_files_to_clean()
        self.__produce_cleaned_meta_data()

    def __get_files_to_clean(self):
        f = []
        for (dirpath, dirnames, filenames) in walk(self.path_for_files_to_exclude):
            f.extend(filenames)
            break
        self.files_to_clean = ["mp3//" + fname[:-3] + "mp3" for fname in f]

    def __produce_cleaned_meta_data(self):
        with open(self.original_meta_data_path, encoding='utf-8') as csvfile:
            rdr = csv.reader(csvfile, delimiter=',', quotechar='"')
            with open(self.cleaned_meta_data_path, mode='w+', encoding='utf-8', newline='') as csvtrimmed:
                wrt = csv.writer(csvtrimmed, delimiter=',', quotechar='"',)
                i = 0
                for row in rdr:
                    if any(filename == row[-1] for filename in self.files_to_clean):
                        continue
                    else:
                        wrt.writerow(row)
