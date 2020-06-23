import os


class MetaData:

    def __init__(self, base_path, source_data_path, work_data_path):
        self.base_path = base_path
        self.source_data_path = base_path + source_data_path
        self.work_data_path = base_path + work_data_path
        self.dataset_size = self.get_dataset_size()

    def get_dataset_size(self):
        for (dirpath, dirnames, filenames) in os.walk(self.work_data_path):
            return len(filenames)


