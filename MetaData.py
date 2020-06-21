import pandas as pd
import os



class MetaData:

    def __init__(self, base_path, source_data_path, work_data_path, source_meta_data_file_path, work_meta_data_file_path):
        self.base_path = base_path
        self.source_data_path = base_path + source_data_path
        self.work_data_path = base_path + work_data_path
        if os.path.isfile(base_path + work_meta_data_file_path):
            self.info = pd.read_csv(base_path + work_meta_data_file_path)
        else:
            self.info = pd.read_csv(base_path + source_meta_data_file_path)

        self.dataset_size = self.info.shape[0]
        self.work_meta_data_path = base_path + work_meta_data_file_path
        self.source_meta_data_path = base_path + source_meta_data_file_path

    def get_source_data_paths(self):
        return [self.source_data_path + path[4:-3] + "wav" for path in self.info["Path"]]

    def get_work_data_paths(self):
        return [self.work_data_path + path[4:-3] + "mel_spec" for path in self.info["Path"]]
