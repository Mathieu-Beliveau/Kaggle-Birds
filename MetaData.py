import pandas as pd

class MetaData:

    meta_data = None

    def __init__(self, source_data_path, work_data_path, meta_data_file_path):
        self.source_data_path = source_data_path
        self.work_data_path = work_data_path
        self.info = pd.read_csv(meta_data_file_path)
        self.dataset_size = self.info.shape[0]

    def get_source_data_paths(self):
        return [self.source_data_path + path[4:-3] + "wav" for path in self.info["Path"]]

    def get_work_data_paths(self):
        return [self.work_data_path + path[4:-3] + "wav" for path in self.info["Path"]]