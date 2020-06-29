import pandas as pd
import pickle
import os
from random import shuffle

class MetaData:

    def __init__(self, base_path, source_data_path, work_data_path, meta_data_path):
        self.base_path = base_path
        meta_data_path = base_path + meta_data_path
        self.source_data_path = base_path + source_data_path
        self.work_data_path = base_path + work_data_path
        self.csv_data = pd.read_csv(meta_data_path)

    def __load_file_and_labels_dataset(self):
        suffled_paths_file = self.base_path + "suffled_paths.pkl"
        if os.path.isfile(suffled_paths_file):
            with open(suffled_paths_file, 'rb') as f:
                paths_labels_pairs = pickle.load(f)
        else:
            files, meta_data = self.get_working_files_meta_data()
            one_hot_labels = self.__transform_labels_to_one_hot_vector(meta_data)
            paths_labels_pairs = [(file, one_hot_label) for file, one_hot_label in zip(files, one_hot_labels)]
            shuffle(paths_labels_pairs)
            with open(suffled_paths_file, 'wb') as f:
                pickle.dump(paths_labels_pairs, f)
        shuffled_paths = [path for path, _ in paths_labels_pairs]
        shuffled_labels = [label for _, label in paths_labels_pairs]
        return shuffled_paths, shuffled_labels

    def get_working_files_meta_data(self):
        meta_data = []
        dirpath, dirnames, filenames = os.walk(self.work_data_path)
        for filename in filenames:
            file_meta_data = self.get_matching_meta_data_for_file(filename)
            meta_data.append(file_meta_data)
        return filenames, meta_data

    def get_matching_meta_data_for_file(self, file_name):
        suffix = file_name.split("-")[2]
        recording_id = suffix.split("_")[0]
        return self.csv_data['Recording_ID'].match(recording_id)

    def get_source_wavs(self):
        dataframe = self.__get_filtered_csv_data()
        paths = [self.source_data_path + path[4:-4] + ".wav" for path in dataframe['Path']]
        return paths

    def __get_filtered_csv_data(self):
        csv_data = self.csv_data
        filters_fn = self.__get_filters()
        for filter_fn in filters_fn:
            csv_data = filter_fn(csv_data)
        return csv_data

    # Filters

    def __get_filters(self):
        return [lambda df: self.filter_species(df, ['Sonus naturalis'], sample_ratio=0.7),
                lambda df: self.filter_audio_length(df, max_length_in_minutes=20),
                lambda df: self.filter_multi_species_audio(df, max_other_species_allowed=22)]

    @staticmethod
    def filter_multi_species_audio(dataframe, max_other_species_allowed=2):
        mask = MetaData.create_mask_for_rows_having_more_than_n_species(dataframe, max_other_species_allowed)
        return dataframe[mask]

    @staticmethod
    def create_mask_for_rows_having_more_than_n_species(dataframe, max_species):
        return dataframe.iloc[:, 5:35].applymap(MetaData.occurrence_to_int).sum(axis=1) < max_species

    @staticmethod
    def occurrence_to_int(elm):
        if pd.isnull(elm):
            return 0
        else:
            return 1

    @staticmethod
    def filter_species(dataframe, species_to_filter, sample_ratio=1.0):
        mask = dataframe['Species'].isin(species_to_filter)
        specie = dataframe[mask]
        return specie.sample(frac=sample_ratio)

    @staticmethod
    def filter_audio_length(dataframe, min_length_in_minutes=None, max_length_in_minutes=None):
        dataframe = MetaData.convert_length_column_to_secs(dataframe)
        if min_length_in_minutes is None and max_length_in_minutes is None:
            return dataframe
        elif min_length_in_minutes is not None and max_length_in_minutes is None:
            return dataframe[dataframe['Length'] > (min_length_in_minutes * 60)]
        elif min_length_in_minutes is None and max_length_in_minutes is not None:
            return dataframe[dataframe['Length'] < (max_length_in_minutes * 60)]
        else:
            return dataframe[(min_length_in_minutes * 60) < dataframe['Length'] < (max_length_in_minutes * 60)]

    @staticmethod
    def convert_length_column_to_secs(dataframe):
        dataframe['Length'] = dataframe['Length'].map(MetaData.convert_length_to_secs)
        return dataframe

    @staticmethod
    def convert_length_to_secs(str_length):
        if str_length is None:
            return 0
        length_parts = str_length.split(':')
        mins = int(length_parts[0])
        secs = int(length_parts[1])
        return mins * 60 + secs

    @staticmethod
    def __transform_labels_to_one_hot_vector(labels):
        labels_df = pd.DataFrame(labels)
        return pd.get_dummies(labels_df, dtype=float).values

