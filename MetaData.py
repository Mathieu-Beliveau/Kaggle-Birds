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

    def get_source_wavs(self):
        dataframe = self.__get_filtered_csv_data()
        paths = [self.source_data_path + path[4:-4] + ".wav" for path in dataframe['Path']]
        return paths

    # Apply defined filters to the CSV meta data file
    def __get_filtered_csv_data(self):
        csv_data = self.csv_data
        filters_fn = self.__get_filters()
        for filter_fn in filters_fn:
            csv_data = filter_fn(csv_data)
        return csv_data

    def __get_filters(self):
        return [lambda df: self.filter_species(df, ['Sonus naturalis',
                                                    'Fringilla coelebs',
                                                    'Parus major',
                                                    'Turdus merula',
                                                    'Turdus philomelos',
                                                    'Sylvia communis',
                                                    'Emberiza citrinella',
                                                    'Sylvia atricapilla',
                                                    'Emberiza calandra'], sample_ratio=1),
                lambda df: self.filter_audio_length(df, max_length_in_minutes=20),
                lambda df: self.filter_multi_species_audio(df, max_other_species_allowed=3)]

    def load_file_and_labels_dataset(self):
        suffled_paths_file = self.base_path + "suffled_paths.pkl"
        if os.path.isfile(suffled_paths_file):
            with open(suffled_paths_file, 'rb') as f:
                files_with_one_hot_labels = pickle.load(f)
        else:
            files_with_meta_data = self.get_working_files_meta_data()
            files_with_one_hot_labels = self.__transform_file_labels_to_one_hot_vector(files_with_meta_data)
            shuffle(files_with_one_hot_labels)
            with open(suffled_paths_file, 'wb') as f:
                pickle.dump(files_with_one_hot_labels, f)
        shuffled_paths = [path for path, _ in files_with_one_hot_labels]
        shuffled_labels = [label for _, label in files_with_one_hot_labels]
        return shuffled_paths, shuffled_labels

    def get_working_files_meta_data(self):
        files_grouped_by_recording = self.group_files_by_recording_id()
        files_with_meta_data = []
        for recording_id in files_grouped_by_recording:
            meta_data = self.get_matching_label_for_recording_id(recording_id)
            for file in files_grouped_by_recording[recording_id]:
                files_with_meta_data.append((file, meta_data))
        return files_with_meta_data

    def group_files_by_recording_id(self):
        grouped_files = {}
        for (dirpath, dirnames, filenames) in os.walk(self.work_data_path):
            paths = filenames
            for filename in paths:
                recording_id = MetaData.get_recording_id_from_filename(filename)
                if recording_id not in grouped_files:
                    grouped_files[recording_id] = []
                grouped_files[recording_id].append(filename)
            break
        return grouped_files

    @staticmethod
    def get_recording_id_from_filename(file_name):
        suffix = file_name.split("-")[2]
        return suffix.split("_")[0]

    def get_matching_label_for_recording_id(self, recording_id):
        meta_data = self.csv_data[self.csv_data['Recording_ID'].isin([int(recording_id)])]
        label = MetaData.get_label_for_meta_data_row(meta_data)
        return label

    @staticmethod
    def get_label_for_meta_data_row(meta_data_row):
        call_types = meta_data_row['Vocalization_type'].item()
        call_types_arr = call_types.split(',')
        male = "male"
        female = "female"
        if female in call_types_arr and male in call_types_arr:
            return meta_data_row['Species'].item()
        elif female in call_types_arr:
            return meta_data_row['Species'].item() + "_" + female
        elif male in call_types_arr:
            return meta_data_row['Species'].item() + "_" + male
        else:
            return meta_data_row['Species'].item()

    # Filters
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
    def __transform_file_labels_to_one_hot_vector(files_with_label):
        labels = [label for _, label in files_with_label]
        labels_df = pd.DataFrame(labels)
        one_hot_labels = pd.get_dummies(labels_df, dtype=float).values
        return [(file_with_label[0], one_hot_label) for one_hot_label, file_with_label in zip(one_hot_labels,
                                                                                              files_with_label)]

