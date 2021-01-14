import pandas as pd


class DataLoader:
    def __init__(self, config):
        self.base_data_path = config.labels_generator.paths.base_data_path
        self.test_data_path = config.defects_classifier.paths.test_data_path
        self.generated_labels_path = config.labels_generator.paths.generated_labels_path

        self.test_df = self.load_data(self.test_data_path)
        self.positive_labelled_df = self.load_data(self.generated_labels_path)
        self.base_df = self.load_data(self.base_data_path)
        self.train_df = self.load_training_data()

        self.test_protocol = config.defects_classifier.evaluate.protocol

    @staticmethod
    def load_data(path):
        return pd.read_csv(path, encoding='utf-8')

    def load_training_data(self):
        pos_train_df = self.load_pos_training_data()
        unlabelled_train_df = self.load_unlablled_training_data()
        train_df = pd.concat([pos_train_df, unlabelled_train_df])
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        return train_df

    def load_pos_training_data(self):
        """
        Creates a dataframe for positive-labelled training data that excludes rows in self.test_df
        :return: pd dataframe
        """
        pos_train_df = self.positive_labelled_df.merge(self.test_df.drop_duplicates(),
                                                       on=['col1', 'col2'],
                                                       how='left',
                                                       indicator=True)
        pos_train_df['_merge'] == 'left_only'
        return pos_train_df

    def load_unlabelled_training_data(self):
        """
        Creates a dataframe for unlabelled training data that excludes rows in self.test_df and self.positive_labelled_df
        :return: pd dataframe
        """
        unlabelled_train_df = self._remove_duplicates(self.base_df, self.test_df)
        unlabelled_train_df = self._remove_duplicates(unlabelled_train_df, self.positive_labelled_df)
        unlabelled_train_df['_merge'] == 'left_only'
        return unlabelled_train_df

    @staticmethod
    def _remove_duplicates(df1, df2):
        """
        removes duplicated rows in df2 from df1
        """
        dedup_df = df1.merge(df2.drop_duplicates(),
                                        on=['col1', 'col2'],
                                        how='left',
                                        indicator=True)
        return dedup_df

    def get_training_data(self):
        return self.train_df

    def get_testing_data(self):
        return self.test_df
