from sklearn.model_selection import train_test_split
import pandas as pd

class DataLoader:
    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def get_file_name(self, data_option):
        if data_option == 'Benchmark Dataset':
            return 'datasets/airbnb_london.csv'
        elif data_option == 'Structured Dataset':
            return 'datasets/fake_structured_data.csv'
        elif data_option == 'Unstructured Dataset':
            return 'datasets/fake_data.csv'
    
    def load_data(self, data_option=None, dataframe=None):
        """
        Loads the data from either a data option or a dataframe.

        Parameters:
        data_option (string): File to import data from.
        dataframe (pd.DataFrame): Dataframe to set the objects data.
        """
        file_name = self.get_file_name(data_option)
        if (file_name is not None):
            self.data = pd.read_csv(file_name)
        elif (dataframe is not None):
            self.data = dataframe
        else:
            raise ValueError("Data has not been loaded. Missing file_name or dataframe")

    def get_features_labels(self):
        """
        Reads the fake data file and returns the features and labels as separate variables.

        Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        """
        data = self.data.copy()
        if self.data is not None:
            X = data.drop('price', axis=1)
            y = data['price']
            return X, y
        else:
            raise ValueError("Data has not been loaded. Call load_data() first.")

    def split_data(self):
        """
        Splits the data into training and testing sets.

        Returns:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Testing labels
        """
        if self.data is not None:
            X, y = self.get_features_labels()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            raise ValueError("Data has not been loaded. Call load_data() first.")