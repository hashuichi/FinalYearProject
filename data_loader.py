from sklearn.model_selection import train_test_split
import pandas as pd

class DataLoader:
    """
    A utility class for loading, accessing, and splitting datasets. Supports loading data
    from specified file paths or directly from given pandas DataFrames. Additionally,
    facilitates the division of data into features and target values and splits these into
    training and testing sets.
    """

    def get_data(self):
        """
        Retrieves the current dataset stored within the DataLoader instance.

        Returns:
            pd.DataFrame: The dataset currently loaded into the DataLoader instance.
        """
        return self.data

    def get_file_name(self, data_option):
        """
        Maps the dataset selection to a file path.

        Parameters:
            data_option (str): A string identifier for the dataset.

        Returns:
            str: The file path associated with the given data_option. If the data_option
            does not match any predefined options, None is returned.
        """
        if data_option == 'Benchmark Dataset':
            return 'datasets/airbnb_london.csv'
        elif data_option == 'Structured Dataset':
            return 'datasets/fake_structured_data.csv'
        elif data_option == 'Unstructured Dataset':
            return 'datasets/fake_data.csv'
    
    def load_data(self, data_option=None, dataframe=None):
        """
        Loads data into the DataLoader instance. If a data_option is provided, it tries 
        to load data from the associated file. If a dataframe is provided, it sets this 
        DataFrame as the data for the DataLoader instance.

        Parameters:
            data_option (str, optional): The identifier for the dataset to load from file.
            dataframe (pd.DataFrame, optional): The DataFrame to directly set as the data for the instance.

        Raises:
            ValueError: If both data_option and dataframe are None, indicating that no source
            of data has been provided.
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
        Separates the dataset loaded into the DataLoader instance into features and target values.

        Returns:
            X (pd.DataFrame): The features of the dataset.
            y (pd.Series): The target values of the dataset.

        Raises:
            ValueError: If data has not been loaded before calling this method.
        """
        data = self.data.copy()
        if self.data is not None:
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            return X, y
        else:
            raise ValueError("Data has not been loaded. Call load_data() first.")

    def split_data(self):
        """
        Splits the dataset into training and testing sets. This method should be called
        after data has been loaded and features and target values have been identified.

        Returns:
            X_train (pd.DataFrame): Feature matrix of the training data.
            X_test (pd.DataFrame): Feature matrix of the test data.
            y_train (pd.Series): Target values of the training data.
            y_test (pd.Series): Target values of the test data.

        Raises:
            ValueError: If data has not been loaded before calling this method.
        """
        if self.data is not None:
            X, y = self.get_features_labels()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            raise ValueError("Data has not been loaded. Call load_data() first.")