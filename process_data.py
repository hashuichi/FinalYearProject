from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd


def get_fake_data():
    """
    Reads the non-structured fake data file and returns the features and labels as separate variables.

    Returns:
    X (pd.DataFrame): Features.
    y (pd.Series): Labels.
    """
    data = pd.read_csv('fake_data.csv')
    data['Hotel Name'] = LabelEncoder.fit_transform(data['Hotel Name'])
    X = data.drop('Price', axis=1)
    y = data['Price']
    return X, y

def get_fake_structured_data():
    """
    Reads the fake structured data file and returns the features and labels as separate variables.
    
    Returns:
    X (pd.DataFrame): Features.
    y (pd.Series): Labels.
    """
    data = pd.read_csv("fake_structured_data.csv")
    X = data[["star_rating", "distance"]]
    y = data["price"]
    return X, y

def split_data(X, y):
    """
    Splits the data into training and testing sets.

    Parameters:
    X (pd.DataFrame): Features.
    y (pd.series): Labels.

    Returns:
    X_train (pd.DataFrame): Training features
    X_test (pd.DataFrame): Testing features
    y_train (pd.Series): Training labels
    y_test (pd.Series): Testing labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test