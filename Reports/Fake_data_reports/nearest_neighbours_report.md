# Nearest Neighbours Results Report

## MSE Performance
The Mean Squared Error (MSE) of 16987.4748046875 does seem relatively high, indicating that the K-nearest neighbors (KNN) model may not be performing very well on this dataset. A lower MSE would indicate better predictive performance.

There are a few potential reasons for the relatively high MSE:

1. **Feature Engineering**: The features in the dataset may not be well-suited for predicting room prices accurately. I might want to consider if there are additional features that could improve prediction accuracy.
2. **Outliers**: Outliers in the data can significantly affect the KNN model's performance. I might want to check for and handle outliers in the data.
3. **Data Quality**: Data quality issues, such as missing values or inaccuracies, can also impact model performance. This is the most likely reason as the data is completely fake and is not based on real life data.

## Prediction Performance
Considering how low the quality of the data is, the model performed quite well when it came to price prediction. However, this is not to say that it was accurate. Just that it did well considering the state of the current data.
