# Term 1

## Week 1 (September 25th)
- Found research papers related to the topic
- Completed project plan initial draft
- Startd researching about the implementation of the machine learning algorithms

## Week 2 (October 2nd)
- Completed final draft for the project plan
- Researched technologies to use for the project

## Week 3 (October 9th)
- Researched data fields needed for a hotel pricing dataset
- Generated fake dataset

## Week 4 (October 16th)
- Created implementation for nearest neighbours
- Used hyperparameter tuning to find best K
- Found MSE
- Predicted price of new data
- Reported the results

## Week 5 (October 23rd)
- Refactored code by creating 3 separate files:
    - main.py
    - nearest_neighbours.py
    - gui.py
- Created GUI with Streamlit to:
    - Predict hotel price from user input
    - Represent MSE of Nearest Neighbours algorithm with different k values on a graph
    - Display best k and mse values for the training set

## Week 6 (October 30th)
- Generated structured data based on star rating and distances (Replaces old data generated with Faker)
- Generated graphs to check structure of data with matplotlib
- Scaled data to fit the correct range

## Week 7 (November 6th)
- Rewrote Nearest Neighbours to fit new data
- Wrote unit tests for testing nearest neighbours
- Started working on interim report
- Added a navbar and created pages to prepare for the development of the other algorithms
- Added charts to represent the data quality on the homepage
- Refactored code by moving gui.pu methods to nearest_neighbours.py

## Week 8 (November 13th)
- Created a separate DataLoader.py file generalising data file processing in preparation for the development of the rest of the algorithms
- Prepared GUI fully to prepare for the rest of the algorithms
- Added a radio selection to change the data set

## Week 9 (November 20th)
- Changed Home, KNN and DataLoader to classes to follow OOP
- Modified testknn.py to fit new oop changes of knn
- Split algorithms and pages for dedicated algorithms into separate files
- Implemented test class for Linear Regression
- Implemented Linear Regression
- Evaluated performance of Linear Regression
- Implemented a super class BaseModel to instantiate common parameters and methods for the alogrithms


## Week 10 (November 27th)
- Implemented test class for Decision Tree
- Implemented Decision Tree
- Evaluated performance of Decision Tree
- Finished writing interim report
- Finished making the presentation to prepare for viva

__________________________

# Term 2

## Week 1 (January 15th)
- Found a few possible benchmark datasets on kaggle
- Selected the best dataset (London Airbnb Listings)

## Week 2 (January 22nd)
- Started preprocessing and normalising the chosen dataset
- Cleaned dataset by removing 20000 rows of redundant data
- Incorporated the new dataset as an option in the select box
- Started implementing session states from Streamlit to differentiate between the old datasets and the new benchmark dataset

## Week 3 (January 29th)
- Removed session states as it did not work as intended. So reverted to simply using a global variable to differentiate the chosen dataset
- Added new input boxes for the new features of the dataset
- Rewrote knn tests to work for new methods
- Started rewriting knn from scratch
- Encountered errors due to the room type being a string so had to map the room type values to integers

## Week 4 (February 5th)
- Finished rewriting knn from scratch
- Refactored code to ensure program still works for both old and new datasets
- Created a variation of the Data page for the benchmark dataset which includes:
    - Table of the data
    - A key for room type to int mapping
    - Plots to visualise the data frequency and quality

## Week 5 (February 12th)
- Rewrote lr tests to work for new methods
- Finished rewriting Linear Regression from scratch using two methods:
    - Analytical method of normal equation
    - Iterative method of gradient descent
- Refactored to ensure other algorithms still work as expected

## Week 6 (February 19th)
- Rewrote dt tests to work for new methods
- Finished rewriting Decision Tree from scratch (Took longer than expected due to the complexity of the grow_tree method)

## Week 7 (February 26th)
- Wrote neural net tests with the expectation I will implement two different architectures:
    - Feedforward Neural Networks
    - Recurrent Neural Networks (Specifically long-short term memory network)
- Implemented Feedforward Neural Networks algorithm with keras
- Implemented methods to calculate y_pred and rmse
- Separated contents into 3 tabs for visibility

## Week 8 (March 4th)
- Merged Neural Networks commits
- Starting working on the report

## Week 9 (March 11th)
- Implemented Recurrent Neural Networks with keras
- Implemented methods to calculate y_pred and rmse for Recurrent networks
- Renamed methods and variables to differentiate between feedforward and recurrent

## Week 10 (March 18th)
- Cached algorithm results using streamlit to speed up loading times when adjusting parameters for prediction and visualisation
- Completed the following sections of the report:
    - Introduction
    - Machine Learning
    - Professional Issues

## Week 11 (March 25th)
- Attend supervisor meeting
- Complete final adjustments and refactoring to code
- Finalise the report
- Prepare the poster
