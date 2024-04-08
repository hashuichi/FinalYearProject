# Final Year Project
This is a project to demonstrate a comparison of various machine learning algorithms in the realm of optimising prices for hotels to find the best prices to maximise revenue.

## Project Directory Structure
The project main directory contains a super class base_model.py along with the machines learning models classes that extend its functions. Additionally, you will also find other relevant classes including the main file that runs the app Data.py. Also, you can find multiple folders containing relevant files such as datasets (contains the used datasets and the code used to generate them), pages (contains the pages in the web app with the data page being the main file Data.py in the main directory) and tests (contains the test files). The reasoning for the names of the pages files (Data.py, 1_Nearest_Neighbours.py etc.) is due to the conventions of the streamlit library to detect and order the pages in the web app.

## Running the app
The project can be run by running the command "**streamlit run Data.py**" in the terminal. 

Here is a list of the main packages used by the app:
* streamlit
* scikit-learn
* numpy
* matplotlib
* pandas
* keras

These packages can be downloaded using the command "**pip install [package_name]**". Downloading these should make the app runnable. In the case the app is still not running, refer to requirements.txt for a list of all the versions of the used packages.

## Running the tests

The test files can be run using the command "**python -m tests.[test_file]**" where [test_file] represents the file to be tested. There are 4 test files, one for each algorithm:
* test_knn
* test_lr
* test_dt
* test_nn

For example to run the tests for 'test_knn.py' use the command "**python -m tests.test_knn**"

## Video Link

Here is the link to the video showing how to run the app, the app’s features and how to run the test files.

https://youtu.be/CdNedJdxu_s