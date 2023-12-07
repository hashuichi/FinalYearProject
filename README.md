# Final Year Project
This is a project to demonstrate a comparison of various machine learning algorithms in the realm of optimising prices for hotels to find the best prices to maximise revenue.

## Project Directory Structure
The project main directory contains a super class base_model.py along with the machines learning models classes that inherit it. Additionally, you will also find other relevant classes including the main file that runs the app Home.py. Also, you can find multiple folders containing relevant files such as datasets (contains the used datasets and the code used to generate them), pages (contains the pages in the web app with the home page being the main file Home.py in the main directory) and tests (contains the test files). The reasoning for the names of the pages files (Home.py, 1_Nearest_Neighbours.py etc.) is due to the conventions of the streamlit library to detect and order the pages in the web app.

## Running the app
The project can be run by running the command "**streamlit run Home.py**" in the terminal. 

Here is a list of the main packages used by the app:
* streamlit
* scikit-learn
* numpy
* matplotlib
* pandas

These packages can be downloaded using the command "**pip install [package_name]**". Downloading these should make the app runnable. In the case the app is still not running, refer to requirements.txt for a list of all the used packages, dependancies and versions.

The test files can be run using the command "**python -m tests.[test_file]**" where [test_file] represents the file to be tested. There are 3 test files, one for each algorithm:
* test_knn
* test_lr
* test_dt

For example to run the tests for 'test_knn.py' use the command "**python -m tests.test_knn**"

## Video Links
**Running the web app:** https://youtu.be/yd0p0_lZBeg

**Demonstrating the features:** https://youtu.be/uFQyXQcjC7U 

**Running the test files:** https://youtu.be/qZ9Y_omEJ-I
