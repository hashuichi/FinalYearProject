# Final Year Project

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