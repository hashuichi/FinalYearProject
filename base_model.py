import matplotlib.pyplot as plt

class BaseModel:
    """
    A base class for building and evaluating predictive models forS pricing.

    This class provides a structured framework for initialising models with training and testing data,
    transforming input data for predictions, and generating visualisations for model evaluation. It
    supports customisation for different types of accommodations or hotels by adapting to various input
    features and configurations.
    """

    def __init__(self, selected_df, X_train, X_test, y_train, y_test):
        """
        Initialises the BaseModel with training and testing data, along with dataset specifics.

        Parameters:
            selected_df (string): The dataset selected for the model.
            X_train (pd.DataFrame): Feature matrix of the training data.
            X_test (pd.DataFrame): Feature matrix of the test data.
            y_train (pd.Series): Target values of the training data.
            y_test (pd.Series): Target values of the test data.
        """
        self.selected_df = selected_df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.y_pred = None
        self.room_type_map = {
            'Private room': 1,
            'Entire home/apt': 2,
            'Shared room': 3,
            'Hotel room': 4
        }
    
    def get_new_hotel_fields(self, st):
        """
        Dynamically generates input fields in the Streamlit UI for new hotel entries.

        Depending on the dataset selected, this method presents different input fields in the UI for
        capturing the details of a new hotel entry, facilitating prediction of its price.

        Parameters:
            st (Streamlit): The Streamlit module for generating UI components.

        Returns:
            list: A list of input values collected from the user through the UI.
        """
        if self.selected_df == "Benchmark Dataset":
            col1, col2, col3, col4 = st.columns(4)
            room_type = col1.selectbox('Room Type', self.room_type_map.keys())
            occupancy = col2.number_input('Number of Occupants', 1, 16, 2)
            bathrooms = col3.selectbox('Number of Bathrooms', sorted(self.X_train['bathrooms'].unique()), 1)
            beds = col4.number_input('Number of Beds', 1, 50, 1)
            return [self.room_type_map[room_type], occupancy, bathrooms, beds]
        else:
            col1, col2, col3 = st.columns([3, 1, 1])
            col1.text_input('Hotel Name', 'Hazel Inn')
            star_rating = col2.number_input('Star Rating', 1, 5, 2)
            distance = col3.number_input('Distance', 100, 5000, 100)
            return [star_rating, distance]
        
    def generate_plots(self, st, y_pred=None):
        """
        Generates and displays plots for model evaluation in the Streamlit UI.

        This method creates plots for comparing predicted and actual prices, and for visualising
        the distribution and residuals of the predictions.

        Parameters:
            st (Streamlit): The Streamlit module for displaying plots in the UI.
            y_pred (list, optional): A list of predicted values. If not provided, uses `self.y_pred`.
        """
        col1, col2 = st.columns(2)
        if y_pred is not None:
            self.y_pred = y_pred

        col2.pyplot(self.plot_predicted_actual())
        col1.pyplot(self.plot_residuals_distribution())
        col1.pyplot(self.plot_residual_plot())
        
    def plot_predicted_actual(self):
        """
        Creates a scatter plot comparing the actual prices with the predicted prices.

        This visualisation helps in assessing the accuracy of the model by showing how closely
        the predicted prices align with the actual prices. A closer alignment indicates a higher
        accuracy of the model.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure object containing the scatter plot.
        """
        fig, ax = plt.subplots()
        ax.scatter(self.y_test, self.y_pred, c='blue', label='True Prices')
        ax.set_title('Predicted Prices vs Actual Prices')
        ax.set_xlabel('Actual Prices')
        ax.set_ylabel('Predicted Prices')
        return fig
    
    def plot_residual_plot(self):
        """
        Generates a residual plot to visualise the difference between actual and predicted prices.

        The residuals (actual - predicted prices) are plotted against a relevant feature to
        identify patterns or biases in the predictions. The color coding can represent different
        categories such as room type or star rating, providing insights into how these factors
        might influence prediction errors.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure object containing the residual plot.
        """
        if self.selected_df == "Benchmark Dataset":
            X_line = self.X_test['accommodates']
            X_label = 'Number of Occupants'
            colours = self.X_test['room_type']
            legend_title = 'Room Type'
        else:
            X_line = self.X_test['distance']
            X_label = 'Distance'
            colours = self.X_test['star_rating']
            legend_title = 'Star Rating'

        fig, ax = plt.subplots()
        scatter = ax.scatter(X_line, (self.y_test - self.y_pred), c=colours, cmap='viridis', label='Residual Plot')
        ax.set_title('Residual Plot')
        ax.set_xlabel(X_label)
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.legend(*scatter.legend_elements(), title=legend_title)
        return fig
    
    def plot_residuals_distribution(self):
        """
        Creates a histogram to show the distribution of the residuals (actual - predicted prices).

        This plot helps in understanding the variance in the prediction errors, indicating whether
        the model tends to underpredict or overpredict prices. Ideally, the distribution should be
        centered around zero, indicating accurate predictions without systematic bias.

        Returns:
            matplotlib.figure.Figure: A matplotlib figure object containing the histogram of residuals.
        """
        fig, ax = plt.subplots()
        ax.hist((self.y_test - self.y_pred), bins=30, color='blue', edgecolor='black')
        ax.set_title('Distribution of Residuals')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        return fig
