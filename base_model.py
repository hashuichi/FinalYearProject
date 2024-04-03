import matplotlib.pyplot as plt

class BaseModel:
    def __init__(self, selected_df, X_train, X_test, y_train, y_test):
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
        Creates inputs for hotel data and returns the values
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
        
    def generate_plots(self, st, optional_y_pred=None):
        '''
        Displays the different plots to visualise the performance of the model.
        '''
        col1, col2 = st.columns(2)
        if optional_y_pred is not None:
            self.y_pred = optional_y_pred

        col2.pyplot(self.plot_predicted_actual())
        col1.pyplot(self.plot_residuals_distribution())
        col1.pyplot(self.plot_residual_plot())
        
    def plot_predicted_actual(self):
        """
        Plots the scatter plot of predicted vs actual prices

        Returns:
        fig (Figure): The figure containing the plot
        """
        fig, ax = plt.subplots()
        ax.scatter(self.y_test, self.y_pred, c='blue', label='True Prices')
        ax.set_title('Predicted Prices vs Actual Prices')
        ax.set_xlabel('Actual Prices')
        ax.set_ylabel('Predicted Prices')
        return fig
    
    def plot_residual_plot(self):
        """
        Plots the residual of the actual prices - predicted prices

        Returns:
        fig (Figure): The figure containing the plot
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
        Plots a histogram of the distribution of residuals

        Returns:
        fig (Figure): The figure containing the plot
        """
        fig, ax = plt.subplots()
        ax.hist((self.y_test - self.y_pred), bins=30, color='blue', edgecolor='black')
        ax.set_title('Distribution of Residuals')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        return fig
