import streamlit as st
from Data import DataPage
from data_loader import DataLoader
from decision_tree import DecisionTree

class DecisionTreePage:
    """
    A class that encapsulates the functionality for displaying a page dedicated to 
    Decision Tree Regression using Streamlit. It allows users to upload data, train 
    a decision tree model, predict prices for new hotel entries, and view performance metrics.
    """
    def run(self):
        """
        Main execution function for the Decision Tree page. It handles data loading, 
        training of the Decision Tree model, prediction of new entries, and 
        visualisation of the model's performance.
        """
        st.title("Decision Tree")
        dl = DataLoader()
        selected_df = DataPage().get_data_selection()
        dl.load_data(selected_df)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        tree = DecisionTree(selected_df, self.X_train, self.X_test, self.y_train, self.y_test, 10)
        model = tree.fit(self.X_train.values, self.y_train.values)

        st.subheader('Predict Hotel Price')
        new_entry = tree.get_new_hotel_fields(st)
        new_price = tree.predict(new_entry)
        st.write(f'Predicted Price Per Night: £{new_price:.2f}')

        st.subheader('Results')
        y_pred = tree.get_y_pred(self.X_test.values)
        rmse = tree.calculate_rmse(self.y_test, y_pred)
        st.write(f'**RMSE:** {rmse:.2f}')

        st.subheader('Performance Graphs')
        tree.generate_plots(st, y_pred)

if __name__ == '__main__':
    app = DecisionTreePage()
    app.run()