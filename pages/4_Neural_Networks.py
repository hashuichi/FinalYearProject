import streamlit as st
from data_loader import DataLoader
from neural_networks import NeuralNetworks

class NeuralNetworksPage:
    """
    A class that encapsulates the functionality for displaying a page dedicated to 
    Neural Networks using Streamlit. It allows users to load data, train feedforward 
    and recurrent neural network models, predict prices for new hotel entries using 
    both models, display performance metrics, and visualise performance graphs.
    """
    def run(self):
        """
        Main execution function for the Neural Networks page. It handles data loading, 
        training of feedforward and recurrent neural network models, prediction of prices 
        for new hotel entries using both models, display of performance metrics, and 
        visualisation of performance graphs.
        """
        st.title("Neural Networks")
        dl = DataLoader()
        selected_df = "Benchmark Dataset"
        dl.load_data(selected_df)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()

        neural_net = NeuralNetworks(selected_df, self.X_train, self.X_test, self.y_train, self.y_test, 5)
        feedforward_model = neural_net.train_feedforward()
        recurrent_model = neural_net.train_recurrent()
        y_pred_feedforward = neural_net.get_y_pred_feedforward(feedforward_model)
        st.subheader('Optimise Hotel Price')
        new_entry = neural_net.get_new_hotel_fields(st)
        new_price_feedforward = neural_net.predict_feedforward(feedforward_model, new_entry)
        st.write(f'Best Selling Price Per Night (Feedforward Networks): £{new_price_feedforward:.2f}')
        new_price_recurrent = neural_net.predict_recurrent(recurrent_model, new_entry)
        st.write(f'Best Selling Price Per Night (Recurrent Networks): £{new_price_recurrent:.2f}')
        
        st.subheader('Results')
        rmse_feedforward = neural_net.calculate_rmse(y_pred_feedforward)
        st.write(f'**RMSE with feedforward networks:** {rmse_feedforward:.2f}')
        rmse_recurrent = neural_net.evaluate_model(recurrent_model)
        st.write(f'**RMSE with recurrent networks:** {rmse_recurrent:.2f}')

        st.subheader('Performance Graphs')
        neural_net.generate_plots(st, y_pred_feedforward)


if __name__ == '__main__':
    app = NeuralNetworksPage()
    app.run()