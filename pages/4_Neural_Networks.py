import streamlit as st
from Data import DataPage
from data_loader import DataLoader
from neural_networks import NeuralNetworks

class NeuralNetworksPage:
    def run(self):
        st.title("Neural Networks")
        dl = DataLoader()
        selected_df = "Benchmark Dataset"
        dl.load_data(selected_df)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()

        model = NeuralNetworks(selected_df, self.X_train, self.X_test, self.y_train, self.y_test, 3)
        neural_net = model.train_model()
        y_pred = model.calculate_y_pred(neural_net)
        st.subheader('Optimise Hotel Price')
        new_entry = model.get_new_hotel_fields(st)
        new_price = model.predict(neural_net, new_entry)
        st.write(f'Predicted Price Per Night: £{new_price:.2f}')
        
        st.subheader('Results')
        rmse = model.get_rmse(y_pred)
        st.write(f'**RMSE:** {rmse:.2f}')

        st.subheader('Performance Graphs')
        model.generate_plots(st, y_pred)


if __name__ == '__main__':
    app = NeuralNetworksPage()
    app.run()