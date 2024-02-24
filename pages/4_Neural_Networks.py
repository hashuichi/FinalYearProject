import streamlit as st
from Data import DataPage
from data_loader import DataLoader
from neural_networks import NeuralNetworks

class NeuralNetworksPage:
    def run(self):
        st.title("Neural Networks")
        dl = DataLoader()
        selected_df = DataPage().get_data_selection()
        dl.load_data(selected_df)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()

        model = NeuralNetworks(selected_df, self.X_train, self.X_test, self.y_train, self.y_test)
        model.train_model(10)

        st.subheader('Optimise Hotel Price')
        new_entry = model.get_new_hotel_fields(st)
        new_price = model.predict_price(new_entry)
        st.write(new_price)
        st.write(f'Predicted Price Per Night: £{float(new_price[0]):.2f}')
        
        st.subheader('Results')
        rmse = model.calculate_rmse_value()
        st.write(f'**RMSE:** {rmse:.2f}')

        st.subheader('Performance Graphs')
        model.generate_plots(st)


if __name__ == '__main__':
    app = NeuralNetworksPage()
    app.run()