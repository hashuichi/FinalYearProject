import streamlit as st
from Data import DataPage
from data_loader import DataLoader
from decision_tree import DecisionTree

class DecisionTreePage:
    def run(self):
        st.title("Decision Tree")
        dl = DataLoader()
        dl.load_data(DataPage().load_data())
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        tree = DecisionTree(self.X_train, self.X_test, self.y_train, self.y_test)
        tree.train_model()

        st.subheader('Predict Hotel Price')
        star_rating, distance = tree.get_new_hotel_fields(st)
        new_price = tree.predict_price(star_rating, distance)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')

        st.subheader('Results')
        rmse = tree.calculate_rmse_value()
        st.write(f'**RMSE:** {rmse:.2f}')

        st.subheader('Performance Graphs')
        tree.generate_plots(st)

if __name__ == '__main__':
    app = DecisionTreePage()
    app.run()