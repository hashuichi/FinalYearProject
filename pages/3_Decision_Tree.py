import streamlit as st
from Data import DataPage
from data_loader import DataLoader
from decision_tree import DecisionTree

class DecisionTreePage:
    def run(self):
        st.title("Decision Tree")
        dl = DataLoader()
        selected_df = DataPage().get_data_selection()
        dl.load_data(selected_df)
        self.X_train, self.X_test, self.y_train, self.y_test = dl.split_data()
        tree = DecisionTree(selected_df, self.X_train, self.X_test, self.y_train, self.y_test)
        # tree.train_model()
        tree.fit()

        st.subheader('Predict Hotel Price')
        new_entry = tree.get_new_hotel_fields(st)
        # new_price = tree.predict_price(new_entry)
        new_price = tree.predict(new_entry)
        st.write(f'Predicted Price Per Night: £{new_price[0]:.2f}')

        st.subheader('Results')
        rmse = tree.calculate_rmse_value()
        st.write(f'**RMSE:** {rmse:.2f}')

        st.subheader('Performance Graphs')
        tree.generate_plots(st)

if __name__ == '__main__':
    app = DecisionTreePage()
    app.run()