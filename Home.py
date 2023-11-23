import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataLoader

class DataPage:
    def __init__(self):
        self.data = None

    def run(self):
        st.set_page_config(page_title="Data", layout="wide")
        dl = DataLoader()
        dl.load_data(self.get_data_selection())
        st.title("Data")
        self.data = dl.get_data()
        col1, col2 = st.columns([2, 1.5])
        col1.pyplot(self.display_price_distance_chart())
        col1.pyplot(self.display_price_star_rating_chart())
        col2.table(self.rename_data_columns())

    def get_data_selection(self):
        data_option = st.selectbox("**Select Dataset**", ["Structured Dataset", "Unstructured Dataset"])
        if data_option == 'Structured Dataset':
            return 'datasets/fake_structured_data.csv'
        elif data_option == 'Unstructured Dataset':
            return 'datasets/fake_data.csv'

    def rename_data_columns(self):
        if self.data is not None:
            new_column_names = {
                'star_rating': 'Star Rating (1-5)',
                'distance': 'Distance to city centre (Metres)',
                'price': 'Price (£)'
            }
            return self.data.rename(columns=new_column_names)
        else:
            st.warning("No data selected.")
            return pd.DataFrame()

    def display_price_distance_chart(self):
        if self.data is not None:
            fig, ax = plt.subplots()
            scatter = ax.scatter(self.data['distance'], self.data['price'], c=self.data['star_rating'], cmap='viridis')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Price')
            ax.legend(*scatter.legend_elements(), title='Star Rating')
            return fig
        else:
            st.warning("No data selected.")
            return plt.figure()

    def display_price_star_rating_chart(self):
        if self.data is not None:
            fig, ax = plt.subplots()
            ax.scatter(self.data['star_rating'], self.data['price'])
            ax.set_xlabel('Hotel Star Rating')
            ax.set_ylabel('Price')
            ax.set_xticks(np.arange(self.data['star_rating'].min(), self.data['star_rating'].max() + 1, 1))
            return fig
        else:
            st.warning("No data selected.")
            return plt.figure()

if __name__ == '__main__':
    app = DataPage()
    app.run()