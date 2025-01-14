import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from data_loader import DataLoader

class DataPage:
    """
    A class to encapsulate the functionality for displaying and interacting with datasets
    within a Streamlit application. This class allows users to select, view, and explore
    different aspects of datasets through interactive visualisations and tables.
    """

    def __init__(self):
        """
        Initialises the DataPage object with default attributes.
        """
        self.data = None
        self.room_type_map = {
            1: 'Private room',
            2: 'Entire home/apt',
            3: 'Shared room',
            4: 'Hotel room'
        }

    def run(self):
        """
        Configures the Streamlit page settings and orchestrates the rendering of the page
        content based on user interactions and dataset selection.
        """
        st.set_page_config(page_title="Data", layout="wide")
        st.title("Data")
        self.load_data()
        col1, col2 = st.columns([2, 1])
        if st.session_state.selected_df == "Benchmark Dataset":
            with col1:
                self.generate_plots()
            col2.write(self.rename_benchmark_data())
        else:
            col1.pyplot(self.display_price_distance_chart())
            col1.pyplot(self.display_price_star_rating_chart())
            col2.table(self.rename_fake_data())

    def load_data(self):
        """
        Loads data based on the user's selection from a dropdown menu.
        """
        data_option = self.get_data_selection()
        if st.session_state.selected_df == data_option:
            dl = DataLoader()
            dl.load_data(data_option)
            self.data = dl.get_data()
            st.session_state.df = self.data

    def get_data_selection(self):
        """
        Displays a dropdown for dataset selection and returns the user's choice.

        Returns:
            str: The selected dataset name.
        """
        return st.selectbox("**Select Dataset**", ["Choose a Dataset", "Benchmark Dataset", "Structured Dataset", "Unstructured Dataset"], key='selected_df')

    def rename_benchmark_data(self):
        """
        Renames columns of the benchmark dataset for readability and maps numeric room types to their string representations.

        Returns:
            pandas.DataFrame: The renamed dataset if available; otherwise, an empty DataFrame.
        """
        if self.data is not None:
            new_column_names = {
                'room_type': 'Room Type',
                'accommodates': 'Occupants',
                'bathrooms': 'Bathrooms',
                'beds': 'Beds',
                'price': 'Price ($)'
            }
            renamed_data = self.data.rename(columns=new_column_names)
            renamed_data.iloc[:, 0] = renamed_data.iloc[:, 0].map(self.room_type_map)
            return renamed_data
        else:
            st.warning("No data selected.")
            return pd.DataFrame()
        
    def rename_fake_data(self):
        """
        Renames columns of the fake dataset for readability.

        Returns:
            pandas.DataFrame: The renamed dataset if available; otherwise, an empty DataFrame.
        """
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
        """
        Creates and displays a scatter plot of price against distance with coloring based on star rating.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot if data is available; otherwise, an empty figure.
        """
        if self.data is not None:
            if st.session_state.selected_df == "Benchmark Dataset":
                fig, ax = plt.subplots()
                scatter = ax.scatter(self.data['distance'], self.data['price'], c=self.data['star_rating'], cmap='viridis')
                ax.set_xlabel('Distance')
                ax.set_ylabel('Price')
                ax.legend(*scatter.legend_elements(), title='Star Rating')
                return fig
            else:
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
        """
        Generates a scatter plot showing the relationship between hotel star rating and price.

        Returns:
            matplotlib.figure.Figure: The figure containing the plot if data is available; otherwise, an empty figure.
        """
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
        
    def generate_plots(self):
        """
        Generates and displays histograms, scatter plots, and box plots for various features of the dataset,
        offering insights into the distribution, relationships, and variability of data points.
        """
        col1, col2 = st.columns(2)
        
        # Histograms
        room_type_counts = self.data['room_type'].map(self.room_type_map).value_counts()
        fig, ax = plt.subplots()
        room_type_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Room Type')
        ax.set_ylabel('Frequency')
        ax.set_xticklabels(room_type_counts.index, rotation=45)
        col1.pyplot(fig)
        for col in self.data.select_dtypes(include='number').columns.drop(['room_type', 'price']):
            fig, ax = plt.subplots()
            ax.hist(self.data[col])
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            col1.pyplot(fig)

        # Scatter plot for price vs accommodates
        fig, ax = plt.subplots()
        ax.scatter(self.data['accommodates'], self.data['price'])
        ax.set_xlabel('Accommodates')
        ax.set_ylabel('Price')
        ax.grid(True)
        col2.pyplot(fig)

        # Scatter plot for price vs beds
        fig, ax = plt.subplots()
        ax.scatter(self.data['beds'], self.data['price'])
        ax.set_xlabel('Beds')
        ax.set_ylabel('Price')
        ax.grid(True)
        col2.pyplot(fig)

        # Box plot for price based on room type
        fig, ax = plt.subplots(figsize=(8, 6))
        self.data.boxplot(column='price', by='room_type', ax=ax)
        ax.set_ylabel('Price')
        ax.set_xlabel('Room Type')
        ax.set_xticklabels(['Private room', 'Entire home/apt', 'Shared room', 'Hotel room'])
        ax.grid(True)
        col2.pyplot(fig)

if __name__ == '__main__':
    app = DataPage()
    app.run()