import streamlit as st
import pandas as pd

class NeuralNetworksPage:
    def __init__(self):
        st.set_page_config(page_title="Neural Networks", layout="wide")

    def run(self):
        st.title("Neural Networks")

        df = pd.read_csv('datasets/airbnb_london.csv')
        # df['property_type'] = df['property_type'].apply(lambda x: 'Private Room' if 'private room' in x.lower() else x)
        # df['property_type'] = df['property_type'].apply(lambda x: 'Shared Room' if 'shared room' in x.lower() else x)
        # df['property_type'] = df['property_type'].apply(lambda x: x.replace('Entire', '').strip().capitalize() if 'Entire' in x else x)
        property_types = df['room_type'].unique()
        selected_property_type = st.selectbox('Select Room Type', property_types)
        # Filter DataFrame based on selected property type
        filtered_data = df[df['room_type'] == selected_property_type]
        filtered_data_reset_index = filtered_data.reset_index(drop=True)
        st.dataframe(filtered_data_reset_index)


if __name__ == '__main__':
    app = NeuralNetworksPage()
    app.run()