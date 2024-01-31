import streamlit as st
import pandas as pd

class NeuralNetworksPage:
    def __init__(self):
        st.set_page_config(page_title="Neural Networks", layout="wide")

    def run(self):
        st.title("Neural Networks")

        df = pd.read_csv('datasets/airbnb_london.csv')

        selected_room_type = st.selectbox('Select Room Type', df['room_type'].unique())

        # Filter DataFrame based on selected property type
        filtered_data = df[df['room_type'] == selected_room_type]
        filtered_data = filtered_data.reset_index(drop=True)
        st.dataframe(filtered_data)


if __name__ == '__main__':
    app = NeuralNetworksPage()
    app.run()