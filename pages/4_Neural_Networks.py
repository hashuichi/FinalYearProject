import streamlit as st
import pandas as pd

class NeuralNetworksPage:
    def __init__(self):
        st.set_page_config(page_title="Neural Networks", layout="wide")

    def run(self):
        st.title("Neural Networks")


if __name__ == '__main__':
    app = NeuralNetworksPage()
    app.run()