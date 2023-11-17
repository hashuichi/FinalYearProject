from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import process_data

def main():
    st.set_page_config(page_title="Financial Agent", layout="wide")
    st.title("Data")

    data = pd.read_csv('fake_structured_data.csv')

    col1, col2 = st.columns([1.5, 2])
    col1.table(renameDataColumns(data))
    col2.pyplot(display_price_distance_chart(data))
    col2.pyplot(display_price_star_rating_chart(data))

def renameDataColumns(data):
    new_column_names = {
        'star_rating': 'Star Rating (1-5)',
        'distance': 'Distance to city centre (Metres)',
        'price': 'Price (£)'
    }
    return data.rename(columns=new_column_names)
    
def display_price_distance_chart(data):
    fig, ax = plt.subplots()
    # scatter = ax.scatter(data['Distance'], data['Price'], c=data['Hotel Star Rating'], cmap='viridis')
    scatter = ax.scatter(data['distance'], data['price'], c=data['star_rating'], cmap='viridis')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Price')
    ax.legend(*scatter.legend_elements(), title='Star Rating')
    return fig

def display_price_star_rating_chart(data):
    fig, ax = plt.subplots()
    # ax.scatter(data['Hotel Star Rating'], data['Price'])
    ax.scatter(data['star_rating'], data['price'])
    ax.set_xlabel('Hotel Star Rating')
    ax.set_ylabel('Price')
    ax.set_xticks(np.arange(data['star_rating'].min(), data['star_rating'].max() + 1, 1))
    return fig

if __name__ == '__main__':
    main()