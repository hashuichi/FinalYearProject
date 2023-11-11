import streamlit as st
from pages import nearest_neighbours

def display_new_hotel_fields():
    st.subheader('Predict Hotel Price')
    col1, col2, col3 = st.columns([3,1,1])
    new_hotel_name = col1.text_input('Hotel Name', 'Hazel Inn')
    star_rating = col2.number_input('Star Rating', 1, 5, 2)
    distance = col3.number_input('Distance', 100, 5000, 100)

    return new_hotel_name, star_rating, distance

def display_prediction(result):
    st.write(f'Predicted Price Per Night: £{result[0]:.2f}')

def display_mse_chart(num_neighbours, mse_values):
    st.subheader('Mean Squared Error for n_neighbors 1 to 25')
    st.line_chart(dict(zip(num_neighbours, mse_values)))

def display_best_k_and_mse(best_k, best_mse):
    st.write(f'Best k: {best_k}')
    st.write(f'Best Mean Squared Error: {best_mse:.2f}')