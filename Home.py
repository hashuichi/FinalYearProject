from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd

def main():
    st.set_page_config(page_title="Financial Agent", layout="centered")
    data = pd.read_csv('fake_dataset.csv')
    st.title("Data Page")
    st.write("Displaying data and quality charts here...")
    st.table(data)

def set_selected_page(selected_page):
    st.session_state.selected_page = selected_page
    # Remove the 'highlighted' class from all buttons
    st.markdown('<style>button{}</style>', unsafe_allow_html=True)
    # Add the 'highlighted' class to the selected button
    st.markdown(f'<style>#{selected_page.lower().replace(" ", "_")}_button{{background-color: #4CAF50; color: white;}}</style>', unsafe_allow_html=True)    

if __name__ == '__main__':
    main()