import streamlit as st

class DecisionTree:
    def __init__(self):
        st.set_page_config(page_title="Decision Tree", layout="wide")

    def run(self):
        st.title("Decision Tree")
        st.write("Decision Tree implementation will be displayed here...")

if __name__ == '__main__':
    app = DecisionTree()
    app.run()