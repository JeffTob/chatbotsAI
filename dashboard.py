import streamlit as st
import pandas as pd
from data_analyzer import DataAnalyzer
import matplotlib.pyplot as plt

class Dashboard:
    def __init__(self, data_analyzer: DataAnalyzer):
        self.data_analyzer = data_analyzer

    def generate(self):
        st.subheader("Data Summary")
        summary = self.data_analyzer.get_summary()        
        st.write(summary)

        #st.sidebar.subheader("Column Names")
        #column_names = self.data_analyzer.get_column_names()
        #st.sidebar.write(column_names)

        st.subheader("Data Visualization")
        st.write(self.data_analyzer.df)

        #st.sidebar.subheader("Column Names")
        column_names = self.data_analyzer.get_column_names()
        #st.sidebar.write(column_names)

        # Dropdown for column selection
        selected_column = st.selectbox('Select a column for visualization', column_names)        

        st.subheader("Chart Visualization")
        st.bar_chart(self.data_analyzer.df[selected_column])

        st.subheader("Histogram")
        plt.hist(self.data_analyzer.df[selected_column], bins=20)
        st.pyplot(plt)