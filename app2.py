import streamlit as st
import pandas as pd


def app(train, test):
    st.dataframe(train)
