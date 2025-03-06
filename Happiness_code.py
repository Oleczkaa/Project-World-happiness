import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_all = pd.read_csv("df_all.csv")
df_clean = pd.read_csv("df_clean.csv")

st.title("Project: World Happiness")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)