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

import matplotlib.pyplot as plt
import plotly.express as px

continent_colors = {
    'Africa': '#FF7550',
    'Asia': '#FFD000',
    'Europe': '#31BBDB',
    'North America': '#117C7C',
    'South America': '#FF486B',
    'Oceania': '#00DF9D',
    'Antarctica': '#7EE2E1'
}

region_colors = {
    #Europe
    'Western Europe': "#002D74",
    'Northern Europe':'#007EA6',
    'Southern Europe':'#31BBDB',
    'Eastern Europe':'#B0E5F1',
    #Africa
    'Northern Africa':'#3EAA94',
    'Southern Africa':'#A2D1B1',
    'Eastern Africa': '#FF7550',
    'Western Africa':'#C62E14',
    'Middle Africa':'#660015',
    #Asia
    'Central Asia': '#864401',
    'Eastern Asia': '#E1A100',
    'Western Asia': '#FFD000',
    'South Eastern Asia': '#B8A0D2',
    'Southern Asia': '#6B3AA2',
    #America
    'South America': '#117C7C',
    'Central America': '#6B3A74',
    'Northern America': '#FF486B',
    #Other Regions
    'Australia and New Zealand':'#00DF9D',
    'Caribbean':'#FF7BAC'
}

if page == pages[0] : 
  st.write("### Presentation of data")

  st.dataframe(df_all.head(10))
  st.write(df_all.shape)
  st.dataframe(df_all.describe())
  if st.checkbox("Show NA") :
    st.dataframe(df_all.isna().sum())

if page == pages[1] : 
  st.markdown("<h3 style='color: #6E66CC;'>Data Vizualization</h3>", unsafe_allow_html=True)

  st.write("### Data Vizualization")

  #BOX PLOT FOR MULTIPLE NUMERICAL VARIABLES

  numerical_columns = [
    "Life Ladder", "Log GDP per capita", "Social support",
    "Healthy life expectancy at birth", "Freedom to make life choices",
    "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"]

  plt.figure(figsize=(12, 6))
  sns.boxplot(data=df_all[numerical_columns], palette="Set2")
  plt.title("Box Plot for Multiple Numerical Variables")
  plt.xlabel("Variables")
  plt.ylabel("Values")
  plt.xticks(rotation=45)
  st.pyplot(plt)


  # Add a title to the page
  st.title("Comparison of Indicators from 2023")

  df_2023 = df_all[df_all['year'] == 2023]

  # Create a layout with three columns
  col1, col2, col3 = st.columns(3)

  # Create a boxplot for each column
  with col1:
    st.write("Life Ladder & Log GDP per capita")
    box1 = plt.boxplot(
        [df_2023['Life Ladder'], df_2023['Log GDP per capita'].dropna()],
        patch_artist=True,
        widths=0.3
    )
    plt.title("Life Ladder & Log GDP per capita")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

  with col2:
    st.write("Healthy life expectancy at birth")
    box2 = plt.boxplot(
        [df_2023['Healthy life expectancy at birth'].dropna()],
        patch_artist=True,
        widths=0.15
    )
    plt.title("Healthy life expectancy at birth")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

  with col3:
    st.write("Other Metrics")
    box3 = plt.boxplot(
        [df_2023[col].dropna() for col in [
            'Social support', 'Freedom to make life choices', 'Generosity',
            'Perceptions of corruption', 'Positive affect', 'Negative affect'
        ]],
        patch_artist=True,
        widths=0.8
    )
    plt.title("Other Metrics")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

  # Add a title to the page
  st.title("Distribution of Indices for Selected Years")
  
  # Define the styling function
  def style_boxplot(box, facecolor, edgecolor, linewidth, flier_color, median_color, whisker_color):
    for patch in box['boxes']:
        patch.set_facecolor(facecolor)
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(linewidth)
    for flier in box['fliers']:
        flier.set(marker='o', markerfacecolor=flier_color, alpha=0.5)
    for median in box['medians']:
        median.set(color=median_color, linewidth=2)
    for whisker in box['whiskers']:
        whisker.set(color=whisker_color, linewidth=1, linestyle='--')

  # Apply styling to each boxplot
  style_boxplot(box1, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  style_boxplot(box2, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  style_boxplot(box3, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  
  #Distribution of indices next to eachother for 2013, 2018, 2021, 2023 (to compare pro and after covid)

  # List of indices to analyze
  indices = ['Life Ladder', 'Log GDP per capita', 'Social support',
            'Healthy life expectancy at birth', 'Freedom to make life choices',
            'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect']

  # Specify the years we would like to to analyze
  years_to_plot = [2013, 2018, 2021,2023]

  # Create a figure for the histograms
  fig, axes = plt.subplots(nrows=len(indices), ncols=len(years_to_plot), figsize=(20, 25))  # Rows for indices, columns for years
  fig.suptitle("Distribution of Indices for Selected Years", fontsize=20)

  # Loop over each variable and year
  for i, index in enumerate(indices):
      for j, year in enumerate(years_to_plot):
          ax = axes[i, j]

          # Filter data for the current year
          yearly_data = df_all[df_all['year'] == year]

          # Plot the histogram
          ax.hist(yearly_data[index].dropna(), bins=20, color='#6E66CC', edgecolor='black')
          ax.set_title(f"{index} ({year})")
          ax.set_xlabel(index)
          ax.set_ylabel("Frequency")

  plt.tight_layout(rect=[0, 0, 1, 0.95])
  st.pyplot(fig)



    
    






