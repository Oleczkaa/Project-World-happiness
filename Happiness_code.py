import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df_all = pd.read_csv("df_all.csv")
df_clean = pd.read_csv("df_clean.csv")

st.title("Project: World Happiness")
st.sidebar.title("Table of contents")
pages=["Project Description","Presentation of Data", "Data Vizualization", "Modelling"]
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
  st.write("") 
  st.write("") 
  #st.markdown("<h3 style='color: #6E66CC;'>Project Description</h3>", unsafe_allow_html=True)
  import streamlit as st

  st.markdown("""
  ## 🤔 A Single Factor or the Interplay of Many?

  How happy is our society – and is happiness measurable?  
  Various indicators help describe both the **subjective** and **objective** well-being of individuals and societies.  

  But which country is **truly the happiest**?  
  Can we identify differences based on **prosperity, social environment, or political conditions**?  

  Is happiness shaped by **individual factors alone**, or is it the result of a complex **interplay of multiple influences**?
  """)

  st.markdown(" ")  # This adds an empty line as a break.
  st.markdown(" ")  # This adds an empty line as a break.
  
  st.markdown("""
              ### 📌 Project Description  
              This project analyzes global happiness trends based on data from the **World Happiness Report**. 
              It explores factors like:  
              - 🌍 **GDP per capita**  
              - 👨‍👩‍👧‍👦 **Social Support**  
              - ❤️ **Life Expectancy**  
              - 🗽 **Freedom to Make Choices**  
              - 💖 **Generosity**  
              - 🚨 **Perception of Corruption**  
              - 😊 **Positive Affect**  
              - 😞 **Negative Affect**
              """)
  
         
  st.write("") 
  st.write("")
  st.write("") 
 
  st.markdown("""          
              
              📊 We explored the data across various dimensions, such as:
              - 🌐 **By Continent**
              - 🗺️ **By Region**
              - 🌎 **By Country**
              - 📅 **By Year**
              """)
  st.write("") 
  st.markdown("The **goal** is to identify patterns that influence happiness across different countries and continents.")
  
  st.write("") 
  st.write("")
  st.write("")

if page == pages[1] : 
  st.write("")
  st.markdown("Sample of the official **World Happiness Report**")

  st.dataframe(df_all.head(10))
  st.write(df_all.shape)

  st.markdown("Summary Statiscis")
  st.dataframe(df_all.describe())
  if st.checkbox("Show NA") :
    st.dataframe(df_all.isna().sum())

if page == pages[2] : 

  #BOX PLOT FOR MULTIPLE NUMERICAL VARIABLES
  
  st.markdown("""
  ### Data Exploration

  **How large is the range of each value? What are the mean values, and are there any outliers?**

  To start, we will examine each value using a **boxplot** to identify:
  - where the **mean** lies,  
  - the **minimum and maximum** values, and  
  - whether there are any **outliers**.

  This approach will be applied to **all available indicators**.
  """)
  
  numerical_columns = [
    "Life Ladder", "Log GDP per capita", "Social support",
    "Healthy life expectancy at birth", "Freedom to make life choices",
    "Generosity", "Perceptions of corruption", "Positive affect", "Negative affect"]

  plt.figure(figsize=(14, 8))
  sns.boxplot(data=df_all[numerical_columns], palette="Set2")
  plt.title("Box Plot for Multiple Numerical Variables")
  plt.xlabel("Variables")
  plt.ylabel("Values")
  plt.xticks(rotation=45)
  st.pyplot(plt)

  st.markdown("""
    ### Initial Observations

    In our first plot, we displayed **all indicators**, including data for **all years** and **all countries**.

    One thing immediately stands out:  
    the **scales of the indicators are not consistent**, making only *"Healthy life expectancy at birth"* truly legible.  
    This is because its scale is presented in **years**, whereas the other indicators use a scale ranging from **0 to 1**.

    Despite the plot being difficult to read, one **notable observation** emerges:  
    a life expectancy of **less than ten years** seems extremely low.

    We need to **investigate this further** to rule out any errors in the dataset,  
    as such discrepancies could undermine the **credibility of our analysis**.
    """)
  st.write("")
  st.write("")

  st.markdown("""
  ### Visualizing Indicator Scales

  First, we will create a **boxplot** to visualize the **different scales** and clearly compare all the indicators.

  Below is a comparison of indicators from **2005 to 2023**:
  """)
 
  st.write("")
  st.write("")

  fig2, ax2 = plt.subplots(1,3, figsize=(12, 6))

  box4 = ax2[0].boxplot(
        [df_all['Life Ladder'], df_all['Log GDP per capita'].dropna()],
        patch_artist=True,
        widths = 0.3)

    #show Boxplots with nearly same scale
  box5 = ax2[1].boxplot(
        [df_all['Healthy life expectancy at birth'].dropna()],
        patch_artist=True,
        widths = 0.15)

    #show Boxplots with nearly same scale
  box6 = ax2[2].boxplot(
        [df_all['Social support'].dropna(),
        df_all['Freedom to make life choices'].dropna(),
        df_all['Generosity'].dropna(),
        df_all['Perceptions of corruption'].dropna(),
        df_all['Positive affect'].dropna(),
        df_all['Negative affect'].dropna()],
        patch_artist=True,
        widths = 0.8)

    #Title
  fig2.suptitle("Comparison of Indicators from 2005–2023", fontsize=18, fontweight='bold')

  titles = ['Life Ladder & Log GDP per capita', 'Healthy life expectancy at birth', 'Other Metrics']
  xticks = [ ['Life Ladder', 'Log GDP per capita'], ['Healthy life expectancy at birth'],
        ['Social Support', 'Freedom', 'Generosity', 'Corruption', 'Positive', 'Negative']
    ]

  for i, ax_i in enumerate(ax2):
        ax_i.set_title(titles[i])
        ax_i.set_xticklabels(xticks[i], rotation=45, ha='right')

    # Coloring Boxplot – Styling

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

    # Apply styling
  style_boxplot(box4, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  style_boxplot(box5, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  style_boxplot(box6, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  st.pyplot(fig2)



  with st.expander("🔍 Interpretation of 2005-2023 Boxplot"):
    st.markdown("""
    ### Detailed Interpretation of Boxplots

    **a) Life Ladder & Log GDP per capita**

    - **Life Ladder**  
    • Values mainly range between 2 and 8  
    • The median is approximately 5  
    • Some outliers fall below 2  
    **Analysis:** About half of the countries are below average satisfaction, and half are above. Globally, most people report moderate happiness—neither extremely happy nor extremely unhappy.

    - **Log GDP per capita**  
    • Values range between 7 and 12  
    • The median is around 10  
    • No obvious outliers  
    **Analysis:** The range reveals significant income disparities between the wealthiest and poorest countries. A higher Log GDP per capita reflects greater economic prosperity and a higher standard of living. The median being closer to the upper end suggests that most countries are middle- to high-income, with fewer at the very low end.

    ---

    **b) Healthy Life Expectancy at Birth**

    - Values range from approximately 40 to 70 years  
    - The median is around 65 years  
    - Several outliers exist below 40 years—some even under 10, which may indicate potential data errors  
    **Note:** Extremely low values may require further investigation to confirm their accuracy.

    ---

    **c) Other Metrics**

    - **Social Support**  
    • Values mainly range between 0.5 and 1.0  
    • The median is around 0.8  
    • Some outliers  
    **Analysis:** A median of 0.8 suggests that most countries report high levels of social support.

    - **Freedom to Make Life Choices**  
    • Values mainly range between 0.7 and 0.9  
    • The median is around 0.8  
    • Some outliers  
    **Analysis:** Most countries demonstrate relatively high levels of personal freedom, with a narrow range concentrated at the upper end of the scale.

    - **Generosity**  
    • Shows a wide distribution, including negative values  
    • Many outliers are visible  
    **Analysis:** The variability in generosity scores may reflect cultural, economic, or societal differences in how generosity is measured or expressed.

    - **Perceptions of Corruption**  
    • Values range between 0.2 and 1.0  
    • Numerous outliers  
    **Analysis:** The median corruption score is approximately 0.8 (on a scale where higher means more perceived corruption), indicating that many countries report relatively high corruption levels. The outliers highlight significant variation across nations, likely tied to unique political or social conditions.

    - **Positive Affect**  
    • Values range between 0.3 and 0.9  
    • The median is around 0.7  
    **Analysis:** On average, countries report relatively high levels of positive emotion, though there is noticeable variation.

    - **Negative Affect**  
    • Values range between 0 and 0.5  
    • The median is around 0.3  
    • Fewer outliers  
    **Analysis:** Most countries report relatively low levels of negative emotion.

    ---

    """)
    
  st.markdown("""
    ### Conclusion

    The boxplots reveal differing distributions and scales across indicators. Notable outliers in **life expectancy** and **generosity** suggest areas for further investigation. Additionally, the inconsistent scaling of indicators implies that a unified visual representation could improve readability.

    ---
    """)

  st.markdown("""
    ### What About a Boxplot from 2023?

    To get a snapshot of the **current year**, we will now generate a **boxplot specifically for 2023** and analyze its insights.
    """)


  df_2023 = df_all[df_all['year'] == 2023]

  #show Boxplots with nearly same scale
  fig, ax = plt.subplots(1,3, figsize=(12, 6))

  box1 = ax[0].boxplot(
    [df_2023['Life Ladder'], df_2023['Log GDP per capita'].dropna()],
     patch_artist=True,
     widths = 0.3)

  #show Boxplots with nearly same scale
  box2 = ax[1].boxplot(
    [df_2023['Healthy life expectancy at birth'].dropna()],
    patch_artist=True,
    widths = 0.15)

  #show Boxplots with nearly same scale
  box3 = ax[2].boxplot(
    [df_2023[col].dropna() for col in [
        'Social support', 'Freedom to make life choices', 'Generosity',
        'Perceptions of corruption', 'Positive affect', 'Negative affect'
    ]],
     patch_artist=True,
     widths = 0.8)

  #Title
  fig.suptitle("Comparison of Indicators from 2023", fontsize=18, fontweight='bold')

  #Subplot Titels

  titles = ['Life Ladder & Log GDP per capita', 'Healthy life expectancy at birth', 'Other Metrics']
  xticks = [ ['Life Ladder', 'Log GDP per capita'], ['Healthy life expectancy at birth'],
    ['Social Support', 'Freedom', 'Generosity', 'Corruption', 'Positive', 'Negative']
  ]

  for i, ax_i in enumerate(ax):
    ax_i.set_title(titles[i])
    ax_i.set_xticklabels(xticks[i], rotation=45, ha='right')

  # Coloring Boxplot – Styling

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

  # Apply styling
  style_boxplot(box1, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  style_boxplot(box2, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  style_boxplot(box3, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  st.pyplot(fig)
  
  with st.expander("🔍 Interpretation of 2023 Boxplot"):
    st.markdown("""
     ### Interpreting 2023 Data

     The data from **2023** appears more stable, with **fewer extreme values (outliers)** across several categories.  
     This may suggest improvements in data collection or reflect a broader trend of stability in key indicators across countries.

     However, some categories—such as **Generosity**—continue to show greater variability, highlighting ongoing challenges or country-specific differences.
        """)

  # # Add a title to the page
  # st.title("Distribution of Indices for Selected Years")
  
  # # Define the styling function
  # def style_boxplot(box, facecolor, edgecolor, linewidth, flier_color, median_color, whisker_color):
  #   for patch in box['boxes']:
  #       patch.set_facecolor(facecolor)
  #       patch.set_edgecolor(edgecolor)
  #       patch.set_linewidth(linewidth)
  #   for flier in box['fliers']:
  #       flier.set(marker='o', markerfacecolor=flier_color, alpha=0.5)
  #   for median in box['medians']:
  #       median.set(color=median_color, linewidth=2)
  #   for whisker in box['whiskers']:
  #       whisker.set(color=whisker_color, linewidth=1, linestyle='--')

  # # Apply styling to each boxplot
  # style_boxplot(box1, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  # style_boxplot(box2, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  # style_boxplot(box3, '#6E66CC', 'black', 1, '#FF7BAC', '#FFD000', 'black')
  
  # #Distribution of indices next to eachother for 2013, 2018, 2021, 2023 (to compare pro and after covid)

  # # List of indices to analyze
  # indices = ['Life Ladder', 'Log GDP per capita', 'Social support',
  #           'Healthy life expectancy at birth', 'Freedom to make life choices',
  #           'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect']

  # # Specify the years we would like to to analyze
  # years_to_plot = [2013, 2018, 2021,2023]

  # # Create a figure for the histograms
  # fig, axes = plt.subplots(nrows=len(indices), ncols=len(years_to_plot), figsize=(20, 25))  # Rows for indices, columns for years
  # fig.suptitle("Distribution of Indices for Selected Years", fontsize=20)

  # # Loop over each variable and year
  # for i, index in enumerate(indices):
  #     for j, year in enumerate(years_to_plot):
  #         ax = axes[i, j]

  #         # Filter data for the current year
  #         yearly_data = df_all[df_all['year'] == year]

  #         # Plot the histogram
  #         ax.hist(yearly_data[index].dropna(), bins=20, color='#6E66CC', edgecolor='black')
  #         ax.set_title(f"{index} ({year})")
  #         ax.set_xlabel(index)
  #         ax.set_ylabel("Frequency")

  # plt.tight_layout(rect=[0, 0, 1, 0.95])
  # st.pyplot(fig)
  


  # Ranking of the Top 10 Countries

  #Data sorting for Life Ladder
  df_2023 = df_all[df_all['year'] == 2023]
  df_2023_sorted = df_2023.sort_values(by='Life Ladder', ascending=False)
  top_10 = df_2023_sorted.head(10)
  bottom_10 = df_2023_sorted.tail(10)

  #Diagram in Seaborn

  fig, axes = plt.subplots(1, 2, figsize=(14, 6))

  #show Barplot Top 10
  sns.barplot(
      ax=axes[0],
      data = top_10,
      x='Life Ladder',
      y='Country name',
      hue='Continent',
      palette = continent_colors
  )

  # delet line top and right
  axes[0].spines['top'].set_visible(False)
  axes[0].spines['right'].set_visible(False)
  axes[0].spines['bottom'].set_visible(False)

  axes[0].set_title("Top 10 Countries \n by Life Ladder (2023)\n\n", fontsize = 16, fontweight = "bold")
  axes[0].set_xlabel("Life Ladder", fontsize = 12)
  axes[0].set_ylabel("Countries", fontsize = 12)
  axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)
  axes[0].grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
  axes[0].set_axisbelow(True)

  #show Barplot Bottom 10
  sns.barplot(
      ax=axes[1],
      data = bottom_10,
      x='Life Ladder',
      y='Country name',
      hue='Continent',
      palette= continent_colors
  )
  axes[1].set_title("10 Lowest-Ranked Countries \n by Life Ladder (2023)\n\n", fontsize = 16, fontweight = "bold")
  axes[1].set_xlabel("Life Ladder", fontsize = 12)
  axes[1].set_ylabel("Countries", fontsize = 12);
  axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4)
  axes[1].grid(True, which='major', axis='x', linestyle='--', linewidth=0.5)
  axes[1].set_axisbelow(True)

  plt.tight_layout()
  st.pyplot(plt)

  #To analyze further, we check the average Life Ladder per Continent

  continent_avg_life_ladder = df_all.groupby('Continent')['Life Ladder'].mean().reset_index()
  continent_avg_life_ladder = continent_avg_life_ladder.sort_values('Life Ladder', ascending=False)


  # Create a bar plot to show average Life Ladder by continent
  fig = px.bar(continent_avg_life_ladder, x="Life Ladder", y="Continent",
              title="Average Life Ladder by Continent", color = "Continent",
              color_discrete_map=continent_colors,
              orientation = "h",
              labels={"Continent": "Continent", "Life Ladder": "Average Life Ladder"})

  st.plotly_chart(fig)

  #Life ladder per country (top 10 per each continent)

  # Group by 'Continent' and 'Country name' to get the average "Life Ladder"
  # Calculate the average "Life Ladder" for each country within each continent
  top_countries_life_ladder = df_all.groupby(['Continent', 'Country name'])['Life Ladder'].mean().reset_index()

  # For each continent, select the top 10 countries based on the average "Life Ladder"
  top_countries_per_continent = top_countries_life_ladder.groupby('Continent').apply(lambda x: x.nlargest(10, 'Life Ladder')).reset_index(drop=True)

  # Calculate the aggregate "Life Ladder" for each continent
  continent_life_ladder = top_countries_per_continent.groupby('Continent')['Life Ladder'].mean().reset_index()
  continent_life_ladder = continent_life_ladder.sort_values(by='Life Ladder', ascending=False)

  # Ensure the continents are ordered based on their aggregate "Life Ladder"
  continent_order = continent_life_ladder['Continent'].tolist()

  # Plot the data
  fig = px.bar(top_countries_per_continent,
              x='Country name',
              y='Life Ladder',
              color='Continent',
              category_orders={'Continent': continent_order},  # Set the order of continents
              color_discrete_map=continent_colors,
              title='Top 10 Countries by Average Life Ladder for Each Continent',
              labels={'Life Ladder': 'Average Life Ladder', 'Country name': 'Country', 'Continent': 'Continent'})


  st.plotly_chart(fig)

  #Bottom 10 countries per continent

  # Calculate the average "Life Ladder" for each country within each continent
  top_countries_life_ladder = df_all.groupby(['Continent', 'Country name'])['Life Ladder'].mean().reset_index()

  # For each continent, select the bottom 10 countries based on the average "Life Ladder"
  bottom_countries_per_continent = top_countries_life_ladder.groupby('Continent').apply(lambda x: x.nsmallest(10, 'Life Ladder')).reset_index(drop=True)

  # Calculate the aggregate "Life Ladder" for each continent (for sorting continents)
  continent_life_ladder = bottom_countries_per_continent.groupby('Continent')['Life Ladder'].mean().reset_index()
  continent_life_ladder = continent_life_ladder.sort_values(by='Life Ladder', ascending=True)

  # Ensure the continents are ordered based on their aggregate "Life Ladder" in ascending order
  continent_order = continent_life_ladder['Continent'].tolist()

  # Plot the data
  fig = px.bar(bottom_countries_per_continent,
              x='Country name',
              y='Life Ladder',
              color='Continent',
              category_orders={'Continent': continent_order},  # Set the order of continents
              color_discrete_map=continent_colors,
              title='Bottom 10 Countries by Average Life Ladder for Each Continent',
              labels={'Life Ladder': 'Average Life Ladder', 'Country name': 'Country', 'Continent': 'Continent'})

  st.plotly_chart(fig)

  #Life ladder by continent over time (animation)


  df_anim = df_all[df_all['year']>2010].sort_values(by='year')

  fig = px.box(df_anim, x="Continent", y="Life Ladder", color="Continent",
              color_discrete_map=continent_colors,
              animation_frame="year", range_y=[0,9], title = "Life Ladder by continent over time (animation)",
              color_discrete_sequence=px.colors.qualitative.Dark24)
  st.plotly_chart(fig)

  #AVERAGE LIFE LADDER BY CONTINENT


  #Evolution by year by Life Ladder
  # Group by 'year' and calculate the global average 'Life Ladder'
  global_life_ladder = df_all.groupby("year")["Life Ladder"].mean().reset_index()

  # Create a line plot
  fig = px.line(global_life_ladder, x="year", y="Life Ladder",
                title="Evolution of Life Ladder (Global)",
                labels={"Life Ladder": "Life Ladder", "year": "Year"})

  st.plotly_chart(fig)

  #Evolution of Life Ladder by Continent
  # Group by 'year' and 'Region' to get the average Life Ladder for each region
  region_life_ladder = df_all.groupby(["year", "Continent"])["Life Ladder"].mean().reset_index()

  # Create a line plot for each region
  fig = px.line(region_life_ladder, x="year", y="Life Ladder", color="Continent",
                title="Evolution of Life Ladder by Continent",
                color_discrete_map=continent_colors,
                labels={"Life Ladder": "Life Ladder", "year": "Year", "Continent": "Continent"})

  st.plotly_chart(fig)

  #Evolution of Life Ladder by Region

  mean_per_year = df_all.groupby('year')['Life Ladder'].mean().reset_index()
  region_per_year = df_all.groupby(['Region', 'year'])['Life Ladder'].mean().reset_index()
  region_europe = region_per_year[region_per_year['Region'].isin(['Southern Europe', 'Western Europe', 'Northern Europe', 'Eastern Europe'])]
  region_africa = region_per_year[region_per_year['Region'].isin(['Southern Africa', 'Western Africa', 'Northern Africa', 'Eastern Africa', 'Middle Africa'])]
  region_asia = region_per_year[region_per_year['Region'].isin(['Southern Asia', 'South Eastern Asia', 'Western Asia',  'Eastern Asia', 'Central Asia'])]
  region_america = region_per_year[region_per_year['Region'].isin(['South America', 'Northern America', 'Central America'])]
  region_other = region_per_year[region_per_year['Region'].isin(['Australia and New Zealand', 'Caribbean'])]

  fig, axes = plt.subplots(2, 3, figsize=(20, 12))

  #order of the regions
  europe_order = ['Western Europe', 'Northern Europe','Southern Europe','Eastern Europe']
  africa_order = ['Northern Africa','Southern Africa','Eastern Africa', 'Western Africa', 'Middle Africa']
  asia_order = ['Central Asia','Eastern Asia','Western Asia', 'South Eastern Asia','Southern Asia']
  america_order = ['Northern America', 'Central America','South America']
  other_order = ['Australia and New Zealand', 'Caribbean']


  #show Lineplot

  sns.lineplot(ax=axes[0, 0], data=mean_per_year, x='year', y='Life Ladder', linewidth=2, marker='o', color = '#6E66CC') #used above for loop instead of this code
  #sns.lineplot(ax=axes[0, 1], data=region_per_year, x='year', y='Life Ladder', hue = 'Region', marker='o')
  sns.lineplot(ax=axes[0, 1], data=region_europe, x='year', y='Life Ladder', linewidth=2, hue = 'Region', hue_order=europe_order, marker='o', palette=region_colors)
  sns.lineplot(ax=axes[0, 2], data=region_africa, x='year', y='Life Ladder', linewidth=2, hue = 'Region', hue_order=africa_order, marker='o', palette=region_colors)
  sns.lineplot(ax=axes[1, 0], data=region_asia, x='year', y='Life Ladder', linewidth=2, hue = 'Region', hue_order=asia_order, marker='o', palette=region_colors)
  sns.lineplot(ax=axes[1, 1], data=region_america, x='year', y='Life Ladder', linewidth=2, hue = 'Region', marker='o', hue_order=america_order, palette=region_colors)
  sns.lineplot(ax=axes[1, 2], data=region_other, x='year', y='Life Ladder', linewidth=2, hue = 'Region', hue_order = other_order, marker='o', palette=region_colors)

  titles = [
      "World – Mean Life Ladder Over Years\n\n\n",
      "Europe – Mean Life Ladder by Region\n\n\n",
      "Africa – Mean Life Ladder by Region\n\n\n",
      "Asia – Mean Life Ladder by Region\n\n\n",
      "America – Mean Life Ladder by Region\n\n\n",
      "Other Regions – Mean Life Ladder\n\n\n"
  ]

  life_ladder_avg = df_all['Life Ladder'].mean()

  # print average Line into the Plot
  for ax in axes.flat:
      ax.axhline(y=life_ladder_avg, color='black', linestyle='--', linewidth=1.5, label='Average Life Ladder')
      ax.legend(loc='upper right')

  for ax, title in zip(axes.flat, titles):
      ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
      ax.set_title(title, fontsize=14, fontweight = 'bold')
  #axes[1, 0].legend(loc='lower left')


  plt.tight_layout()
  st.pyplot(plt);

  plt.figure(figsize = [200,200])
  fig = px.scatter(df_2023,
                  x = 'Life Ladder',
                  y = 'Healthy life expectancy at birth',
                  #size = 'Life Ladder',
                  hover_name = 'Country name',
                  title='Life Ladder vs. Healthy life expectancy at birth',
                  facet_col = "Continent",
                  color = "Region",
                  #size_max=20,
                  color_discrete_map=region_colors
                  )
  st.plotly_chart(fig);

  #WHAT DOES HAPPINESS LOOK LIKE WHEN MAPPED

  # Use the necessary columns without modifying the original dataset
  map_data = df_all[['Country name', 'Life Ladder']]

  # Create a choropleth map for Life Ladder scores
  fig = px.choropleth(
      map_data,
      locations="Country name",  # Column with country names
      locationmode="country names",  # Match country names to Plotly's list
      color="Life Ladder",  # Happiness metric for coloring
      title="World Happiness Map (Life Ladder Scores)",
      color_continuous_scale=px.colors.sequential.RdBu,  # Customizable color scale, Plasma, RdBu is good
      labels={"Life Ladder": "Happiness Score"},  # Tooltip label customization
  )

  # Customize the layout (optional)
  fig.update_layout(
      geo=dict(showframe=False, showcoastlines=True, projection_type="equirectangular")
  )

  # Display the map
  st.plotly_chart(fig)

  # Count the number of unique countries per year
  countries_per_year = df_all.groupby('year')['Country name'].nunique().reset_index()
  countries_per_year.rename(columns={'Country name': 'Number of Countries'}, inplace=True)

  # Plot the data
  fig = px.line(countries_per_year, x='year', y='Number of Countries',
                title="Number of Countries in Dataset Over the Years",
                labels={'year': 'Year', 'Number of Countries': 'Number of Countries'},
                markers=True)
  fig.update_traces(line=dict(width=3), marker=dict(size=8))

  # Update the x-axis show years
  fig.update_xaxes(type='category')

  # Show the plot
  st.plotly_chart(fig)

  #MISSING DATA PER COUNTRY BY YEAR

  from matplotlib.colors import ListedColormap

  # Check for missing values by country and year
  missing_data = df_all.pivot_table(index='Country name', columns='year', aggfunc='size', fill_value=0)

  # Calculate the number of missing values per country per year
  missing_counts = df_all.groupby(['Country name', 'year']).size().unstack(fill_value=0)

  limited_countries = missing_counts[(missing_counts > 0).sum(axis=1) <= 3].index

  # Display a summary of missing counts
  #print("Missing Data Summary (Presence of Data per Year for Each Country):")
  #print(missing_counts)

  # Check countries with inconsistent appearances across years
  inconsistent_countries = missing_counts[(missing_counts == 0).any(axis=1)]

  custom_cmap = ListedColormap(["#7EE2E1", "#FE4F57"])

  # Create a heatmap to see missing values
  plt.figure(figsize=(15, 10))
  ax = sns.heatmap(missing_counts == 0, cmap=custom_cmap, cbar_kws={'label': 'Missing Data'},
              xticklabels=True, yticklabels=True)

  new_yticks = [name if name in limited_countries else "" for name in missing_counts.index]
  ax.set_yticklabels(new_yticks, fontsize=10)

  plt.title("Heatmap of Missing Data (Countries by Year)", fontsize=16)
  plt.xlabel("Year", fontsize=14)
  plt.ylabel("Countries", fontsize=14)
  st.pyplot(plt)

  #Analysis: the low number of countries in 2005, combined with the significant amount of missing data for that year, led us to decide to exclude 2005 from our analysis

  import pandas as pd
  import plotly.express as px


  import matplotlib.pyplot as plt
  import seaborn as sns

  plt.figure(figsize=[400,400])
  fig = px.scatter(df_all, x="Life Ladder", y="Social support",
                  hover_name="Healthy life expectancy at birth", title="Life Ladder vs social support",trendline="ols")
  st.plotly_chart(fig);


if page == pages[3] : 
  st.markdown("<h3 style='color: #6E66CC;'>Modeling</h3>", unsafe_allow_html=True)
  
  df_all['filled_percentage'] = df_all.notna().sum(axis=1) / len(df_all.columns)

  #How much rows have missing more than 50% of the data
  rows_below_50 = df_all[df_all['filled_percentage'] < 0.5]

  print("\n")
  print("Shape of rows_below_50:", rows_below_50.shape)
  print("No row is under 50% we can use the Dataset of every row")

  #Data clean up –> delete columns


  delete_col = ["year", 'Continent', 'Country name', 'filled_percentage', 'People per square kilometer','Population in millions', 'Area in square kilometer', "Area Code (M49)"]

  df_clean = df_all.drop(delete_col, axis = 1)


  from sklearn.model_selection import train_test_split

  feats = df_clean.drop('Life Ladder', axis = 1)
  target = df_clean['Life Ladder']

  #separate the dataset into a training set and a test set – the test set contains 20% of the data

  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state = 42)

  print("Traindata, Rows and Columns:", X_train.shape)
  print("Testdata, Rows and Columns:", X_test.shape)

  #Data clean up –> handle outliers

  #We remove all values in the column "Healthy life expectancy at birth" with the value < 50 and replace the value with median

  threshold = 45

  #we start with showing us the outlier rows for X_train
  outliers = X_train[X_train['Healthy life expectancy at birth'] < threshold]

  #print(outliers)

  #median per Region

  medians_per_region = X_train[X_train['Healthy life expectancy at birth'] >= threshold] \
      .groupby('Region')['Healthy life expectancy at birth'].median()
  print("Median per Region (Training Dataset):")
  print(medians_per_region)

  #replace outliers in Training Set

  X_train['Healthy life expectancy at birth'] = X_train.apply(
      lambda row: medians_per_region[row['Region']]
      if row['Healthy life expectancy at birth'] < threshold
      else row['Healthy life expectancy at birth'],
      axis=1
  )

  #replace outliers in Test Set

  X_test['Healthy life expectancy at birth'] = X_test.apply(
      lambda row: medians_per_region[row['Region']]
      if row['Healthy life expectancy at birth'] < threshold
      else row['Healthy life expectancy at birth'],
      axis=1
  )

  #Data clean up –> NaN replace missing values with the median (per subregion)
  # We do that after splitting the dataset into test and training sets because we don't want to have a data leakage

  #We check which column has NaNs
  #print(X_train.isna().sum())

  columns_to_fix = [
      "Log GDP per capita",
      "Social support",
      "Freedom to make life choices",
      "Generosity",
      "Perceptions of corruption",
      "Positive affect",
      "Negative affect"
  ]

  #we need a function for cleaning
  def fix_column_by_region(dataframe, column_name, group_column="Region"):
      # calculate Median per Region for columns
      medians_per_region = dataframe.groupby(group_column)[column_name].median()

  # replace NaN-Werte in column with Median
      dataframe[column_name] = dataframe.apply(
          lambda row: medians_per_region[row[group_column]]
          if pd.isna(row[column_name])
          else row[column_name],
          axis=1
      )
      return dataframe

  for col in columns_to_fix:
      X_train = fix_column_by_region(X_train, col)
      X_test = fix_column_by_region(X_test, col)



  #Data Clean -> Region could be One hot encode before split (try to do it after tho)

  #X_train = X_train.drop("Region", axis=1)
  #X_test = X_test.drop("Region", axis=1)

  #X_train.head()

  #Data clean up -> Standard-Scalar

  from sklearn.preprocessing import StandardScaler

  columns_to_scale = [
      "Log GDP per capita",
      "Social support",
      "Healthy life expectancy at birth",
      "Freedom to make life choices",
      "Generosity",
      "Perceptions of corruption",
      "Positive affect",
      "Negative affect"
  ]

  scaler = StandardScaler()


  # Fit
  X_train_scaled_columns = scaler.fit_transform(X_train[columns_to_scale])

  # Transform in X_test
  X_test_scaled_columns = scaler.transform(X_test[columns_to_scale])

  # Replace columns in X_train
  X_train[columns_to_scale] = X_train_scaled_columns

  # Replace columns in X_test
  X_test[columns_to_scale] = X_test_scaled_columns

  print(X_train.dtypes)  # Check column data types
  print(X_train.select_dtypes(include=['object']).head())  # Show any text data

  #One-hot encode the 'Region' column after splitting
  X_train_encoded = pd.get_dummies(X_train, columns=['Region'], drop_first=True)
  X_test_encoded = pd.get_dummies(X_test, columns=['Region'], drop_first=True)

  # Ensure that X_test_encoded has the same columns as X_train_encoded
  X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

  # Start Training: Linear Model

  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  import joblib
  
  # # Train the Linear Regression model on the encoded data
  #linear_model = LinearRegression(fit_intercept=True)
  #linear_model.fit(X_train_encoded, y_train)  # Use X_train_encoded
  
  # # Save the trained model using joblib
  
  #joblib.dump(linear_model, 'newest_linear_model.pkl')  # Save the model to a file
  #print("Model saved successfully!")
  
  linear_model = joblib.load('newest_linear_model.pkl')
  # Predict and evaluate on the encoded test data
  y_pred_linear_test = linear_model.predict(X_test_encoded)  # Use X_test_encoded

  # Predict and evaluate on the encoded train data
  y_pred_linear_train = linear_model.predict(X_train_encoded)  # Use X_train_encoded

  # Evaluation for test data
  mse_linear_test = mean_squared_error(y_test, y_pred_linear_test)
  mae_linear_test = mean_absolute_error(y_test, y_pred_linear_test)
  r2_linear_test = r2_score(y_test, y_pred_linear_test)

  # Evaluation for train data
  mse_linear_train = mean_squared_error(y_train, y_pred_linear_train)
  mae_linear_train = mean_absolute_error(y_train, y_pred_linear_train)
  r2_linear_train = r2_score(y_train, y_pred_linear_train)

  # Print the results
  print("Linear Regression Model Performance:")
  print(f"Mean Squared Error (Test): {mse_linear_test:.4f}")
  print(f"Mean Absolute Error (Test): {mae_linear_test:.4f}")
  print(f"R² Score (Test): {r2_linear_test:.4f}")
  print(f"Mean Squared Error (Train): {mse_linear_train:.4f}")
  print(f"Mean Absolute Error (Train): {mae_linear_train:.4f}")
  print(f"R² Score (Train): {r2_linear_train:.4f}")


  # Feature Importance for Linear Regression (Coefficients)
  coefficients = pd.DataFrame({
      'Feature': X_train_encoded.columns,  # Feature names from the encoded training set
      'Coefficient': linear_model.coef_  # Coefficients from the linear model
  })
  coefficients = coefficients.sort_values(by='Coefficient', ascending=False)

  print("\nFeature Importance for Linear Regression (Coefficients):")
  print(coefficients)



  #Interpretation: R2=0.8228 means the Linear Regression model explains 82.28% of the variance in the target variable "Life Ladder."
  #A higher 𝑅2 (closer to 1) indicates better model performance.
  #The model performs quite well but is not perfect. It captures most of the relationships between the features and "Life Ladder," but there is still 23.73% variance unexplained.
  #This suggests that there might be non-linear relationships or factors influencing "Life Ladder" that the linear model can't capture.

  #Scatterplot
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(8, 6))  # Set figure size (width, height)

  plt.scatter(y_test, y_pred_linear_test, label="Predictions vs actual", color='#6E66CC', edgecolor='black', alpha=0.7)
  plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="black", linewidth=1, linestyle="--", label="Ideal Fit")
  plt.xlabel("Actual Values", family = 'monospace')
  plt.ylabel("Prediction", family = 'monospace')
  plt.title("Actual Values vs. Prediction", fontsize=18, fontweight='bold')
  plt.legend()
  st.pyplot(fig)

  # Decision Tree Model - Purpose: Predict the actual numeric value of the "Life Ladder" variable (continuous target).


  from sklearn.tree import DecisionTreeRegressor
  from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
  from sklearn.tree import plot_tree
  import matplotlib.pyplot as plt

  #comment out the code below so the model does not retrain

  # Train the Decision Tree model with a limited max depth 
  #decision_tree_model = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_leaf=10, min_samples_split=2)
  #decision_tree_model.fit(X_train_encoded, y_train)

  # Save the trained model using joblib
  #joblib.dump(decision_tree_model, 'newest_decision_tree_model.pkl')  # Save the model to a file
  #print("Model saved successfully!")
  
  #Load the saved model
  decision_tree_model = joblib.load('newest_decision_tree_model.pkl')
  
  
  # Predict using the encoded test data
  y_pred_tree_test = decision_tree_model.predict(X_test_encoded)

  # Predict using the encoded train data
  y_pred_tree_train = decision_tree_model.predict(X_train_encoded)

  # Evaluate for test data
  mse_tree_test = mean_squared_error(y_test, y_pred_tree_test)
  mae_tree_test = mean_absolute_error(y_test, y_pred_tree_test)
  r2_tree_test = r2_score(y_test, y_pred_tree_test)

  # Evaluate for train data
  mse_tree_train = mean_squared_error(y_train, y_pred_tree_train)
  mae_tree_train = mean_absolute_error(y_train, y_pred_tree_train)
  r2_tree_train = r2_score(y_train, y_pred_tree_train)

  # Print the results
  print("Decision Tree Model Performance:")
  print(f"Mean Squared Error (Test): {mse_tree_test:.4f}")
  print(f"Mean Absolute Error (Test): {mae_tree_test:.4f}")
  print(f"R² Score (Test): {r2_tree_test:.4f}")
  print(f"Mean Squared Error (Train): {mse_tree_train:.4f}")
  print(f"Mean Absolute Error (Train): {mae_tree_train:.4f}")
  print(f"R² Score (Train): {r2_tree_train:.4f}")

  # Feature Importance for Decision Tree
  feature_importance_tree = pd.DataFrame({
      'Feature': X_train_encoded.columns,  # Feature names from the training set
      'Importance': decision_tree_model.feature_importances_  # Feature importances from the decision tree model
  })
  feature_importance_tree = feature_importance_tree.sort_values(by='Importance', ascending=False)

  print("\nFeature Importance for Decision Tree:")
  print(feature_importance_tree)

  # ================================
  # Simpler Visualization of the Decision Tree
  # ================================

  plt.figure(figsize=(12, 8))
  plot_tree(decision_tree_model,
            feature_names=X_train_encoded.columns,
            filled=True,
            rounded=True,
            fontsize=12)
  plt.title("Simplified Decision Tree Visualization")
  st.pyplot(plt)

  #Start training RandomForest

  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_squared_error, r2_score

  #comment out the code below so the model does not retrain
  # Initialisation
  # rf_model = RandomForestRegressor(random_state=42, n_estimators=100, min_samples_split=2, 
  #                                  min_samples_leaf=1, max_features='log2', max_depth=None, 
  #                                  bootstrap=False)

  # # Train the model using the encoded data
  # rf_model.fit(X_train_encoded, y_train)  # Use X_train_encoded, not X_train

  # #Save the trained model using joblib. Once I created a dump file, 
  # # I commented it out and put the file to github
  # joblib.dump(rf_model, 'newest_rf_model.pkl')  # Save the model to a file
  
  #Load the saved model
  rf_model = joblib.load('newest_rf_model.pkl')
  
  # Predict using the encoded test data
  y_pred_test = rf_model.predict(X_test_encoded)  # Use X_test_encoded, not X_test

  # Predict using the encoded train data
  y_pred_train = rf_model.predict(X_train_encoded)  # Use X_train_encoded for training predictions

  # Evaluation for test data
  mae_rf_test = mean_absolute_error(y_test, y_pred_test)
  mse_rf_test = mean_squared_error(y_test, y_pred_test)
  r2_rf_test = r2_score(y_test, y_pred_test)

  # Evaluation for train data
  mae_rf_train = mean_absolute_error(y_train, y_pred_train)
  mse_rf_train = mean_squared_error(y_train, y_pred_train)
  r2_rf_train = r2_score(y_train, y_pred_train)

  # Print the results
  print(f"Mean Absolute Error (Test) Random Forest: {mae_rf_test:.2f}")
  print(f"Mean Squared Error (Test) Random Forest: {mse_rf_test:.2f}")
  print(f"R² Score (Test) Random Forest: {r2_rf_test:.2f}")
  print(f"Mean Absolute Error (Train) Random Forest: {mae_rf_train:.2f}")
  print(f"Mean Squared Error (Train) Random Forest: {mse_rf_train:.2f}")
  print(f"R² Score (Train) Random Forest: {r2_rf_train:.2f}")

  # Feature importances from the trained Random Forest model

  import numpy as np

  rf_feature_importances = rf_model.feature_importances_

  # Sort feature importances in descending order and their corresponding feature names
  rf_sorted_idx = np.argsort(rf_feature_importances)[::-1]
  rf_sorted_importances = rf_feature_importances[rf_sorted_idx]
  rf_sorted_features = [X_train_encoded.columns[i] for i in rf_sorted_idx]

  # Print feature importance as a list
  print("\nRandom Forest Feature Importances:")
  for feature, importance in zip(rf_sorted_features, rf_sorted_importances):
      print(f"{feature}: {importance:.4f}")

  # Get top 10 features for each model

  # For Linear Regression (based on coefficients)
  top_10_linear = coefficients.head(10)

  # For Decision Tree (based on feature importance)
  top_10_tree = feature_importance_tree.head(10)

  # For Random Forest (based on feature importance)
  top_10_rf_features = rf_sorted_features[:10]
  top_10_rf_importances = rf_sorted_importances[:10]

  # Create subplots for side-by-side comparison
  fig, axes = plt.subplots(1, 3, figsize=(18, 6))

  # Linear Regression Feature Importance Plot (based on coefficients)
  sns.barplot(x='Coefficient', y='Feature', data=top_10_linear, ax=axes[0], palette='viridis', hue='Feature')
  axes[0].set_title('Feature Importance for Linear Regression')
  axes[0].set_xlabel('Coefficient Value', fontweight='bold')
  axes[0].set_ylabel('Feature', fontweight='bold')

  # Decision Tree Feature Importance Plot
  sns.barplot(x='Importance', y='Feature', data=top_10_tree, ax=axes[1], palette='viridis', hue='Feature')
  axes[1].set_title('Feature Importance for Decision Tree')
  axes[1].set_xlabel('Importance', fontweight='bold')
  axes[1].set_ylabel('Feature', fontweight='bold')

  # Random Forest Feature Importance Plot
  sns.barplot(x=top_10_rf_importances, y=top_10_rf_features, ax=axes[2], palette='viridis')
  axes[2].set_title('Feature Importance for Random Forest')
  axes[2].set_xlabel('Importance', fontweight='bold')
  axes[2].set_ylabel('Feature', fontweight='bold')

  # Adjust layout for better spacing
  plt.subplots_adjust(wspace=0.5)  # Increase space between the subplots
  plt.tight_layout()
  st.pyplot(plt)

  
  #Model performance comparison

  print("\nModel Performance Comparison:")

  # For Linear Regression
  print(f"Linear Regression - Train MAE: {mae_linear_train:.4f}, Test MAE: {mae_linear_test:.4f}, "
        f"Train MSE: {mse_linear_train:.4f}, Test MSE: {mse_linear_test:.4f}, "
        f"Train R²: {r2_linear_train:.4f}, Test R²: {r2_linear_test:.4f}")

  # For Decision Tree
  print(f"    Decision Tree - Train MAE: {mae_tree_train:.4f}, Test MAE: {mae_tree_test:.4f}, "
        f"Train MSE: {mse_tree_train:.4f}, Test MSE: {mse_tree_test:.4f}, "
        f"Train R²: {r2_tree_train:.4f}, Test R²: {r2_tree_test:.4f}")

  # For Random Forest
  print(f"    Random Forest - Train MAE: {mae_rf_train:.4f}, Test MAE: {mae_rf_test:.4f}, "
        f"Train MSE: {mse_rf_train:.4f}, Test MSE: {mse_rf_test:.4f}, "
        f"Train R²: {r2_rf_train:.4f}, Test R²: {r2_rf_test:.4f}")

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

