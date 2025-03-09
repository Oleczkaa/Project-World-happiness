import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
  st.markdown("<h3 style='color: #6E66CC;'>Project Description</h3>", unsafe_allow_html=True)
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
              The goal is to identify patterns that influence happiness across different countries and continents.
              """)

if page == pages[1] : 
  st.markdown("<h3 style='color: #6E66CC;'>Presentation of Data</h3>", unsafe_allow_html=True)

  st.dataframe(df_all.head(10))
  st.write(df_all.shape)
  st.dataframe(df_all.describe())
  if st.checkbox("Show NA") :
    st.dataframe(df_all.isna().sum())

if page == pages[2] : 
  st.markdown("<h3 style='color: #6E66CC;'>Data Vizualization</h3>", unsafe_allow_html=True)

  #BOX PLOT FOR MULTIPLE NUMERICAL VARIABLES

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

      
      






