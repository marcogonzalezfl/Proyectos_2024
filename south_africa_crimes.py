#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import numpy as np


# In[3]:


# Leer el archivo CSV

df_crime = pd.read_csv('C:/Users/Marco/Proyectos/south_africa/crime_sa/SouthAfricaCrimeStats_v2.csv')
df_crime


# In[4]:


# Análisis básico

print(df_crime.info())


# In[5]:


print(df_crime.describe())


# In[6]:


# Filtrar datos por provincia

province_group = df_crime.groupby('Province').sum(numeric_only=True)


# In[7]:


# Visualizar la tendencia del crimen total por provincia

province_group.T.plot(figsize=(10, 6), marker='o')
plt.title('Criminal Activity Trends by Province (2005-2016)')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.legend(title='Province', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[8]:


# Asegurarnos de trabajar con columnas numéricas únicamente para la agregación
numeric_columns = df_crime.columns[3:]  # Suponiendo que las primeras tres columnas son 'Province', 'Station' y 'Category'

# Agrupación por 'Province' y 'Category', sumando los valores numéricos
df_group = df_crime.groupby(['Province', 'Category'])[numeric_columns].sum().reset_index()

# Mostrar los primeros registros del nuevo dataset
print(df_group.head())


# In[9]:


# Assuming 'Province' is the column containing province names
province_count = df_group['Province'].nunique()

print(f"There are {province_count} unique provinces.")


# In[10]:


import geopandas as gpd

# Replace with the path to the actual shapefile (.shp file)
shapefile_path = r"C:\Users\Marco\Proyectos\south_africa\div_politicas\zaf_admbnda_adm1_sadb_ocha_20201109.shp"

# Load the shapefile using Geopandas
gdf = gpd.read_file(shapefile_path)

# Check the first few rows to verify it's loaded correctly
print(gdf)


# In[11]:


# Assuming df_group is your crime data
df_long = df_group.melt(id_vars=['Province', 'Category'], 
                       value_vars=['2005-2006', '2006-2007', '2007-2008', 
                                    '2008-2009', '2009-2010', '2010-2011', 
                                    '2011-2012', '2012-2013', '2013-2014', 
                                    '2014-2015', '2015-2016'],
                       var_name='Year', value_name='CrimeCount')

# Check the reshaped dataframe
df_long


# In[71]:


import matplotlib.pyplot as plt

# Example: Plotting trends for a specific category (e.g., 'Arson') over time
category_data = df_long[df_long['Category'] == 'Arson']

# Pivoting the data for easier plotting
category_pivot = category_data.pivot_table(index='Year', columns='Province', values='CrimeCount')

# Plot the trends
category_pivot.plot(kind='line', figsize=(12, 8), title="Arson Trends by Province")
plt.xlabel('Year')
plt.ylabel('Crime Count')
plt.grid(True)
plt.legend(title="Province")
plt.show()


# In[73]:


df_merged_gdf


# In[77]:


print(df_merged_gdf['Category'].unique())
print(df_merged_gdf['Year'].unique())
print(df_merged_gdf['Province'].unique())


# In[81]:


# Reemplazar 'North Cape' por 'Northern Cape' en df_long
df_long['Province'] = df_long['Province'].replace({
    'North Cape': 'Northern Cape'
})



# In[89]:


gdf['ADM1_EN'] = gdf['ADM1_EN'].replace({'Nothern Cape': 'Northern Cape'})
df_long['Province'] = df_long['Province'].replace({'Nothern Cape': 'Northern Cape'})


# In[91]:


import geopandas as gpd
import pandas as pd

# Asegúrate de que df_long y gdf tengan el mismo formato de nombre para las provincias
# Verifica si hay diferencias en la representación de los nombres de las provincias (por ejemplo, 'Kwazulu/Natal' vs 'KwaZulu-Natal')
df_long['Province'] = df_long['Province'].replace({
    'Kwazulu/Natal': 'KwaZulu-Natal',  # Reemplaza si hay un formato inconsistente
    'Northern Cape': 'North Cape',      # Si fuera necesario hacer alguna corrección adicional
    'Ostkap': 'Eastern Cape',           # Si la provincia 'Ostkap' existe y se debe renombrar
    'Nordkap': 'Northern Cape'          # Reemplaza otras abreviaciones si son necesarias
})

# Asegúrate de que la columna 'Province' en gdf esté también en el formato adecuado
gdf['ADM1_EN'] = gdf['ADM1_EN'].replace({
    'Kwazulu/Natal': 'KwaZulu-Natal',  # Asegura la consistencia en los nombres
    'Northern Cape': 'North Cape',      # Si es necesario
    'Ostkap': 'Eastern Cape',           # Consistencia para otras provincias
    'Nordkap': 'Northern Cape'          # Reemplaza otras abreviaciones si son necesarias
})

# Asegúrate de que la columna 'Province' en df_long esté bien formateada
df_long['Province'] = df_long['Province'].str.strip()  # Eliminar espacios extra si existen

# Hacer la fusión de los dos dataframes basándose en la columna 'Province'
df_merged_gdf = gdf.merge(df_long, left_on='ADM1_EN', right_on='Province', how='left')

# Comprobar el dataframe resultante
print(df_merged_gdf)


# In[93]:


df_merged_gdf['Province'].unique()


# In[111]:


import imageio.v2 as imageio
import os
import matplotlib.pyplot as plt
from PIL import Image

# Function to generate the GIF
def create_gif(df_category, category, output_folder):
    images = []
    standard_size = (800, 800)  # Define a fixed size for all images

    # Loop through each year in the dataset and generate the corresponding image
    for year in df_category['Year'].unique():
        print(f"Procesando año: {year}")

        # Filter data for the current year
        year_data = df_category[df_category['Year'] == year]

        # Create the plot (replace with your own plotting code)
        fig, ax = plt.subplots(figsize=(10, 10))  # Set initial figure size
        # (Example plot, replace with your actual plotting code)
        ax.bar(year_data['Province'], year_data['CrimeCount'])

        # Save the figure temporarily
        image_path = os.path.join(output_folder, f"temp_{category}_{year}.png")
        plt.savefig(image_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # Resize the image to a standard size
        img = Image.open(image_path)
        img_resized = img.resize(standard_size)

        # Append the resized image to the list
        images.append(img_resized)

        # Optionally, delete the temporary image file after resizing
        os.remove(image_path)

    # Create the GIF
    gif_path = os.path.join(output_folder, f"{category}_evolution.gif")
    imageio.mimsave(gif_path, images, duration=1)  # 'duration' is the time between frames (in seconds)
    print(f'GIF creado para la categoría {category} y guardado en {gif_path}')

# Example usage
output_folder = './output'  # Define the output folder for the GIF
os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

# Assuming df_category is already filtered by category and contains the relevant data
# Call the function for each category
for category in df_long['Category'].unique():
    df_category = df_long[df_long['Category'] == category]
    create_gif(df_category, category, output_folder)


# In[122]:


import imageio.v2 as imageio
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from PIL import Image
import pandas as pd

# Function to create a GIF showing the evolution of crime categories
def create_category_gif(df_category, category, gdf, output_folder):
    images = []
    standard_size = (800, 800)  # Standard size for resizing images

    # Loop through each year in the dataset and generate the corresponding map
    for year in df_category['Year'].unique():
        print(f"Procesando año: {year}")

        # Filter data for the current year
        year_data = df_category[df_category['Year'] == year]

        # Merge the crime data with the shapefile data (gdf) for the provinces
        year_gdf = gdf.merge(year_data[['Province', 'CrimeCount']], left_on='ADM1_EN', right_on='Province', how='left')

        # Create the plot (map visualization of crime counts by province)
        fig, ax = plt.subplots(figsize=(10, 10))  # Set initial figure size
        year_gdf.plot(column='CrimeCount', ax=ax, legend=True,
                      legend_kwds={'label': "Crime Count by Province", 'orientation': "horizontal"})

        # Customize the plot
        ax.set_title(f"{category} - {year}", fontsize=15)
        ax.set_axis_off()  # Turn off the axis for cleaner maps

        # Save the figure temporarily
        image_path = os.path.join(output_folder, f"temp_{category}_{year}.png")
        plt.savefig(image_path, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # Resize the image to a standard size
        img = Image.open(image_path)
        img_resized = img.resize(standard_size)

        # Append the resized image to the list
        images.append(img_resized)

        # Optionally, delete the temporary image file after resizing
        os.remove(image_path)

    # Create the GIF for the current category
    gif_path = os.path.join(output_folder, f"{category}_evolution.gif")
    imageio.mimsave(gif_path, images, duration=5)  # 'duration' is the time between frames (in seconds)
    print(f'GIF creado para la categoría {category} y guardado en {gif_path}')

# Main code for generating GIFs by category
def generate_gifs_by_category(df_long, gdf, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all unique categories and create a GIF for each one
    for category in df_long['Category'].unique():
        print(f"Creando GIF para la categoría: {category}")
        # Filter the data for the current category
        df_category = df_long[df_long['Category'] == category]
        
        # Create the GIF for the current category
        create_category_gif(df_category, category, gdf, output_folder)

# Example usage:
output_folder = './output_gifs'  # Define the output folder for the GIFs

# Generate GIFs for all crime categories
generate_gifs_by_category(df_long, gdf, output_folder)


# In[119]:


gdf.head()


# In[148]:


# Convert CrimeCount to numeric, coercing non-numeric values to NaN
df_long['CrimeCount'] = pd.to_numeric(df_long['CrimeCount'], errors='coerce')

# Fill NaN values with 0 to ensure that the data is complete
df_long['CrimeCount'] = df_long['CrimeCount'].fillna(0)

# Filter data for a specific category, e.g., 'Burglary'
category_data = df_long[df_long['Category'] == 'Arson']

# Pivot the data, aggregating by summing the 'CrimeCount' for each province and year
category_pivot = category_data.pivot_table(index='Year', columns='Province', values='CrimeCount', aggfunc='sum')

# Plot the trends for each province
category_pivot.plot(kind='line', figsize=(12, 8), title="Arson Trends by Province")
plt.xlabel('Year')
plt.ylabel('Crime Count')
plt.grid(True)
plt.legend(title="Province")
plt.tight_layout()

# Save the plot to the output folder
output_folder = 'outputs'  # Ensure this path is correct
plt.savefig(os.path.join(output_folder, 'arson_trends_by_province.png'))
plt.close()


# In[ ]:




