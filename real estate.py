#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[12]:


import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv('Real_Estate.csv')

# Revisar las primeras filas del dataframe
print(df.head())


# In[14]:


# Revisar el tipo de datos y verificar si las columnas de latitud y longitud existen
print(df.dtypes)

# Verificar las columnas del dataframe
print(df.columns)


# In[26]:


import folium

# Crear un mapa base centrado en las coordenadas promedio del dataframe
latitude_center = df['Latitude'].mean()
longitude_center = df['Longitude'].mean()

# Crear el mapa base
m = folium.Map(location=[latitude_center, longitude_center], zoom_start=12, 
               tiles='cartodb positron')  # Usando CartoDB con etiquetas en español


# Agregar los puntos de latitud y longitud al mapa con etiquetas en español
for index, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6,
        # Agregar un Tooltip con información en español
        tooltip=f"Ubicación: {row['Latitude']}, {row['Longitude']}<br>Precio: {row['House price of unit area']} $"
    ).add_to(m)

# Mostrar el mapa
m



# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import folium
from sklearn.preprocessing import StandardScaler

# Cargar el archivo CSV
df = pd.read_csv('Real_Estate.csv')

# Revisión del DataFrame
print(df.head())
print(df.dtypes)

# 1. Relación entre la distancia a la estación MRT y el precio de la casa
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Distance to the nearest MRT station', y='House price of unit area', data=df, color='teal')
plt.title('Relación entre Distancia a la Estación MRT y Precio de la Casa', fontsize=16)
plt.xlabel('Distancia a la Estación MRT (metros)', fontsize=12)
plt.ylabel('Precio por Unidad de Casa (en $)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 2. Distribución de precios por edad de la casa
plt.figure(figsize=(12, 6))
sns.boxplot(x='House age', y='House price of unit area', data=df, palette='Set2')
plt.title('Distribución de Precios por Edad de la Casa', fontsize=16)
plt.xlabel('Edad de la Casa (años)', fontsize=12)
plt.ylabel('Precio por Unidad de Casa (en $)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 3. Relación entre el número de tiendas de conveniencia y el precio de la casa
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Number of convenience stores', y='House price of unit area', data=df, color='orange')
plt.title('Relación entre Tiendas de Conveniencia y Precio de la Casa', fontsize=16)
plt.xlabel('Número de Tiendas de Conveniencia', fontsize=12)
plt.ylabel('Precio por Unidad de Casa (en $)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 4. Visualización de la distribución geográfica de las propiedades según el precio

# Crear el mapa base
latitude_center = df['Latitude'].mean()
longitude_center = df['Longitude'].mean()

m = folium.Map(location=[latitude_center, longitude_center], zoom_start=12)

# Agregar propiedades al mapa con colores basados en el precio
for index, row in df.iterrows():
    color = 'green' if row['House price of unit area'] < df['House price of unit area'].median() else 'red'
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6
    ).add_to(m)

# Mostrar el mapa
m

# 5. Análisis de Clustering geográfico

# Primero estandarizamos las características para mejorar el clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Latitude', 'Longitude', 'House price of unit area']])

# Realizar clustering con K-means (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualización de los clusters en un gráfico
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Latitude', y='Longitude', hue='Cluster', data=df, palette='viridis', s=100, edgecolor='black')
plt.title('Clustering Geográfico de las Propiedades', fontsize=16)
plt.xlabel('Latitud', fontsize=12)
plt.ylabel('Longitud', fontsize=12)
plt.legend(title='Cluster', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[ ]:




