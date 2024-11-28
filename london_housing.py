#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar librerías necesarias
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Cargar el dataset
file_path = "kaggle_london_house_price_data.csv"  # Cambiar por la ruta del archivo
df = pd.read_csv(file_path)

# Mostrar las primeras filas para verificar la carga
print(df.head())

# Limpieza y transformación inicial
# 1. Eliminar columnas innecesarias (ejemplo: datos redundantes o no relevantes para análisis inicial)
columns_to_keep = [
    "fullAddress", "postcode", "outcode", "latitude", "longitude", 
    "bathrooms", "bedrooms", "floorAreaSqM", "livingRooms",
    "saleEstimate_currentPrice", "saleEstimate_upperPrice", 
    "saleEstimate_confidenceLevel", "saleEstimate_valueChange.numericChange",
    "saleEstimate_valueChange.percentageChange", "saleEstimate_valueChange.saleDate",
    "history_date", "history_price", "history_percentageChange", "history_numericChange"
]
df = df[columns_to_keep]

# 2. Manejo de valores faltantes (NaN)
# Puedes decidir rellenar, eliminar o dejar los valores faltantes según el análisis.
df = df.dropna(subset=["latitude", "longitude"])  # Asegurarse de tener coordenadas válidas
df.fillna({"bathrooms": 0, "bedrooms": 0, "livingRooms": 0, "floorAreaSqM": 0}, inplace=True)

# 3. Convertir las fechas a tipo datetime
df["saleEstimate_valueChange.saleDate"] = pd.to_datetime(df["saleEstimate_valueChange.saleDate"], errors="coerce")
df["history_date"] = pd.to_datetime(df["history_date"], errors="coerce")

# 4. Agregar una columna de geometría para análisis geoespacial
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # Usar WGS84

# Guardar como archivo GeoJSON o Shapefile para análisis geográfico posterior
gdf.to_file("london_properties.geojson", driver="GeoJSON")

# Análisis inicial de datos
# Resumen estadístico
print(df.describe())

# Análisis espacial: distribución de propiedades por zona
property_distribution = df["outcode"].value_counts()
print(property_distribution)

# Análisis económico: promedio de precios actuales y históricos
avg_current_price = df["saleEstimate_currentPrice"].mean()
avg_history_price = df["history_price"].mean()

print(f"Precio promedio actual de venta: {avg_current_price:.2f}")
print(f"Precio promedio histórico de venta: {avg_history_price:.2f}")

# Visualización geográfica básica (opcional)
import matplotlib.pyplot as plt

# Plot rápido de las propiedades en un mapa
gdf.plot(figsize=(10, 10), color="blue", alpha=0.5, markersize=5)
plt.title("Distribución Geográfica de Propiedades en Londres")
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.show()


# In[2]:


# Agrupación por 'outcode' y análisis de precios
grouped_by_outcode = df.groupby("outcode").agg({
    "saleEstimate_currentPrice": ["mean", "median", "std", "count"],
    "saleEstimate_valueChange.percentageChange": "mean"
}).reset_index()
grouped_by_outcode.columns = ["outcode", "mean_price", "median_price", "std_dev_price", "property_count", "avg_percentage_change"]

print(grouped_by_outcode)

# Unión con coordenadas geográficas promedio por 'outcode' para mapeo
centroids = df.groupby("outcode")[["latitude", "longitude"]].mean().reset_index()
geo_grouped = pd.merge(grouped_by_outcode, centroids, on="outcode", how="left")

# Crear un GeoDataFrame para visualización
geo_grouped["geometry"] = [Point(xy) for xy in zip(geo_grouped["longitude"], geo_grouped["latitude"])]
geo_gdf = gpd.GeoDataFrame(geo_grouped, geometry="geometry", crs="EPSG:4326")

# Visualización del mapa de calor (ejemplo básico)
geo_gdf.plot(
    column="mean_price", 
    cmap="coolwarm", 
    legend=True, 
    figsize=(20, 20), 
    markersize=geo_gdf["property_count"]*0.5
)
plt.title("Distribución de Precios Promedio por Outcode")
plt.show()


# In[3]:


# Conversión de fechas y creación de columnas adicionales para análisis temporal
df["history_year"] = df["history_date"].dt.year
df["history_month"] = df["history_date"].dt.to_period("M")
df["sale_year"] = df["saleEstimate_valueChange.saleDate"].dt.year

# Agrupación anual de precios históricos y estimados
price_trends = df.groupby("history_year").agg({
    "history_price": "mean",
    "saleEstimate_currentPrice": "mean"
}).reset_index()

price_trends.columns = ["year", "avg_historical_price", "avg_estimated_price"]

# Visualización de tendencias temporales
plt.figure(figsize=(12, 6))
plt.plot(price_trends["year"], price_trends["avg_historical_price"], label="Precio Histórico Promedio", marker="o")
plt.plot(price_trends["year"], price_trends["avg_estimated_price"], label="Precio Estimado Promedio", marker="o")
plt.title("Evolución Temporal de Precios")
plt.xlabel("Año")
plt.ylabel("Precio Promedio (£)")
plt.legend()
plt.grid()
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Filtrar filas donde 'saleEstimate_currentPrice' sea NaN
df_clean = df.dropna(subset=["saleEstimate_currentPrice"])

# Selección de variables relevantes
features = ["bathrooms", "bedrooms", "floorAreaSqM", "latitude", "longitude", "history_price"]
X = df_clean[features].fillna(0)  # Rellenar valores faltantes en las características
y = df_clean["saleEstimate_currentPrice"]

# Normalización de características continuas (opcional, mejora modelos sensibles a escalas)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[5]:


# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el desempeño del modelo
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")


# In[6]:


import matplotlib.pyplot as plt

# Comparar valores reales con predicciones
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label="Predicciones")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2, label="Línea perfecta")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Comparación: Valores Reales vs Predicciones")
plt.legend()
plt.grid()
plt.show()


# In[7]:


# Importancia de características
feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

# Gráfico de barras
plt.figure(figsize=(8, 6))
plt.barh(feature_importance["feature"], feature_importance["importance"], color="skyblue")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.title("Importancia de las Características en el Modelo")
plt.gca().invert_yaxis()  # Invertir el eje Y para mostrar la más importante arriba
plt.show()


# In[8]:


import geopandas as gpd
import folium
from folium.plugins import HeatMap

# Crear un GeoDataFrame para las propiedades
gdf = gpd.GeoDataFrame(
    df_clean,
    geometry=gpd.points_from_xy(df_clean['longitude'], df_clean['latitude']),
    crs="EPSG:4326"
)

# Crear un mapa base con Folium
m = folium.Map(location=[51.509865, -0.118092], zoom_start=10)

# Añadir el mapa de calor
heat_data = [[row['latitude'], row['longitude']] for index, row in df_clean.iterrows()]
HeatMap(heat_data).add_to(m)

# Mostrar el mapa
m


# In[9]:


# Calcular el precio promedio por outcode
avg_prices = df_clean.groupby('outcode')['saleEstimate_currentPrice'].mean().sort_values()

# Graficar un mapa de precios con ajustes
fig, ax = plt.subplots(figsize=(26, 10))  # Aumentar el tamaño del gráfico
avg_prices.plot(kind="bar", color="teal", alpha=0.7, ax=ax)

# Configurar el título y etiquetas
plt.title("Precio Promedio por Outcode", fontsize=16)
plt.xlabel("Outcode", fontsize=5)
plt.ylabel("Precio Promedio (£)", fontsize=14)

# Ajustar las etiquetas del eje X
plt.xticks(rotation=90, fontsize=10)  # Reducir el tamaño de la fuente de las etiquetas

# Mostrar el gráfico
plt.tight_layout()  # Ajustar el diseño para evitar solapamiento
plt.show()



# In[10]:


# Descripción estadística
desc_stats = df_clean[["bathrooms", "bedrooms", "floorAreaSqM", "saleEstimate_currentPrice"]].describe()
print(desc_stats)

# Histogramas de características clave
df_clean[["bathrooms", "bedrooms", "floorAreaSqM", "saleEstimate_currentPrice"]].hist(bins=20, figsize=(12, 8), color="skyblue", edgecolor="black")
plt.suptitle("Distribuciones de las Características")
plt.show()


# In[12]:


import folium
from folium.plugins import MarkerCluster

# Filtrar datos inválidos antes de trabajar con ellos
df_clean = df_clean[df_clean['saleEstimate_currentPrice'].notnull()]  # Eliminar valores nulos en los precios
df_clean = df_clean[df_clean['floorAreaSqM'] > 0]  # Eliminar propiedades con área inválida

# Calcular el precio unitario (por m²)
df_clean['price_per_sqm'] = df_clean['saleEstimate_currentPrice'] / df_clean['floorAreaSqM']

# Filtrar outliers extremos en el precio unitario (opcional)
df_clean = df_clean[df_clean['price_per_sqm'] < df_clean['price_per_sqm'].quantile(0.99)]

# Crear un mapa base centrado en Londres
price_map = folium.Map(location=[51.509865, -0.118092], zoom_start=11)

# Crear un agrupador de marcadores (para mejorar la visualización con muchas propiedades)
marker_cluster = MarkerCluster().add_to(price_map)

# Añadir marcadores con información al mapa
for _, row in df_clean.iterrows():
    popup_info = f"""
    <strong>Dirección:</strong> {row['fullAddress']}<br>
    <strong>Precio Unitario:</strong> £{row['price_per_sqm']:.2f} / m²<br>
    <strong>Superficie:</strong> {row['floorAreaSqM']} m²<br>
    """
    
    # Añadir un marcador con círculo
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=5,  # Radio del círculo
        color='blue',  # Color del borde
        fill=True,
        fill_color='orange',  # Color de relleno
        fill_opacity=0.7,  # Opacidad del relleno
        popup=folium.Popup(popup_info, max_width=300)  # Ventana emergente con la información
    ).add_to(marker_cluster)

# Mostrar el mapa
price_map


# In[26]:


df.head()


# In[29]:


# Asegurarse de que la columna history_date esté en formato datetime
df['history_date'] = pd.to_datetime(df['history_date'], errors='coerce')

# Encontrar la fecha más antigua
oldest_date = df['history_date'].min()

print(f"La fecha más antigua en 'history_date' es: {oldest_date}")


# In[41]:


import pandas as pd
import matplotlib.pyplot as plt

# Asegurarse de que history_date esté en formato datetime
df['history_date'] = pd.to_datetime(df['history_date'], errors='coerce')

# Verificar valores válidos
print(f"Total de registros originales: {len(df)}")
print(f"Registros con precios nulos: {df['history_price'].isnull().sum()}")
print(f"Registros con área no válida (<= 0): {(df['floorAreaSqM'] <= 0).sum()}")

# Filtrar valores no válidos
df = df[df['history_price'].notnull()]
df = df[df['floorAreaSqM'] > 0]

# Calcular precio por m² histórico
df['price_per_sqm_historic'] = df['history_price'] / df['floorAreaSqM']

# Verificar datos después del cálculo
print(f"Registros después del cálculo: {len(df)}")
print(f"Registros con price_per_sqm_historic nulo: {df['price_per_sqm_historic'].isnull().sum()}")

# Filtrar outliers opcionales (percentil 99)
threshold = df['price_per_sqm_historic'].quantile(0.99)
print(f"Umbral para el percentil 99: £{threshold:.2f}")
df = df[df['price_per_sqm_historic'] < threshold]

# Extraer el año de history_date
df['year'] = df['history_date'].dt.year

# Calcular el precio promedio por m² por año
avg_price_per_sqm_by_year = df.groupby('year')['price_per_sqm_historic'].mean()

# Verificar datos agregados
print(avg_price_per_sqm_by_year)

# Graficar la evolución
plt.figure(figsize=(12, 6))
avg_price_per_sqm_by_year.plot(kind='line', marker='o', color='teal')
plt.title("Evolución del precio promedio por m² (1995-presente)", fontsize=14)
plt.xlabel("Año", fontsize=12)
plt.ylabel("Precio promedio por m² (£)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[45]:


import folium
from folium.plugins import MarkerCluster
import geopandas as gpd

# Asegurarse de que no haya valores nulos ni áreas inválidas
df_clean = df_clean[df_clean['history_price'].notnull()]  # Eliminar valores nulos en precios históricos
df_clean = df_clean[df_clean['floorAreaSqM'] > 0]  # Filtrar áreas inválidas

# Calcular precios históricos por m²
df_clean['price_per_sqm_historic'] = df_clean['history_price'] / df_clean['floorAreaSqM']

# Verificar y continuar con el análisis geográfico
import folium
from folium.plugins import MarkerCluster

# Extraer el año de la fecha histórica
df_clean['year'] = df_clean['history_date'].dt.year

# Agrupar por 'outcode' y año
geo_avg_price = (
    df_clean.groupby(['outcode', 'year'])['price_per_sqm_historic']
    .mean()
    .reset_index()
)

# Calcular coordenadas promedio por 'outcode'
geo_coords = (
    df_clean.groupby('outcode')[['latitude', 'longitude']]
    .mean()
    .reset_index()
)

# Combinar precios promedio con coordenadas
geo_avg_price = geo_avg_price.merge(geo_coords, on='outcode', how='left')

# Crear el mapa base
price_map = folium.Map(location=[51.509865, -0.118092], zoom_start=10)

# Crear un agrupador de marcadores
marker_cluster = MarkerCluster().add_to(price_map)

# Añadir marcadores al mapa
for _, row in geo_avg_price.iterrows():
    popup_info = f"""
    <strong>Área (Outcode):</strong> {row['outcode']}<br>
    <strong>Año:</strong> {row['year']}<br>
    <strong>Precio Promedio:</strong> £{row['price_per_sqm_historic']:.2f} / m²<br>
    """
    folium.CircleMarker(
        location=(row['latitude'], row['longitude']),
        radius=5,
        color='blue',
        fill=True,
        fill_color='red',
        fill_opacity=0.6,
        popup=folium.Popup(popup_info, max_width=300)
    ).add_to(marker_cluster)

# Mostrar el mapa
price_map



# Exportar resultados mas relevantes en formato png

# In[62]:


import geopandas as gpd
from shapely.geometry import Point

# Ensure there is data to process
df_clean = df_clean.dropna(subset=['latitude', 'longitude'])  # Drop rows with missing coordinates

# Create the geometry from latitude and longitude
df_clean['geometry'] = df_clean.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)

# Create the GeoDataFrame
gdf = gpd.GeoDataFrame(df_clean, geometry='geometry')

# Check if the GeoDataFrame is now valid
print(gdf.info())  # Ensure there is data after cleaning

if gdf.empty:
    print("GeoDataFrame is empty after cleaning.")
else:
    # Proceed with plotting
    gdf.plot(column="price_per_sqm_historic", cmap="coolwarm", legend=True, figsize=(10, 10))
    plt.title("Mapa de Evolución de Precios por m²")
    plt.savefig(os.path.join(output_dir, "geographical_price_evolution.png"))
    plt.close()


# In[76]:


import matplotlib.pyplot as plt
import geopandas as gpd
import os
import imageio

# Ensure there are no null values in 'price_per_sqm_historic'
gdf = gdf[gdf['price_per_sqm_historic'].notnull()]

# Output folder for images
output_dir = "housing_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# --- 1. Geographical Distribution of Current Prices (Zoom In) ---
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Plot the map (not using scatter but plot directly)
gdf.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
         c="saleEstimate_currentPrice", cmap="viridis", colorbar=True, ax=ax)

# Dynamically adjust the limits to fit the entire data extent
minx, miny, maxx, maxy = gdf.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Title and labels in English
ax.set_title("Geographical Distribution of Current Prices", fontsize=16)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
plt.colorbar(label="Current Price (£)")  # Colorbar label
plt.savefig(os.path.join(output_dir, "current_price_distribution_zoomed.png"))
plt.close()

# --- 2. Evolution of Price per m² (Historic) by Year (GIF Creation) ---
years = sorted(gdf['history_year'].unique())  # Get unique years for animation
frames = []  # List to store image frames

for year in years:
    # Filter the data for the current year
    year_gdf = gdf[gdf['history_year'] == year]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the data for the current year
    year_gdf.plot(column="price_per_sqm_historic", cmap="coolwarm", legend=True, ax=ax, 
                  legend_kwds={'label': "Price per m² (Historic)", 'orientation': "horizontal"},
                  alpha=0.5, edgecolor="k", linewidth=0.5)

    # Title and labels
    ax.set_title(f"Evolution of Historic Property Prices per Square Meter ({year})", fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Customize ticks and grid
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Save the figure
    frame_path = os.path.join(output_dir, f"year_{year}.png")
    plt.savefig(frame_path)
    frames.append(imageio.imread(frame_path))  # Add the frame to the GIF list
    plt.close()

# Create the GIF from the frames
gif_path = os.path.join(output_dir, "evolution_price_per_sqm.gif")
imageio.mimsave(gif_path, frames, duration=1)  # Save as GIF with 1 second per frame
print(f"GIF created and saved to: {gif_path}")

# --- 3. Optional: Checking the result ---
Optionally, you can preview the GIF (this part can be commented out if not needed)
from IPython.display import Image
Image(filename=gif_path)  # This will display the GIF inline in Jupyter


# In[81]:


import matplotlib.pyplot as plt
import geopandas as gpd
import os
from matplotlib import cm
from matplotlib.colors import Normalize
import numpy as np

# Ensure there are no null values in 'saleEstimate_currentPrice'
gdf = gdf[gdf['saleEstimate_currentPrice'].notnull()]

# Output folder for images
output_dir = "housing_analysis_results"
os.makedirs(output_dir, exist_ok=True)

# --- 1. Geographical Distribution of Current Prices (Zoom In) ---
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Plot the data (scatter plot for geographic distribution)
scatter = gdf.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 
                   c="saleEstimate_currentPrice", cmap="viridis", ax=ax)

# Dynamically adjust the limits to fit the entire data extent
minx, miny, maxx, maxy = gdf.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Title and labels in English
ax.set_title("Geographical Distribution of Current Prices", fontsize=16)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)

# Create a colorbar for the scatter plot (using ScalarMappable)
norm = Normalize(vmin=gdf['saleEstimate_currentPrice'].min(), vmax=gdf['saleEstimate_currentPrice'].max())
sm = cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])  # Required for colorbar to work

# Add the colorbar to the plot
plt.colorbar(sm, ax=ax, label="Current Price (£)")  # Colorbar label

# Save the plot
plt.savefig(os.path.join(output_dir, "current_price_distribution_zoomed.png"))
plt.close()


# In[ ]:




