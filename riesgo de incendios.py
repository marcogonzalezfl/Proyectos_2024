#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Instalación de paquetes necesarios
get_ipython().system('pip install matplotlib seaborn pandas numpy')

# Importaciones necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Clase principal para análisis de riesgo
class FireRiskAnalyzer:
    def __init__(self):
        """Inicializa el analizador de riesgo de incendios."""
        self.risk_categories = ['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
        self.risk_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.weights = {'NBR': 0.4, 'NDMI': 0.3, 'NDVI': 0.3}

    def generate_synthetic_data(self, num_points=5000):
        """Genera datos sintéticos para el análisis de riesgo de incendios."""
        np.random.seed(123)
        data = {
            'NBR': np.random.rand(num_points),
            'NDVI': np.random.rand(num_points),
            'NDMI': np.random.rand(num_points),
            'Longitude': np.random.uniform(-58, -57, num_points),
            'Latitude': np.random.uniform(-26.1, -24.9, num_points)
        }
        return pd.DataFrame(data)

    def calculate_risk_index(self, df):
        """Calcula el índice de riesgo de incendio (FRI) usando los índices espectrales."""
        nbr_component = self.weights['NBR'] * (-df['NBR'] + 1) / 2
        ndmi_component = self.weights['NDMI'] * (-df['NDMI'] + 1) / 2
        ndvi_component = self.weights['NDVI'] * (1 - df['NDVI']) / 2
        df['FRI'] = nbr_component + ndmi_component + ndvi_component
        df['FRI'] = df['FRI'].clip(0, 1)

        # Clasificar riesgo
        df['Riesgo_Categoria'] = pd.cut(df['FRI'], bins=self.risk_bins, labels=self.risk_categories, include_lowest=True)
        return df

    def plot_results(self, df):
        """Genera gráficos interactivos para la visualización del riesgo de incendios."""
        plt.figure(figsize=(15, 12))

        # 1. Distribución de categorías de riesgo
        plt.subplot(2, 2, 1)
        sns.countplot(x='Riesgo_Categoria', data=df, palette='RdYlGn')
        plt.title('Distribución de Categorías de Riesgo')

        # 2. Relación entre NDVI y NBR
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='NDVI', y='NBR', hue='FRI', palette='RdYlGn_r', data=df, alpha=0.7)
        plt.title('Relación entre NDVI y NBR')

        # 3. Histograma del FRI
        plt.subplot(2, 2, 3)
        sns.histplot(df['FRI'], bins=20, kde=True, color='skyblue')
        plt.title('Histograma del Índice de Riesgo de Incendio (FRI)')

        # 4. Distribución espacial del riesgo
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['FRI'], cmap='RdYlGn_r', s=20, alpha=0.7)
        plt.colorbar(scatter, label='Índice de Riesgo de Incendio (FRI)')
        plt.title('Distribución Espacial del Riesgo')

        plt.tight_layout()
        plt.show()

    def save_results(self, df, filename="resultados_riesgo_incendio.csv"):
        """Guarda los resultados del análisis en un archivo CSV."""
        df.to_csv(filename, index=False)
        print(f"Resultados guardados en: {filename}")


# Función principal para ejecución
def main():
    try:
        # Inicializar analizador
        analyzer = FireRiskAnalyzer()

        # Generar datos sintéticos de muestra
        df = analyzer.generate_synthetic_data()

        # Calcular el índice de riesgo y clasificarlo
        df = analyzer.calculate_risk_index(df)

        # Visualizar resultados
        analyzer.plot_results(df)

        # Guardar los resultados
        analyzer.save_results(df)

    except Exception as e:
        print(f"Error en la ejecución principal: {e}")

if __name__ == "__main__":
    main()


# In[3]:


get_ipython().system('pip install folium pillow geopandas')


# In[7]:


import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap

# Clase principal para análisis de riesgo
class FireRiskAnalyzer:
    def __init__(self):
        """Inicializa el analizador de riesgo de incendios."""
        self.risk_categories = ['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
        self.risk_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.weights = {'NBR': 0.4, 'NDMI': 0.3, 'NDVI': 0.3}

    def generate_synthetic_data(self, num_points=5000):
        """Genera datos sintéticos para el análisis de riesgo de incendios."""
        np.random.seed(123)
        data = {
            'NBR': np.random.rand(num_points),
            'NDVI': np.random.rand(num_points),
            'NDMI': np.random.rand(num_points),
            'Longitude': np.random.uniform(-58, -57, num_points),
            'Latitude': np.random.uniform(-26.1, -24.9, num_points)
        }
        return pd.DataFrame(data)

    def calculate_risk_index(self, df):
        """Calcula el índice de riesgo de incendio (FRI) usando los índices espectrales."""
        nbr_component = self.weights['NBR'] * (-df['NBR'] + 1) / 2
        ndmi_component = self.weights['NDMI'] * (-df['NDMI'] + 1) / 2
        ndvi_component = self.weights['NDVI'] * (1 - df['NDVI']) / 2
        df['FRI'] = nbr_component + ndmi_component + ndvi_component
        df['FRI'] = df['FRI'].clip(0, 1)

        # Clasificar riesgo
        df['Riesgo_Categoria'] = pd.cut(df['FRI'], bins=self.risk_bins, labels=self.risk_categories, include_lowest=True)
        return df

    def plot_on_map(self, df, map_center=(24.9, -57.5), zoom_start=7):
        """Genera un mapa interactivo con los valores de FRI sobre OpenStreetMap."""
        # Crear un mapa base utilizando Folium con OpenStreetMap
        m = folium.Map(location=map_center, zoom_start=zoom_start, control_scale=True)

        # Agregar los valores de FRI sobre el mapa como un HeatMap
        # Convertir las coordenadas de latitud y longitud a una lista
        heat_data = [[row['Latitude'], row['Longitude'], row['FRI']] for _, row in df.iterrows()]

        # Crear un HeatMap con los valores de FRI
        HeatMap(heat_data, min_opacity=0.2, max_val=1.0, radius=15, blur=10).add_to(m)

        # Guardar el mapa como archivo HTML
        m.save('mapa_interactivo_riesgo.html')

        print("Mapa interactivo guardado como 'mapa_interactivo_riesgo.html'. Puedes abrirlo en tu navegador.")

# Función principal para ejecución
def main():
    try:
        # Inicializar analizador
        analyzer = FireRiskAnalyzer()

        # Generar datos sintéticos de muestra
        df = analyzer.generate_synthetic_data()

        # Calcular el índice de riesgo y clasificarlo
        df = analyzer.calculate_risk_index(df)

        # Mostrar el mapa con los valores de FRI sobre OpenStreetMap
        analyzer.plot_on_map(df)

    except Exception as e:
        print(f"Error en la ejecución principal: {e}")

if __name__ == "__main__":
    main()



# In[ ]:




