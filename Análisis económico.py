#!/usr/bin/env python
# coding: utf-8

# In[5]:


# --- Importación de bibliotecas ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuración global para gráficos
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# --- Clase para Definir Parámetros Económicos ---
class ParametrosEconomicos:
    """Clase para definir los parámetros económicos base."""
    def __init__(self):
        # Parámetros generales
        self.año_inicio = 2010
        self.año_fin = 2024
        
        # PIB
        self.pib_base = 10000
        self.crecimiento_pib_medio = 2.0
        self.volatilidad_pib = 0.2
        
        # Inflación
        self.inflacion_base = 3.0
        self.inflacion_min = 0.0
        self.inflacion_max = 8.0
        self.volatilidad_inflacion = 0.5
        
        # Desempleo
        self.desempleo_base = 7.0
        self.desempleo_min = 4.0
        self.desempleo_max = 12.0
        self.volatilidad_desempleo = 0.3
        
        # Comercio Exterior
        self.ratio_exportaciones_pib = 0.25
        self.ratio_importaciones_pib = 0.23
        self.volatilidad_comercio = 0.02
        
        # Inversión Extranjera
        self.ratio_inversion_pib = 0.05
        self.volatilidad_inversion = 0.01

# --- Clase para Generar Datos Económicos ---
class GeneradorDatosEconomicos:
    def __init__(self, parametros=None):
        """
        Inicializa el generador con parámetros personalizados.
        
        Parámetros:
        parametros (ParametrosEconomicos): Objeto con parámetros económicos.
        """
        self.params = parametros if parametros else ParametrosEconomicos()
        self.num_registros = self.params.año_fin - self.params.año_inicio + 1
        np.random.seed(42)  # Fijar semilla para reproducibilidad

    def generar_datos(self):
        """Genera un conjunto de datos económicos usando los parámetros definidos."""
        # Generar años
        años = pd.date_range(
            start=f'{self.params.año_inicio}-01-01',
            end=f'{self.params.año_fin}-12-31',
            freq='Y'
        ).year
        
        # PIB
        tendencia = np.linspace(0, 5, self.num_registros)
        ciclos = 0.5 * np.sin(np.linspace(0, 4 * np.pi, self.num_registros))
        ruido = np.random.normal(0, self.params.volatilidad_pib, self.num_registros)
        pib_crecimiento = self.params.crecimiento_pib_medio + tendencia + ciclos + ruido
        pib = self.params.pib_base * np.cumprod(1 + pib_crecimiento / 100)
        
        # Inflación
        inflacion_base = (self.params.inflacion_base +
                          0.3 * pib_crecimiento +
                          np.random.normal(0, self.params.volatilidad_inflacion, self.num_registros))
        inflacion = np.clip(inflacion_base, self.params.inflacion_min, self.params.inflacion_max)
        
        # Desempleo
        desempleo_base = (self.params.desempleo_base -
                          0.4 * pib_crecimiento +
                          np.random.normal(0, self.params.volatilidad_desempleo, self.num_registros))
        desempleo = np.clip(desempleo_base, self.params.desempleo_min, self.params.desempleo_max)
        
        # Comercio Exterior
        exportaciones = pib * (self.params.ratio_exportaciones_pib +
                               np.random.normal(0, self.params.volatilidad_comercio, self.num_registros))
        importaciones = pib * (self.params.ratio_importaciones_pib +
                               np.random.normal(0, self.params.volatilidad_comercio, self.num_registros))
        balanza_comercial = exportaciones - importaciones
        
        # Inversión Extranjera
        inversion_extranjera = (pib * self.params.ratio_inversion_pib +
                                balanza_comercial * 0.1 +
                                np.random.normal(0, pib * self.params.volatilidad_inversion, self.num_registros))
        
        # Crear DataFrame
        return pd.DataFrame({
            'año': años,
            'pib': pib.round(2),
            'crecimiento_pib': pib_crecimiento.round(2),
            'inflacion': inflacion.round(2),
            'desempleo': desempleo.round(2),
            'exportaciones': exportaciones.round(2),
            'importaciones': importaciones.round(2),
            'balanza_comercial': balanza_comercial.round(2),
            'inversion_extranjera': inversion_extranjera.round(2)
        })

# --- Clase para Análisis Económico ---
class AnalisisEconomico:
    def __init__(self, datos):
        self.datos = datos

    def resumen_ejecutivo(self):
        """Genera un resumen estadístico de los datos."""
        print("\nResumen descriptivo de los datos:")
        print(self.datos.describe())

    def visualizar_tendencias(self):
        """Visualiza las tendencias económicas a través de gráficos."""
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        sns.lineplot(x='año', y='pib', data=self.datos, marker="o")
        plt.title("Tendencia del PIB")
        plt.ylabel("PIB")

        plt.subplot(2, 2, 2)
        sns.lineplot(x='año', y='inflacion', data=self.datos, marker="o", color="red")
        plt.title("Tendencia de la Inflación")
        plt.ylabel("Inflación (%)")

        plt.subplot(2, 2, 3)
        sns.lineplot(x='año', y='desempleo', data=self.datos, marker="o", color="green")
        plt.title("Tendencia del Desempleo")
        plt.ylabel("Desempleo (%)")

        plt.subplot(2, 2, 4)
        sns.lineplot(x='año', y='balanza_comercial', data=self.datos, marker="o", color="purple")
        plt.title("Tendencia de la Balanza Comercial")
        plt.ylabel("Balanza Comercial")

        plt.tight_layout()
        plt.show()

    def analisis_predictivo(self):
        """Construye un modelo de regresión lineal para predecir el PIB."""
        # Variables predictoras y objetivo
        X = self.datos[['inflacion', 'desempleo', 'exportaciones', 'importaciones']]
        y = self.datos['pib']

        # Entrenamiento del modelo
        modelo = LinearRegression()
        modelo.fit(X, y)

        # Predicciones y métricas
        y_pred = modelo.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Resultados
        print("\n=== Análisis Predictivo ===")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.2f}")
        print("\nCoeficientes del modelo:")
        for var, coef in zip(X.columns, modelo.coef_):
            print(f"{var}: {coef:.4f}")

# --- Ejecución de Ejemplo ---
# Crear parámetros y generador de datos
params = ParametrosEconomicos()
generador = GeneradorDatosEconomicos(params)
datos = generador.generar_datos()

# Análisis del escenario
analisis = AnalisisEconomico(datos)
analisis.resumen_ejecutivo()
analisis.visualizar_tendencias()
analisis.analisis_predictivo()


# In[ ]:




