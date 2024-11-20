
# Análisis Económico Simulado y Predictivo

Este proyecto tiene como objetivo generar un conjunto de datos económicos simulados, 
realizar análisis descriptivos y visualizar tendencias clave. Además, se incluye un 
modelo de regresión lineal para predecir el Producto Interno Bruto (PIB) utilizando 
indicadores económicos como inflación, desempleo, exportaciones e importaciones.

## Componentes Principales

1. **Generación de Datos Simulados**:
   - Los datos se generan utilizando la clase `GeneradorDatosEconomicos`.
   - Los parámetros económicos como crecimiento del PIB, inflación, desempleo y 
     comercio exterior son configurables a través de la clase `ParametrosEconomicos`.

2. **Análisis Descriptivo**:
   - Se genera un resumen estadístico básico de las variables económicas.
   - Se visualizan tendencias clave utilizando gráficos.

3. **Modelo Predictivo**:
   - Se implementa un modelo de regresión lineal con variables independientes como:
     - Inflación
     - Desempleo
     - Exportaciones
     - Importaciones
   - El modelo predice el PIB y evalúa su rendimiento utilizando métricas como el 
     Error Cuadrático Medio (MSE) y el Coeficiente de Determinación (R²).

## Cómo Ejecutar el Proyecto

### Requisitos Previos
- Python 3.7 o superior
- Bibliotecas requeridas: `pandas`, `numpy`, `matplotlib`, `seaborn`, 
  `scikit-learn`.

### Instrucciones
1. Abra el archivo `analisis_economico.ipynb` en Jupyter Notebook.
2. Ejecute las celdas en el orden en que aparecen.
3. Personalice los parámetros iniciales en la clase `ParametrosEconomicos` para 
   crear diferentes escenarios económicos.
4. Analice los resultados y visualizaciones generados.

## Archivos
- **`analisis_economico.ipynb`**: Archivo principal con el código ejecutable.
- **`README.txt`**: Archivo explicativo que describe el proyecto (este documento).

## Resultados Esperados
- Resúmenes estadísticos claros y gráficos de tendencias económicas.
- Predicciones precisas del PIB en función de los indicadores definidos.
- Código modular y reutilizable para análisis económico.

---
**Nota**: Este proyecto es una simulación y los resultados no deben interpretarse 
como representativos de datos económicos reales.
