# Análisis Comparativo de Precios entre Supermercados: Stock vs Superseis

Este proyecto realiza un análisis comparativo de precios de productos entre dos supermercados en Paraguay: **Stock** y **Superseis**. Utilizando datos proporcionados en archivos CSV, el análisis compara los precios, la distribución de categorías y otros aspectos clave para ayudar a obtener información sobre las diferencias entre ambos supermercados.

## Descripción

El código realiza las siguientes tareas:

1. Carga de Datos: Carga dos archivos CSV que contienen información sobre productos de los supermercados "Stock" y "Superseis".
2. Limpieza de Datos: Se eliminan los espacios y las comas en los precios y se ajustan para asegurarse de que sean numéricos.
3. Análisis Comparativo:
   - Precio promedio por supermercado.
   - Número de productos por supermercado.
   - Distribución de categorías por supermercado.
   - Precios máximos y mínimos por supermercado.
   - Productos con los precios más altos y más bajos.
4. Visualización: Se genera una visualización de las comparaciones de precios promedio por supermercado y la distribución de categorías por supermercado.

## Requisitos

Para ejecutar este proyecto, necesitas tener instaladas las siguientes bibliotecas:

- `pandas` para manipulación de datos.
- `matplotlib` para la visualización de gráficos.

Puedes instalar las dependencias necesarias ejecutando:

```bash
pip install pandas matplotlib
