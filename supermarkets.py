#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

# Cargar los datasets
df_stock = pd.read_csv('products_stock.csv')
df_superseis = pd.read_csv('products_superseis.csv')

# Agregar la columna 'supermarket' para identificar el origen de los datos
df_stock['supermarket'] = 'Stock'
df_superseis['supermarket'] = 'Superseis'

# Reemplazar los espacios y las comas en la columna de precios
df_stock['price'] = df_stock['price'].replace({r'\s+': '', ',': ''}, regex=True)
df_superseis['price'] = df_superseis['price'].replace({r'\s+': '', ',': ''}, regex=True)

# Definir una función para manejar precios múltiples (si es necesario)
def handle_multiple_prices(price):
    # Suponiendo que hay precios separados por una coma, tomamos el primer valor
    try:
        price_values = price.split('/')
        return float(price_values[0].strip())  # Devuelve el primer precio, puede adaptarse si hay un formato diferente
    except Exception as e:
        return None  # Si no puede procesarse, devuelve None

# Aplicar la función para manejar valores múltiples
df_stock['price'] = df_stock['price'].apply(handle_multiple_prices)
df_superseis['price'] = df_superseis['price'].apply(handle_multiple_prices)

# Concatenar los dos DataFrames usando 'name' y 'price'
df_final = pd.concat([df_stock[['name', 'price', 'category', 'supermarket']], 
                      df_superseis[['name', 'price', 'category', 'supermarket']]])

# Resetear el índice y mostrar el DataFrame final
df_final.reset_index(drop=True, inplace=True)
print(df_final)



# In[5]:


import pandas as pd

# Suponemos que 'df_final' ya está cargado con los datos combinados de ambos supermercados

# 1. Análisis del precio promedio por supermercado
precio_promedio = df_final.groupby('supermarket')['price'].mean()
print("Precio Promedio por Supermercado:")
print(precio_promedio)

# 2. Número de productos por supermercado
productos_por_supermercado = df_final.groupby('supermarket').size()
print("\nNúmero de Productos por Supermercado:")
print(productos_por_supermercado)

# 3. Distribución de categorías por supermercado
distribucion_categorias = df_final.groupby(['supermarket', 'category']).size().unstack(fill_value=0)
print("\nDistribución de Categorías por Supermercado:")
print(distribucion_categorias)

# 4. Comparación de precios máximos y mínimos por supermercado
precios_max_min = df_final.groupby('supermarket')['price'].agg(['max', 'min'])
print("\nPrecios Máximos y Mínimos por Supermercado:")
print(precios_max_min)

# 5. Análisis de productos con precios más altos y bajos
productos_precio_max = df_final.loc[df_final.groupby('supermarket')['price'].idxmax()][['name', 'supermarket', 'price']]
productos_precio_min = df_final.loc[df_final.groupby('supermarket')['price'].idxmin()][['name', 'supermarket', 'price']]

print("\nProducto con Precio Máximo por Supermercado:")
print(productos_precio_max)

print("\nProducto con Precio Mínimo por Supermercado:")
print(productos_precio_min)

# Si deseas graficar algunas de estas comparaciones, puedes hacerlo con matplotlib o seaborn

# Graficar el precio promedio por supermercado
import matplotlib.pyplot as plt

precio_promedio.plot(kind='bar', title="Precio Promedio por Supermercado", color=['blue', 'green'])
plt.ylabel('Precio Promedio')
plt.xlabel('Supermercado')
plt.xticks(rotation=0)
plt.show()

# Graficar la distribución de categorías por supermercado
distribucion_categorias.plot(kind='bar', stacked=True, title="Distribución de Categorías por Supermercado")
plt.ylabel('Cantidad de Productos')
plt.xlabel('Supermercado')
plt.show()


# In[ ]:




