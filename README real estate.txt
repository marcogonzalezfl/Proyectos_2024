Análisis Geoespacial de Propiedades Inmobiliarias
Este proyecto realiza un análisis geoespacial de un conjunto de datos sobre propiedades inmobiliarias. El análisis incluye la visualización de la ubicación de las propiedades en un mapa interactivo utilizando folium para generar un mapa con marcadores de las propiedades. También se presentan estadísticas descriptivas de las propiedades, como el precio promedio y la edad promedio de las viviendas.

Descripción del Proyecto
El conjunto de datos contiene información sobre propiedades inmobiliarias, incluidas las coordenadas geográficas, el precio por área de cada propiedad, la distancia a la estación de MRT más cercana, la edad de la casa y otros factores relacionados con las características de cada propiedad.

Ubicación Geográfica
El dataset está compuesto por propiedades ubicadas en una ciudad o región específica (la ubicación exacta no se menciona en los datos). Las coordenadas geográficas de las propiedades están disponibles a través de las columnas Latitude y Longitude, lo que permite la representación geoespacial de las propiedades en un mapa interactivo.

Dataset
El conjunto de datos proviene de un archivo CSV denominado Real_Estate.csv. Este archivo contiene las siguientes columnas:

Transaction date: Fecha de la transacción inmobiliaria.
House age: Edad de la casa en años.
Distance to the nearest MRT station: Distancia en metros a la estación de MRT más cercana.
Number of convenience stores: Número de tiendas de conveniencia cercanas.
Latitude: Latitud de la propiedad.
Longitude: Longitud de la propiedad.
House price of unit area: Precio por unidad de área de la propiedad.
Análisis Realizados
Distribución de las Propiedades en el Mapa:

Usamos folium para generar un mapa interactivo donde se muestran las ubicaciones de las propiedades en función de sus coordenadas geográficas (Latitude y Longitude).
Cada propiedad se representa con un marcador en el mapa, con un círculo azul. El mapa se centra automáticamente en las coordenadas promedio de todas las propiedades para ofrecer una visualización general.
Análisis Descriptivos:

Precio Promedio: Calculamos el precio promedio por unidad de área de todas las propiedades en el dataset.
Edad Promedio de la Casa: Se calculó la edad promedio de las viviendas.
Distancia Promedio a la Estación MRT: Calculamos la distancia promedio de todas las propiedades a la estación de MRT más cercana.
Número Promedio de Tiendas de Conveniencia Cercanas: Analizamos el número promedio de tiendas de conveniencia cercanas a las propiedades.
Visualización del Mapa:

El mapa interactivo generado utiliza el centro de la ciudad (promedio de latitudes y longitudes) como punto de inicio para el mapa.
Los marcadores para cada propiedad se colocan en el mapa utilizando círculos azules. Al pasar el mouse sobre cada marcador, se muestra un Tooltip con información adicional sobre la propiedad, como su precio por unidad de área.
Requisitos
El código requiere las siguientes bibliotecas:

folium para la visualización de mapas interactivos.
pandas para la manipulación de datos.
matplotlib para la creación de gráficos adicionales (si es necesario).
numpy para realizar cálculos numéricos.
geopy para cálculos geoespaciales (si se necesita la distancia entre puntos).