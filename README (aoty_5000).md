# Análisis de Álbumes Musicales (AOTY Dataset)

## Descripción
Este script realiza un análisis completo de una base de datos de álbumes musicales, incluyendo calificaciones de usuarios, géneros y tendencias temporales.

## Estructura del Código

### 1. Importaciones
- pandas: Manipulación y análisis de datos
- seaborn y matplotlib: Visualización de datos
- numpy: Operaciones numéricas
- scipy.stats: Análisis estadístico

### 2. Funciones Auxiliares
- limpiar_rating_count(): Limpia y convierte el conteo de calificaciones a formato numérico
- add_trend_line(): Calcula líneas de tendencia para visualizaciones
- analizar_albums(): Función principal que realiza todo el análisis

### 3. Análisis Realizados

#### 3.1 Análisis Básico
- Total de álbumes
- Período temporal cubierto
- Estadísticas de calificaciones
- Métricas de valoraciones

#### 3.2 Análisis por Década
- Distribución de álbumes por década
- Calificaciones promedio
- Desviación estándar
- Promedio de valoraciones

#### 3.3 Visualización Popularidad vs Puntuación
- Gráfico de dispersión
- Escala logarítmica para valoraciones
- Línea de tendencia

#### 3.4 Análisis de Artistas Consistentes
- Identificación de artistas con 3+ álbumes
- Calificación promedio ≥ 85
- Estadísticas detalladas por artista

#### 3.5 Análisis de Géneros Emergentes
- Comparación pre/post 2000
- Identificación de nuevos géneros
- Lista de géneros emergentes

#### 3.6 Análisis de Tendencias por Género
- Correlaciones temporales
- Calificaciones promedio
- Tendencias de mejora/declive

#### 3.7 Top Álbumes
- Álbumes más valorados
- Álbumes mejor calificados
- Filtrado por número mínimo de valoraciones

#### 3.8 Visualización de Tendencias Temporales
- Evolución temporal de calificaciones
- Tamaño de puntos proporcional a popularidad
- Línea de tendencia general

#### 3.9 Datos Interesantes
- Géneros con mayor mejora/declive
- Estadísticas de nuevos géneros
- Artistas más consistentes
- Géneros mejor calificados

## Uso
1. Asegurarse de tener todas las dependencias instaladas
2. Tener el archivo 'aoty.csv' en el mismo directorio
3. Ejecutar el script
4. Los resultados se mostrarán en consola y las visualizaciones en ventanas separadas

## Requisitos
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Formato del CSV de entrada
El archivo 'aoty.csv' debe contener las siguientes columnas:
- title: Título del álbum
- artist: Nombre del artista
- user_score: Calificación de usuarios
- rating_count: Número de valoraciones
- release_date: Fecha de lanzamiento
- genres: Géneros musicales (separados por comas)

## Notas
- El análisis asume que las calificaciones están en escala de 0-100
- Se requiere un mínimo de 1000 valoraciones para algunos análisis
- Los géneros se procesan individualmente cuando están separados por comas
