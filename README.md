Código por secciones principales:

Configuración Inicial (Líneas 1-20)
- Importa las bibliotecas necesarias
Configura el sistema de logging
Inicializa Google Earth Engine

Clase LandsatAnalyzer (Líneas 22-308)
__init__: Inicializa el analizador
create_region_from_points: Crea un polígono para la región de interés
get_landsat_image: Obtiene la imagen Landsat más reciente
analyze_image_coverage: Analiza la cobertura de nubes
calculate_indices: Calcula índices NBR, NDVI y NDMI
sample_points: Toma muestras de puntos en la región
analyze_risk: Calcula el índice de riesgo de incendio (FRI)

Funciones de Visualización (Líneas 309-434)
- _plot_category_distribution: Gráfico de barras de categorías de riesgo
_plot_indices_scatter: Gráfico de dispersión NDVI vs NBR
_plot_fri_histogram: Histograma del FRI
_plot_spatial_distribution: Mapa de distribución espacial

Funciones de Logging (Líneas 435-448)
- Guarda información sobre los gráficos generados
Calcula y registra estadísticas de los índices

Función Principal (Líneas 450-496)
- Define la región de interés
Obtiene y procesa la imagen Landsat
Calcula índices y riesgo de incendio
Guarda resultados en CSV

Ejecución (Líneas 497-499)
- Punto de entrada del programa
Ejecuta el análisis y genera visualizaciones
El código está diseñado para:
Analizar imágenes satelitales Landsat
Calcular índices de vegetación y humedad
Evaluar riesgo de incendios
Generar visualizaciones y reportes
Manejar errores y logging detallado
