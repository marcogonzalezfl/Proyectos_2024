Análisis de Indicadores de Pobreza Multidimensional (2016-2023)

Este código analiza datos de pobreza multidimensional en Colombia, dividido en 4 partes principales:

1. Procesamiento Inicial
Lee archivos CSV con datos de diferentes años (2016-2023)
Maneja diferentes formatos y codificaciones de archivo
Extrae el año del nombre del archivo
Guarda cada archivo procesado en la carpeta 'datos'

2. Consolidación de Datos
Une todos los archivos procesados en uno solo
Verifica que los datos sean consistentes entre años
Ordena los datos cronológicamente
Guarda el archivo consolidado para análisis posterior

3. Análisis Estadístico
Genera un reporte detallado que incluye:
Información general del dataset
Distribución de hogares por año
Estadísticas por cada indicador
Evolución temporal de los indicadores
Correlaciones importantes entre indicadores
Guarda el reporte en formato texto

4. Visualizaciones
Crea gráficos detallados que muestran:
Evolución de todos los indicadores en el tiempo
Distribución de indicadores por año
Matriz de correlación entre indicadores
Tendencias por grupos temáticos:
Salud
Vivienda
Servicios
Educación
Trabajo

Estructura de Carpetas
ipm/
├── raw/          # Datos originales
├── datos/        # Datos procesados
└── resultados/   # Reportes y visualizaciones

Este análisis permite entender la evolución de diferentes aspectos de la pobreza multidimensional y puede ser útil para la toma de decisiones en política pública.
