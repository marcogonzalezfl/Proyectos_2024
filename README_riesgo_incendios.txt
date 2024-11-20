
# Proyecto: Análisis de Riesgo de Incendios con Imágenes Satelitales

Este proyecto utiliza datos satelitales de Landsat 8 y la API de Google Earth Engine para analizar el riesgo de incendios en regiones específicas. 
El sistema calcula índices espectrales relevantes y clasifica áreas según su nivel de riesgo, generando gráficos y estadísticas útiles para estudios ambientales.

---

## Características del Proyecto
1. **Extracción de Datos Landsat**:
   - Obtención automática de imágenes satelitales Landsat 8 con baja cobertura de nubes en un rango de fechas definido.

2. **Cálculo de Índices Espectrales**:
   - Se calculan los siguientes índices:
     - **NBR**: Índice Normalizado de Quemaduras.
     - **NDVI**: Índice de Vegetación.
     - **NDMI**: Índice de Humedad.

3. **Análisis de Cobertura**:
   - Identificación de píxeles válidos (sin nubes ni sombras) en la región analizada.

4. **Muestreo y Clasificación de Riesgo**:
   - Clasificación de riesgo basada en el cálculo del índice FRI (Fire Risk Index) con categorías predefinidas.

5. **Visualización de Resultados**:
   - Generación de gráficos para la distribución del riesgo, relación de índices, y mapas espaciales del riesgo.

---

## Requisitos de Instalación

Antes de ejecutar este proyecto, asegúrate de instalar las siguientes dependencias:

- `earthengine-api`
- `geemap`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`

Instala los paquetes ejecutando:

```bash
pip install earthengine-api geemap numpy pandas matplotlib seaborn
```

---

## Uso del Proyecto

1. **Definir la Región de Interés**:
   - Las coordenadas de la región deben ser definidas en la función `main()`.

2. **Ejecutar el Código**:
   - Ejecuta el script `riesgo de incendios.py`. El sistema descargará las imágenes Landsat más recientes y procesará los datos.

3. **Resultados Generados**:
   - Archivo CSV: `resultados_riesgo_paraguay.csv`, que contiene los datos analizados.
   - Gráficos: `analisis_riesgo_incendio.png`, con visualizaciones de los resultados.

---

## Visualizaciones

El script genera los siguientes gráficos:
- **Distribución de Categorías de Riesgo**: Gráfico de barras.
- **Relación entre Índices Espectrales**: Gráfico de dispersión (NDVI vs NBR).
- **Histograma del Índice FRI**.
- **Distribución Espacial del Riesgo**: Mapa con las ubicaciones y niveles de riesgo.

---

## Notas Importantes
- **Configuración de Earth Engine**:
  - Antes de ejecutar, asegúrate de autenticarte en Google Earth Engine utilizando `ee.Authenticate()`.
- **Rendimiento**:
  - Algunas operaciones pueden ser intensivas en tiempo dependiendo del tamaño de la región analizada y la conexión a Earth Engine.

---

## Créditos
Este proyecto fue desarrollado para analizar datos geoespaciales y proporcionar herramientas de apoyo para la gestión ambiental y la prevención de incendios.

---
