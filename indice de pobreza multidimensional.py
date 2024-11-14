#!/usr/bin/env python
# coding: utf-8

# Parte 1: procesamiento inicial

# In[1]:


# Instalar dependencias
get_ipython().system('pip install pandas numpy tqdm openpyxl')


# In[59]:


import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import re

warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = Path('ipm')
RAW_DIR = BASE_DIR / 'raw'
DATOS_DIR = BASE_DIR / 'datos'

def crear_directorios():
    """Crea los directorios necesarios si no existen"""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATOS_DIR.mkdir(parents=True, exist_ok=True)

def obtener_año(nombre_archivo):
    """Extrae el año del nombre del archivo usando regex"""
    try:
        match = re.search(r'[Mm][Pp][Ii]20(\d{2})', nombre_archivo)
        return int('20' + match.group(1)) if match else None
    except:
        return None

def leer_archivo(ruta):
    """Lee el archivo CSV y retorna un DataFrame con las columnas necesarias"""
    encodings = ['utf-8', 'latin1', 'iso-8859-1']
    separators = [';', ',']
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(ruta, sep=sep, encoding=encoding)
                columnas_base = [col for col in df.columns if col.startswith('hh_d_')]
                
                if columnas_base:
                    return df[columnas_base]
                
            except Exception:
                continue
    
    print(f"Error: No se pudo leer el archivo {ruta}")
    return None

def procesar_archivo(archivo, año):
    """Procesa un archivo individual y añade la columna año"""
    print(f"\nProcesando: {archivo.name}")
    
    df = leer_archivo(archivo)
    if df is None:
        return None
    
    # Añadir columna año
    df['año'] = año
    
    # Guardar archivo procesado
    archivo_salida = DATOS_DIR / f'ipm_{año}_procesado.csv'
    df.to_csv(archivo_salida, index=False)
    
    print(f"Registros: {len(df):,}")
    print(f"Variables: {len(df.columns):,}")
    
    return df

def main():
    """Función principal"""
    crear_directorios()
    
    archivos = list(RAW_DIR.glob('*Mpi20*.csv'))
    print(f"\nArchivos encontrados: {len(archivos)}")
    
    procesados = []
    errores = []
    
    for archivo in archivos:
        try:
            año = obtener_año(archivo.name)
            if año:
                df = procesar_archivo(archivo, año)
                if df is not None:
                    procesados.append((archivo, df))
            else:
                errores.append((archivo, "No se pudo determinar el año"))
        except Exception as e:
            errores.append((archivo, str(e)))
    
    print(f"\nProcesados exitosamente: {len(procesados)}")
    if errores:
        print("\nErrores encontrados:")
        for archivo, error in errores:
            print(f"- {archivo.name}: {error}")

if __name__ == "__main__":
    main()


# Parte 2: consolidacion de datos

# In[61]:


import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = Path('ipm')
DATOS_DIR = BASE_DIR / 'datos'

def cargar_archivos_procesados():
    """Carga y unifica todos los archivos procesados"""
    print("\nCargando archivos procesados...")
    
    dfs = []
    archivos = list(DATOS_DIR.glob('ipm_*_procesado.csv'))
    archivos.sort()
    
    total_registros = 0
    for archivo in archivos:
        try:
            año = int(archivo.stem.split('_')[1])
            if 2015 < año < 2024:  # Validación de años
                df = pd.read_csv(archivo)
                registros = len(df)
                total_registros += registros
                print(f"Año {año}: {registros:,} registros")
                dfs.append(df)
            else:
                print(f"Advertencia: Año fuera de rango - {archivo.name}")
                
        except Exception as e:
            print(f"Error al procesar {archivo.name}: {e}")
            continue
    
    print(f"\nTotal de registros consolidados: {total_registros:,}")
    return dfs

def verificar_consistencia(dfs):
    """Verifica la consistencia de las columnas entre DataFrames"""
    if not dfs:
        return False
        
    columnas_base = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], 2):
        if set(df.columns) != columnas_base:
            print(f"Error: Inconsistencia en columnas del año {df['año'].iloc[0]}")
            return False
    return True

def consolidar_datos():
    """Función principal de consolidación"""
    print("\nIniciando consolidación de datos...")
    
    # Cargar archivos procesados
    dfs = cargar_archivos_procesados()
    if not dfs:
        print("Error: No se encontraron archivos para procesar")
        return
    
    # Verificar consistencia
    if not verificar_consistencia(dfs):
        print("Error: Los archivos no son consistentes")
        return
    
    # Consolidar datos
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Ordenar por año
    df_final = df_final.sort_values('año')
    
    # Guardar archivo consolidado
    archivo_consolidado = DATOS_DIR / 'ipm_consolidado.csv'
    df_final.to_csv(archivo_consolidado, index=False)
    
    print("\nResumen de consolidación:")
    print(f"Total años: {df_final['año'].nunique()}")
    print(f"Total variables: {len(df_final.columns)-1}")
    print(f"Total registros: {len(df_final):,}")
    print(f"\nArchivo consolidado guardado en: {archivo_consolidado}")
    
    return df_final

if __name__ == "__main__":
    consolidar_datos()


# Parte 3: analisis estadistico

# In[65]:


import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = Path('ipm')
DATOS_DIR = BASE_DIR / 'datos'
RESULTADOS_DIR = BASE_DIR / 'resultados'
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

# Diccionario de nombres de variables
NOMBRES_VARIABLES = {
    'hh_d_no_afil': 'Sin Afiliación a Salud',
    'hh_d_sin_basur': 'Sin Servicio de Basura',
    'hh_d_combus': 'Uso Inadecuado de Combustible',
    'hh_d_san_mejor': 'Saneamiento Inadecuado',
    'hh_d_materialidad': 'Materiales Inadecuados Vivienda',
    'hh_d_hacinamiento': 'Hacinamiento',
    'hh_d_agua_mejor': 'Acceso Inadecuado a Agua',
    'hh_d_sin_salud': 'Sin Acceso a Salud',
    'hh_d_logro_min': 'Bajo Logro Educativo',
    'hh_d_esc_retardada': 'Rezago Escolar',
    'hh_d_destotalmax': 'Desempleo de Larga Duración',
    'hh_d_jubi_pens': 'Sin Jubilación ni Pensión',
    'hh_d_subocup_max': 'Subocupación',
    'hh_d_10a17_ocup': 'Trabajo Infantil',
    'hh_d_ni_noasis': 'Inasistencia Escolar'
}

def generar_reporte_estadistico():
    """Genera un reporte estadístico completo"""
    # Cargar datos
    df = pd.read_csv(DATOS_DIR / 'ipm_consolidado.csv')
    
    # Convertir columnas a numéricas (excepto año)
    for col in df.columns:
        if col != 'año':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Crear archivo de reporte
    with open(RESULTADOS_DIR / 'reporte_estadistico.txt', 'w', encoding='utf-8') as f:
        # 1. Información General
        f.write("="*80 + "\n")
        f.write("ANÁLISIS DE INDICADORES DE POBREZA MULTIDIMENSIONAL 2016-2023\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total de hogares analizados: {len(df):,}\n")
        f.write(f"Período analizado: 2016-2023\n")
        f.write(f"Variables analizadas: {len(df.columns)-1}\n\n")
        
        # 2. Distribución por año
        f.write("-"*50 + "\n")
        f.write("DISTRIBUCIÓN DE HOGARES POR AÑO\n")
        f.write("-"*50 + "\n")
        dist_año = df['año'].value_counts().sort_index()
        for año, count in dist_año.items():
            f.write(f"{año}: {count:,} hogares\n")
        f.write("\n")
        
        # 3. Estadísticas por variable
        f.write("-"*50 + "\n")
        f.write("ESTADÍSTICAS POR INDICADOR\n")
        f.write("-"*50 + "\n")
        
        columnas_numericas = [col for col in df.columns if col != 'año']
        for col in columnas_numericas:
            nombre = NOMBRES_VARIABLES.get(col, col)
            promedio = df[col].mean()
            maximo = df[col].max()
            minimo = df[col].min()
            
            f.write(f"\n{nombre}:\n")
            f.write(f"  Promedio: {promedio*100:.1f}%\n")
            f.write(f"  Máximo: {maximo*100:.1f}%\n")
            f.write(f"  Mínimo: {minimo*100:.1f}%\n")
        
        # 4. Evolución temporal
        f.write("\n" + "-"*50 + "\n")
        f.write("EVOLUCIÓN TEMPORAL\n")
        f.write("-"*50 + "\n")
        
        medias_anuales = df.groupby('año')[columnas_numericas].mean()
        for col in columnas_numericas:
            nombre = NOMBRES_VARIABLES.get(col, col)
            primer_año = medias_anuales.iloc[0][col]
            ultimo_año = medias_anuales.iloc[-1][col]
            cambio = ((ultimo_año - primer_año) / primer_año * 100)
            
            if abs(cambio) > 20:  # Solo cambios significativos
                f.write(f"\n{nombre}:\n")
                f.write(f"  2016: {primer_año*100:.1f}%\n")
                f.write(f"  2023: {ultimo_año*100:.1f}%\n")
                f.write(f"  Cambio: {cambio:+.1f}%\n")
        
        # 5. Correlaciones importantes
        f.write("\n" + "-"*50 + "\n")
        f.write("RELACIONES ENTRE INDICADORES\n")
        f.write("-"*50 + "\n")
        
        corr_matrix = df[columnas_numericas].corr()
        correlaciones = []
        for i in range(len(columnas_numericas)):
            for j in range(i+1, len(columnas_numericas)):
                var1 = columnas_numericas[i]
                var2 = columnas_numericas[j]
                corr = corr_matrix.loc[var1, var2]
                if abs(corr) > 0.3:
                    correlaciones.append((var1, var2, corr))
        
        correlaciones.sort(key=lambda x: abs(x[2]), reverse=True)
        for var1, var2, corr in correlaciones[:5]:
            f.write(f"\n{NOMBRES_VARIABLES[var1]} y {NOMBRES_VARIABLES[var2]}:\n")
            f.write(f"  Nivel de relación: {abs(corr)*100:.1f}%\n")
    
    print(f"\nReporte estadístico guardado en: {RESULTADOS_DIR}/reporte_estadistico.txt")
    return df

if __name__ == "__main__":
    generar_reporte_estadistico()


# Parte 4: visualizaciones

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = Path('ipm')
DATOS_DIR = BASE_DIR / 'datos'
RESULTADOS_DIR = BASE_DIR / 'resultados'
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

# Diccionario de nombres de variables
NOMBRES_VARIABLES = {
    'hh_d_no_afil': 'Sin Afiliación a Salud',
    'hh_d_sin_basur': 'Sin Servicio de Basura',
    'hh_d_combus': 'Uso Inadecuado de Combustible',
    'hh_d_san_mejor': 'Saneamiento Inadecuado',
    'hh_d_materialidad': 'Materiales Inadecuados Vivienda',
    'hh_d_hacinamiento': 'Hacinamiento',
    'hh_d_agua_mejor': 'Acceso Inadecuado a Agua',
    'hh_d_sin_salud': 'Sin Acceso a Salud',
    'hh_d_logro_min': 'Bajo Logro Educativo',
    'hh_d_esc_retardada': 'Rezago Escolar',
    'hh_d_destotalmax': 'Desempleo de Larga Duración',
    'hh_d_jubi_pens': 'Sin Jubilación ni Pensión',
    'hh_d_subocup_max': 'Subocupación',
    'hh_d_10a17_ocup': 'Trabajo Infantil',
    'hh_d_ni_noasis': 'Inasistencia Escolar'
}

def configurar_estilo():
    """Configura el estilo general de las visualizaciones"""
    plt.style.use('default')
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def crear_visualizaciones_completas(df):
    """Genera visualizaciones exhaustivas de todos los indicadores"""
    configurar_estilo()
    
    # 1. Evolución de todos los indicadores
    plt.figure(figsize=(20, 12))
    columnas = [col for col in df.columns if col != 'año']
    colores = plt.cm.tab20(np.linspace(0, 1, len(columnas)))
    
    for i, col in enumerate(columnas):
        datos = df.groupby('año')[col].mean()
        plt.plot(datos.index, datos.values * 100, marker='o',
                linewidth=2, label=NOMBRES_VARIABLES[col], color=colores[i])
    
    plt.title('Evolución de Todos los Indicadores (2016-2023)')
    plt.xlabel('Año')
    plt.ylabel('Porcentaje de Hogares Afectados')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / 'evolucion_completa.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Boxplots de distribución por año
    plt.figure(figsize=(20, 12))
    datos_largo = df.melt(id_vars=['año'], 
                         value_vars=[col for col in df.columns if col != 'año'],
                         var_name='Indicador',
                         value_name='Valor')
    datos_largo['Indicador'] = datos_largo['Indicador'].map(NOMBRES_VARIABLES)
    
    sns.boxplot(data=datos_largo, x='año', y='Valor', hue='Indicador')
    plt.title('Distribución de Indicadores por Año')
    plt.xlabel('Año')
    plt.ylabel('Valor del Indicador')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / 'distribucion_anual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Matriz de correlación con todos los detalles
    plt.figure(figsize=(20, 16))
    corr_matrix = df[[col for col in df.columns if col != 'año']].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                center=0,
                square=True,
                xticklabels=[NOMBRES_VARIABLES[col] for col in corr_matrix.columns],
                yticklabels=[NOMBRES_VARIABLES[col] for col in corr_matrix.index])
    
    plt.title('Matriz de Correlación Completa')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / 'correlacion_completa.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Tendencias por grupos de indicadores
    grupos = {
        'Salud': ['hh_d_no_afil', 'hh_d_sin_salud'],
        'Vivienda': ['hh_d_materialidad', 'hh_d_hacinamiento', 'hh_d_sin_basur'],
        'Servicios': ['hh_d_agua_mejor', 'hh_d_san_mejor', 'hh_d_combus'],
        'Educación': ['hh_d_logro_min', 'hh_d_esc_retardada', 'hh_d_ni_noasis'],
        'Trabajo': ['hh_d_destotalmax', 'hh_d_subocup_max', 'hh_d_10a17_ocup', 'hh_d_jubi_pens']
    }
    
    for grupo, variables in grupos.items():
        plt.figure(figsize=(15, 8))
        for var in variables:
            datos = df.groupby('año')[var].mean()
            plt.plot(datos.index, datos.values * 100, marker='o',
                    linewidth=2, label=NOMBRES_VARIABLES[var])
            
            # Añadir etiquetas inicial y final
            plt.annotate(f'{datos.iloc[0]*100:.1f}%',
                        (datos.index[0], datos.iloc[0]*100),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
            plt.annotate(f'{datos.iloc[-1]*100:.1f}%',
                        (datos.index[-1], datos.iloc[-1]*100),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')
        
        plt.title(f'Evolución de Indicadores de {grupo}')
        plt.xlabel('Año')
        plt.ylabel('Porcentaje de Hogares Afectados')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTADOS_DIR / f'evolucion_{grupo.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\nVisualizaciones guardadas en la carpeta 'resultados':")
    print("1. evolucion_completa.png")
    print("2. distribucion_anual.png")
    print("3. correlacion_completa.png")
    print("4. evolucion_[grupo].png (un archivo por cada grupo temático)")

def main():
    """Función principal"""
    # Cargar datos
    df = pd.read_csv(DATOS_DIR / 'ipm_consolidado.csv')
    
    # Convertir columnas a numéricas (excepto año)
    for col in df.columns:
        if col != 'año':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Generar visualizaciones
    crear_visualizaciones_completas(df)

if __name__ == "__main__":
    main()


# In[ ]:




