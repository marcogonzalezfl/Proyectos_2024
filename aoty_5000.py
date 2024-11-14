#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd


# In[48]:


df = pd.read_csv('aoty.csv')


# In[60]:


df.head()


# In[86]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def limpiar_rating_count(x):
    try:
        if isinstance(x, str):
            return int(x.replace(' ratings', '').replace(',', ''))
        return x
    except:
        return np.nan

def add_trend_line(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
    line = slope * x + intercept
    return line, r_value**2

def analizar_albums(df):
    # Preparación inicial de datos
    df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')
    df['rating_count'] = df['rating_count'].apply(limpiar_rating_count)
    df['year'] = df['release_date'].apply(lambda x: int(x.split()[-1]) if pd.notna(x) else None)
    
    # Crear lista de géneros y DataFrame de géneros
    df['genres_list'] = df['genres'].str.split(',')
    df['genres_list'] = df['genres_list'].apply(lambda x: [g.strip() for g in x] if isinstance(x, list) else [])
    
    # Crear DataFrame de géneros expandido
    genre_df = pd.DataFrame([
        {'genre': genre.strip(),
         'user_score': row['user_score'],
         'rating_count': row['rating_count'],
         'year': row['year']}
        for _, row in df.iterrows()
        for genre in row['genres_list']
    ])

    # 1. ANÁLISIS BÁSICO
    print("\n=== Resumen General de la Base de Datos ===")
    print(f"• Total de álbumes analizados: {len(df):,}")
    print(f"• Período analizado: {df['year'].min()} - {df['year'].max()}")
    print(f"• Rango de calificaciones: {df['user_score'].min():.1f} - {df['user_score'].max():.1f}")
    print(f"• Calificación promedio: {df['user_score'].mean():.2f}/100")
    print(f"• Número típico de valoraciones por álbum (mediana): {df['rating_count'].median():,.0f}")
    print(f"• Promedio de valoraciones por álbum: {df['rating_count'].mean():,.0f}")

    # 2. ANÁLISIS POR DÉCADA
    print("\n=== Distribución por Décadas ===")
    decada_stats = df.groupby(df['year'].apply(lambda x: f"{(x//10)*10}s")).agg({
        'title': 'count',
        'user_score': ['mean', 'std'],
        'rating_count': 'mean'
    }).round(2)
    
    decada_stats.columns = ['Número de Álbumes', 'Calificación Promedio', 
                           'Desviación Estándar', 'Promedio de Valoraciones']
    print(decada_stats)

    # 3. VISUALIZACIÓN DE POPULARIDAD VS PUNTUACIÓN
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='user_score', y='rating_count', alpha=0.5)
    sns.regplot(data=df, x='user_score', y='rating_count', scatter=False, color='red')
    
    plt.yscale('log')
    plt.xlim(df['user_score'].min() - 1, df['user_score'].max() + 1)
    plt.ylim(df['rating_count'].min() * 0.9, df['rating_count'].max() * 1.1)
    
    plt.title('¿Los Álbumes Mejor Calificados Son Más Populares?', 
              fontsize=14, pad=20)
    plt.xlabel('Calificación del Álbum', fontsize=12)
    plt.ylabel('Número de Valoraciones (escala logarítmica)', fontsize=12)
    plt.show()

    # 4. ANÁLISIS DE ARTISTAS CONSISTENTES
    print("\n=== Artistas más Consistentes y Aclamados ===")
    artist_stats = df.groupby('artist').agg({
        'title': ['count', list],
        'user_score': ['mean', 'std']
    })
    
    consistent_artists = artist_stats[
        (artist_stats[('title', 'count')] >= 3) & 
        (artist_stats[('user_score', 'mean')] >= 85)
    ].sort_values(('user_score', 'mean'), ascending=False)
    
    for artist in consistent_artists.head().index:
        print(f"\n{artist}")
        print(f"• Número de álbumes: {consistent_artists.loc[artist, ('title', 'count')]:.0f}")
        print(f"• Calificación promedio: {consistent_artists.loc[artist, ('user_score', 'mean')]:.2f}/100")
        print(f"• Desviación estándar: {consistent_artists.loc[artist, ('user_score', 'std')]:.2f}")
        print("• Álbumes:", ', '.join(consistent_artists.loc[artist, ('title', 'list')]))

    # 5. ANÁLISIS DE GÉNEROS EMERGENTES
    print("\n=== Géneros Emergentes del Siglo XXI ===")
    early_genres = set(genre for genres in df[df['year'] < 2000]['genres_list'] for genre in genres)
    recent_genres = set(genre for genres in df[df['year'] >= 2000]['genres_list'] for genre in genres)
    
    emerging_genres = sorted(list(recent_genres - early_genres))
    print("\nNuevos géneros que surgieron después del año 2000:")
    for i, genre in enumerate(emerging_genres[:10], 1):
        print(f"{i}. {genre}")

    # 6. ANÁLISIS DE TENDENCIAS POR GÉNERO
    print("\n=== Evolución de los Géneros Musicales ===")
    genre_trends = []
    
    for genre in genre_df['genre'].value_counts().head(10).index:
        genre_data = genre_df[genre_df['genre'] == genre]
        correlation = genre_data['year'].corr(genre_data['user_score'])
        avg_score = genre_data['user_score'].mean()
        num_albums = len(genre_data)
        
        genre_trends.append({
            'Género': genre,
            'Correlación temporal': correlation.round(3),
            'Calificación promedio': avg_score.round(2),
            'Número de álbumes': num_albums,
            'Tendencia': 'Mejorando' if correlation > 0 else 'Declinando'
        })
    
    trend_df = pd.DataFrame(genre_trends).sort_values('Correlación temporal', ascending=False)
    print("\nTendencias de calificación por género:")
    print(trend_df.to_string(index=False))

    # 7. TOP ÁLBUMES
    print("\n=== Los Álbumes Más Destacados ===")
    print("\nLos 10 Álbumes Más Valorados por los Usuarios:")
    top_rated = df.nlargest(10, 'rating_count')[
        ['title', 'artist', 'user_score', 'rating_count', 'year']
    ].rename(columns={
        'title': 'Título',
        'artist': 'Artista',
        'user_score': 'Calificación',
        'rating_count': 'Valoraciones',
        'year': 'Año'
    })
    print(top_rated.to_string(index=False))
    
    print("\nLos 10 Álbumes Mejor Calificados (con más de 1000 valoraciones):")
    best_albums = df[df['rating_count'] > 1000].nlargest(10, 'user_score')[
        ['title', 'artist', 'user_score', 'rating_count', 'year']
    ].rename(columns={
        'title': 'Título',
        'artist': 'Artista',
        'user_score': 'Calificación',
        'rating_count': 'Valoraciones',
        'year': 'Año'
    })
    print(best_albums.to_string(index=False))

    # 8. VISUALIZACIÓN DE TENDENCIAS TEMPORALES
    plt.figure(figsize=(12, 8))
    plt.scatter(df['year'], df['user_score'], 
               s=df['rating_count']/500,
               alpha=0.5,
               label='Álbumes')
    
    sns.regplot(data=df, x='year', y='user_score', 
                scatter=False, color='red', 
                label='Tendencia general')
    
    plt.title('Evolución de las Calificaciones a lo Largo del Tiempo\n(tamaño = popularidad)',
              fontsize=14, pad=20)
    plt.xlabel('Año de Lanzamiento', fontsize=12)
    plt.ylabel('Calificación', fontsize=12)
    plt.legend()
    plt.show()

    # 9. DATOS INTERESANTES
    print("\n=== Datos Interesantes ===")
    print(f"• Género con mayor mejora: {trend_df.iloc[0]['Género']} (r={trend_df.iloc[0]['Correlación temporal']:.3f})")
    print(f"• Género con mayor declive: {trend_df.iloc[-1]['Género']} (r={trend_df.iloc[-1]['Correlación temporal']:.3f})")
    print(f"• Número de nuevos géneros desde 2000: {len(emerging_genres)}")
    print(f"• Artista más consistente: {consistent_artists.index[0]} ({consistent_artists.iloc[0][('user_score', 'mean')]:.2f}/100)")
    print(f"• Género mejor calificado: {trend_df.sort_values('Calificación promedio', ascending=False).iloc[0]['Género']}")

# Uso:
analizar_albums(df)


# In[ ]:




