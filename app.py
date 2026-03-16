import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importar la lógica del modelo del archivo local
from model_utils import (
    preparar_y_escalar_datos, 
    calcular_inercia, 
    aplicar_kmeans
)

# Definición de las columnas (ajusta si tus nombres son diferentes)
COLUMNAS_CLUSTERING = ['Ingresos Anuales', 'Puntuación de Gasto']

st.set_page_config(layout="wide")
st.title("Segmentación de Clientes con K-means")
st.markdown("---")

# --- Subir archivo ---
st.sidebar.header("Subir archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])

# ====================================================================
# 1. Carga de Datos y Preprocesamiento
# ====================================================================

# Cargar los datos. st.cache_data asegura que solo se ejecute la primera vez.
st.cache_data 
def cargar_y_preprocesar(file):
    # Si hay un archivo subido, lo leemos
    if file is not None:
        return pd.read_csv(file)
    
    # Si no hay nada subido, intentamos el local
    try:
        return pd.read_csv("data/clientes.csv")
    except FileNotFoundError:
        st.error("No se encontró 'data/clientes.csv' ni archivo subido.")
        return None

# LLAMADA A LA FUNCIÓN (Pasándole el archivo del uploader)
df_original = cargar_y_preprocesar(uploaded_file)

if df_original is not None:
    
    # Indicador visual de qué estamos viendo
    if uploaded_file is not None:
        st.sidebar.success(f"✅ Leyendo: {uploaded_file.name}")
    else:
        st.sidebar.info("Usando archivo local: clientes.csv")

    # --- SELECCIÓN DINÁMICA DE COLUMNAS (Para evitar el KeyError) ---
    st.sidebar.subheader("Variables para Clustering")
    
    # Esto permite que el usuario elija qué columnas usar del CSV que subió
    lista_columnas = df_original.columns.tolist()
    
    col_x = st.sidebar.selectbox("Selecciona Eje X:", lista_columnas, index=0)
    col_y = st.sidebar.selectbox("Selecciona Eje Y:", lista_columnas, index=min(1, len(lista_columnas)-1))
    
    # Actualizamos las columnas reales que vamos a usar
    COLUMNAS_CLUSTERING = [col_x, col_y]

    # --- LIMPIEZA SEGURA ---
    # Ahora sí, convertimos a número solo las columnas seleccionadas
    for col in COLUMNAS_CLUSTERING:
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

    # Quitamos filas vacías
    df_original.dropna(subset=COLUMNAS_CLUSTERING, inplace=True)
    
    # Procesamiento para el modelo
    # Usamos una copia limpia para el escalado
    X_original, df_escalado, escalador = preparar_y_escalar_datos(df_original, COLUMNAS_CLUSTERING)

    st.header("1. Datos Originales")
    st.markdown(f"Usando las columnas **{COLUMNAS_CLUSTERING[0]}** y **{COLUMNAS_CLUSTERING[1]}**.")
    st.dataframe(df_original.head(10))
    st.markdown("---")
    
    # ====================================================================
    # 2. Método del Codo Interactivo
    # ====================================================================
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("2. Método del Codo")
        st.write("La **Inercia** mide que tan cerca estan los puntos de su centroide")
        st.markdown("Elige el número de clusters (K) donde la curva se dobla (el 'codo').")
        
        # Interfaz para que el usuario elija K
        k_optimo = st.slider(
            "Elige K:", 
            min_value=2, max_value=10, value=5, 
            key="k_slider"
        )
        
    with col2:
        # Cálculo y Gráfica del Codo
        K_rango, inercia = calcular_inercia(df_escalado)
        fig_codo, ax_codo = plt.subplots(figsize=(8, 4))
        ax_codo.set_title('Método del Codo')
        ax_codo.plot(K_rango, inercia, marker='o', linestyle='--', color='blue')
        ax_codo.axvline(x=k_optimo, color='r', linestyle='--', label=f'K={k_optimo} Seleccionado')
        ax_codo.set_xlabel('Número de Clusters (k)')
        ax_codo.set_ylabel('Inercia (WCSS)')
        ax_codo.legend()
        ax_codo.grid(True)
        st.pyplot(fig_codo, use_container_width=True)
        
    st.markdown("---")
    
    # ====================================================================
    # 3. Aplicación del Modelo Final y Visualización
    # ====================================================================
    
    st.header(f"3. Resultados de Segmentación Final (K = {k_optimo})")
    
    # Llamar a la función de utilidades para aplicar K-means
    etiquetas, centroides_escalados = aplicar_kmeans(df_escalado, k_optimo)
    
    # Añadir las etiquetas al DataFrame original para el análisis
    df_original['Cluster'] = etiquetas
    
    # Transformación inversa de los centroides a la escala original
    centroides_originales = escalador.inverse_transform(centroides_escalados)
    
    # A. Análisis de Perfiles de Cluster
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("A. Perfiles Promedio de los Clusters")
        try:
            # USAMOS df_original que es el que definimos arriba y ya tiene la columna 'Cluster'
            # Agrupamos y promediamos indicando que solo queremos números
            perfiles = df_original.groupby('Cluster')[COLUMNAS_CLUSTERING].mean(numeric_only=True)
            
            # Mostramos con formato bonito
            st.dataframe(perfiles.style.format("{:.2f}").background_gradient(cmap='Blues'))
        except Exception as e:
            st.error(f"Error al calcular perfiles: {e}")
            
    # B. Gráfica de Segmentación
    with col4:
        st.subheader("B. Agrupacion de Clusters")
        st.write("Puntos de datos coloreados por cluster. Centroides marcados con 'X' roja.")

        fig_final, ax_final = plt.subplots(figsize=(10, 6))
        
        # Puntos de datos coloreados por cluster
        scatter = ax_final.scatter(df_original[COLUMNAS_CLUSTERING[0]], df_original[COLUMNAS_CLUSTERING[1]], 
                                   c=df_original['Cluster'], cmap='viridis', s=80, alpha=0.7)
        
        # Centroides marcados con 'X' roja
        ax_final.scatter(centroides_originales[:, 0], centroides_originales[:, 1], 
                         marker='X', s=200, c='red', edgecolor='black', 
                         label='Centroides')
        
        ax_final.set_title(f'Segmentación de clientes con K={k_optimo}')
        ax_final.set_xlabel(COLUMNAS_CLUSTERING[0])
        ax_final.set_ylabel(COLUMNAS_CLUSTERING[1])
        ax_final.legend()
        ax_final.grid(True)
        
        st.pyplot(fig_final, use_container_width=True)
else:
    st.warning("No se pudo cargar el archivo. Asegúrate de subir un CSV válido o que el archivo local exista.")