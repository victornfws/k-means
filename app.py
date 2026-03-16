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
@st.cache_data 
def cargar_y_preprocesar():
    def cargar_datos(archivo_subido):
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
    try:
        return pd.read_csv("data/clientes.csv")
    except FileNotFoundError:
        return None
        
df_original = cargar_y_preprocesar()

if df_original is not None:
    
    # Forzar a que las columnas sean numericas (si no lo son)
    for col in COLUMNAS_CLUSTERING:
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

    # Eliminar filas que hayan quedado con NaN después de la conversión
    df_original.dropna(subset=COLUMNAS_CLUSTERING, inplace=True)

    # Procesamiento
    df_copia = df_original.copy()
    X_original, df_escalado, escalador = preparar_y_escalar_datos(df_copia, COLUMNAS_CLUSTERING)

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
        st.subheader("A. Perfiles promedio de los Clusters")
        # 1. Aseguramos que los datos sean numéricos y calculamos el promedio
        try:
            # Forzamos numérico por si el CSV venía "sucio"
            df_copia[COLUMNAS_CLUSTERING] = df_copia[COLUMNAS_CLUSTERING].apply(pd.to_numeric, errors='coerce')
            
            # Agrupamos y promediamos indicando que solo queremos números
            perfiles = df_original.groupby('Cluster')[COLUMNAS_CLUSTERING].mean(numeric_only=True)
            
            # Mostramos con formato bonito
            st.dataframe(perfiles.style.format("{:.2f}").background_gradient(cmap='Blues'))
        except Exception as e:
            st.error(f"Error al calcular perfiles: {e}")
      
    # B. Gráfica de Segmentación
    with col4:
        st.subheader("B. Agrupacion de Clusters")
        
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