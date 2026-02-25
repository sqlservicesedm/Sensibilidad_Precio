import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Proyecto Recompra", layout="wide")

st.title("Análisis de Recompra y Fidelización")
st.markdown("---")

# ==========================================
# SECCIÓN 1: CONTEXTO Y FICHA TÉCNICA (ESTÁTICA)
# ==========================================
st.header("1. Contexto del Análisis")

col_ctx1, col_ctx2 = st.columns(2)

with col_ctx1:
    st.markdown("""
    **Ficha Técnica del Modelo:**
    * **Periodo de Análisis:** 2023-03-01 – Actualidad.
    * **Metodología:** Evaluación sobre ventanas de tres meses móviles.
    * **Exclusiones Aplicadas:** Canal Online, Eventos, Sale, Replay, Cédulas genéricas.
    """)

with col_ctx2:
    st.markdown("""
    **Grupos de Variables Evaluadas:**
    * **Calidad del Vendedor:** UPT, Antigüedad promedio.
    * **Venta:** Ticket promedio, Porcentaje de descuento, Tasa de devoluciones.
    * **Categorías:** Participación (Top, Bottom, Calzado, Outfit completo).
    * **Habeas Data:** Aceptación de políticas de contacto (Email, SMS).
    """)

st.markdown("---")

# ==========================================
# SECCIÓN 2: MODELO RANDOM FOREST (TU CÓDIGO EXACTO)
# ==========================================
st.header("2. Importancia de Variables (Modelo Global)")

# Cargar el modelo en caché para que solo lo lea una vez
@st.cache_resource
def load_model():
    return joblib.load('data/modelo_recompra.pkl')

try:
    rf = load_model()
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Extraemos los nombres de las variables directamente del modelo guardado
    if hasattr(rf, 'feature_names_in_'):
        feature_names = rf.feature_names_in_
    else:
        # Por si usas una versión antigua de sklearn
        feature_names = np.array([f"Variable {i}" for i in range(len(importances))])

    # TU CÓDIGO DE VISUALIZACIÓN EXACTO
    fig_rf, ax_rf = plt.subplots(figsize=(15, 8)) 
    ax_rf.set_title("¿Qué mueve realmente la aguja de la Recompra?", fontsize=14, fontweight='bold', pad=20)

    colors = ['#1f77b4' if i < 3 else '#95a5a6' for i in range(len(indices))]
    ax_rf.bar(range(len(importances)), importances[indices], align="center", color=colors)
    ax_rf.set_xticks(range(len(importances)))
    ax_rf.set_xticklabels(feature_names[indices], fontsize=10, rotation=90)
    ax_rf.set_ylabel("Peso Relativo (Importancia)", fontsize=10)
    ax_rf.set_ylim(0, max(importances) * 1.15) 

    for i, v in enumerate(importances[indices]):
        ax_rf.text(i, v + 0.002, f'{v:.3f}', color='black', fontsize=9, ha='center', va='bottom')

    ax_rf.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig_rf)

except Exception as e:
    st.error(f"Error al cargar el modelo: Asegúrate de que 'modelo_recompra.pkl' esté en la carpeta 'data/'. Detalles: {e}")

st.markdown("---")

# ==========================================
# SECCIÓN 3: CARGA DE DATOS Y FILTROS
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/datos_recompra.csv")
    df['Alta_Recompra'] = (df['Target_Tasa_Recompra'] > 0.137).astype(int)
    for col in ['Formato', 'Jefe_Zona', 'Ciudad', 'Tienda']:
        if col in df.columns:
            df[col] = df[col].fillna("Sin Asignar")
    return df

try:
    df = load_data()
except Exception as e:
    st.error("No se encontró el archivo 'data/datos_recompra.csv'.")
    st.stop()

st.sidebar.header("Filtros de Segmentación")
ciudades = ["Todas"] + list(df['Ciudad'].unique())
ciudad_sel = st.sidebar.selectbox("Ciudad", ciudades)
df_filt = df[df['Ciudad'] == ciudad_sel] if ciudad_sel != "Todas" else df.copy()

st.header(f"3. Análisis Local: {ciudad_sel if ciudad_sel != 'Todas' else 'Nacional'}")

# ==========================================
# SECCIÓN 4: GRÁFICA DE EXPERIENCIA (TU CÓDIGO)
# ==========================================
st.subheader("Experiencia del Staff (Antigüedad)")

# Calculamos los quintiles globales para la Antigüedad
try:
    _, bins_exp = pd.qcut(df['Antiguedad_Promedio_Ponderada_3M'], q=5, retbins=True, duplicates='drop')
except:
    bins_exp = np.histogram_bin_edges(df['Antiguedad_Promedio_Ponderada_3M'].dropna(), bins=5)

bins_exp[0], bins_exp[-1] = -np.inf, np.inf

# Aplicamos los bins globales a la data filtrada
df_calc = df_filt.copy()
df_calc['Rango_Exp'] = pd.cut(df_calc['Antiguedad_Promedio_Ponderada_3M'], bins=bins_exp)

# Agrupamos
resumen_exp = df_calc.groupby('Rango_Exp', observed=False).agg(
    Tasa_Recompra=('Alta_Recompra', 'mean')
).reset_index()

resumen_exp['Rango_Exp'] = resumen_exp['Rango_Exp'].astype(str)
resumen_exp.dropna(subset=['Tasa_Recompra'], inplace=True)

if len(resumen_exp) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Tu lineplot exacto
    sns.lineplot(data=resumen_exp, x='Rango_Exp', y='Tasa_Recompra', 
                 marker='o', markersize=10, linewidth=3, color='#2c3e50', sort=False, ax=ax)
    
    for x, y in zip(range(len(resumen_exp)), resumen_exp['Tasa_Recompra']):
        ax.text(x, y + 0.01, f'{y*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Relación entre la Antigüedad del Staff y la Recompra', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probabilidad Alta Recompra (%)', fontsize=12)
    ax.set_xlabel('Rango de Antigüedad', fontsize=12)
    ax.set_ylim(0, max(resumen_exp['Tasa_Recompra'].max() * 1.3, 0.5))
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    st.pyplot(fig)
else:
    st.warning("No hay suficientes datos en esta selección para generar la gráfica.")
