import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Para cuando carguemos el modelo real

st.set_page_config(page_title="Proyecto Recompra", layout="wide")

st.title("Análisis de Recompra y Fidelización")
st.markdown("---")

# ==========================================
# 1. IMPORTANCIA DE VARIABLES (RANDOM FOREST)
# ==========================================
st.header("Importancia de Variables (Modelo Global)")

# AQUÍ ENTRARÁ TU MODELO REAL. 
# Por ahora, dejo un espacio o un dummy para que me confirmes si 
# la estructura te gusta, y luego conectamos el .pkl
st.info("Aquí insertaremos el gráfico de Feature Importance generado directamente por tu Random Forest.")

st.markdown("---")

# ==========================================
# 2. CARGA DE DATOS Y FILTROS
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
# (Dejamos solo Ciudad por ahora para probar rápido)
ciudades = ["Todas"] + list(df['Ciudad'].unique())
ciudad_sel = st.sidebar.selectbox("Ciudad", ciudades)
df_filt = df[df['Ciudad'] == ciudad_sel] if ciudad_sel != "Todas" else df.copy()

st.header(f"Análisis Local: {ciudad_sel if ciudad_sel != 'Todas' else 'Nacional'}")

# ==========================================
# 3. GRÁFICA DE EXPERIENCIA (TU CÓDIGO CLONADO)
# ==========================================
st.subheader("1. Experiencia del Staff (Antigüedad)")

# Calculamos los quintiles globales para la Antigüedad
_, bins_exp = pd.qcut(df['Antiguedad_Promedio_Ponderada_3M'], q=5, retbins=True, duplicates='drop')
bins_exp[0], bins_exp[-1] = -np.inf, np.inf

# Aplicamos los bins globales a la data filtrada
df_calc = df_filt.copy()
df_calc['Rango_Exp'] = pd.cut(df_calc['Antiguedad_Promedio_Ponderada_3M'], bins=bins_exp)

# Agrupamos
resumen_exp = df_calc.groupby('Rango_Exp', observed=False).agg(
    Tasa_Recompra=('Alta_Recompra', 'mean') # Usando la probabilidad de superar 13.7%
).reset_index()

resumen_exp['Rango_Exp'] = resumen_exp['Rango_Exp'].astype(str)
resumen_exp.dropna(subset=['Tasa_Recompra'], inplace=True)

# TU GRÁFICA CON SEABORN
if len(resumen_exp) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # El lineplot exacto que tenías en tu notebook
    sns.lineplot(data=resumen_exp, x='Rango_Exp', y='Tasa_Recompra', 
                 marker='o', markersize=10, linewidth=3, color='#2c3e50', sort=False, ax=ax)
    
    # Tus etiquetas de datos en %
    for x, y in zip(range(len(resumen_exp)), resumen_exp['Tasa_Recompra']):
        ax.text(x, y + 0.01, f'{y*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title('Relación entre la Antigüedad del Staff y la Recompra', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probabilidad Alta Recompra (%)', fontsize=12)
    ax.set_xlabel('Rango de Antigüedad', fontsize=12)
    ax.set_ylim(0, resumen_exp['Tasa_Recompra'].max() * 1.3)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    st.pyplot(fig)
else:
    st.warning("No hay suficientes datos en esta selección para generar la gráfica.")
