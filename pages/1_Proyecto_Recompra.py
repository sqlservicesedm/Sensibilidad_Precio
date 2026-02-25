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
# SECCIÓN 2: MODELO RANDOM FOREST (ESTÁTICO)
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('data/modelo_recompra.pkl')

try:
    rf = load_model()
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    if hasattr(rf, 'feature_names_in_'):
        feature_names = rf.feature_names_in_
    else:
        feature_names = np.array([f"Variable {i}" for i in range(len(importances))])

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
# SECCIÓN 3: CARGA DE DATOS Y FILTROS EN CASCADA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/datos_recompra.csv")
    
    # 1. Definimos la Meta Objetiva y las categorías globalmente para fijar el eje X
    META_OBJETIVA = 0.137
    df['Es_Elite'] = df['Target_Tasa_Recompra'] >= META_OBJETIVA
    
    bins_final = [0, 12, 24, 48, 96, 240]
    labels_final = [
        '1. Nuevos (<1 Año)',
        '2. En Consolidación (1-2 Años)',
        '3. Estables (2-4 Años)',
        '4. Expertos (4-8 Años)',
        '5. Veteranos (>8 Años)'
    ]
    df['Rango_Experiencia'] = pd.cut(
        df['Antiguedad_Promedio_Ponderada_3M'], 
        bins=bins_final, 
        labels=labels_final
    )
    
    for col in ['Formato', 'Jefe_Zona', 'Ciudad', 'Tienda']:
        if col in df.columns:
            df[col] = df[col].fillna("Sin Asignar")
            
    # Proxy por si no existe la columna de ticket
    if 'Ticket_Promedio_3M' not in df.columns:
        df['Ticket_Promedio_3M'] = 0
        
    return df

try:
    df = load_data()
except Exception as e:
    st.error("No se encontró el archivo 'data/datos_recompra.csv'.")
    st.stop()

st.sidebar.header("Filtros de Segmentación")

formatos = ["Todos"] + list(df['Formato'].unique())
formato_sel = st.sidebar.selectbox("Formato", formatos)
df_f1 = df[df['Formato'] == formato_sel] if formato_sel != "Todos" else df.copy()

jefes = ["Todos"] + list(df_f1['Jefe_Zona'].unique())
jefe_sel = st.sidebar.selectbox("Jefe de Zona", jefes)
df_f2 = df_f1[df_f1['Jefe_Zona'] == jefe_sel] if jefe_sel != "Todos" else df_f1

ciudades = ["Todas"] + list(df_f2['Ciudad'].unique())
ciudad_sel = st.sidebar.selectbox("Ciudad", ciudades)
df_f3 = df_f2[df_f2['Ciudad'] == ciudad_sel] if ciudad_sel != "Todas" else df_f2

tiendas = ["Todas"] + list(df_f3['Tienda'].unique())
tienda_sel = st.sidebar.selectbox("Tienda", tiendas)
df_filt = df_f3[df_f3['Tienda'] == tienda_sel] if tienda_sel != "Todas" else df_f3

n_registros = len(df_filt)
if n_registros == 0:
    st.warning("No hay registros para la combinación seleccionada.")
    st.stop()
elif n_registros < 30:
    st.sidebar.warning(f"⚠️ Selección con muestra baja ({n_registros} registros). Interpretar promedios con precaución.")

# ==========================================
# SECCIÓN 4: ANÁLISIS DINÁMICO (TU CÓDIGO EXACTO DE EXPERIENCIA)
# ==========================================

st.subheader("Experiencia del Staff")

# Limpieza y preparación sobre la data filtrada (df_filt en vez de df)
df_clean = df_filt[(df_filt['Antiguedad_Promedio_Ponderada_3M'] >= 0) & 
                   (df_filt['Antiguedad_Promedio_Ponderada_3M'] <= 240)].copy()

# Tabla Maestra de Staff
resumen_staff = df_clean.groupby('Rango_Experiencia', observed=False).agg(
    Casos=('Target_Tasa_Recompra', 'count'),
    Seniority_Mediano_Meses=('Antiguedad_Promedio_Ponderada_3M', 'median'),
    Recompra_Mediana=('Target_Tasa_Recompra', 'median'),
    Prob_Exito_Elite=('Es_Elite', 'mean'),  
    Ticket_Mediano=('Ticket_Promedio_3M', 'median')
).reset_index()

resumen_staff['Prob_Exito_Elite'] = (resumen_staff['Prob_Exito_Elite'] * 100).round(1)
resumen_staff['Recompra_Mediana'] = resumen_staff['Recompra_Mediana'].fillna(0) 

# Visualización
fig_staff, ax_staff = plt.subplots(figsize=(12, 6))
sns.lineplot(data=resumen_staff, x='Rango_Experiencia', y='Recompra_Mediana', 
             marker='s', markersize=12, linewidth=4, color='#1b4f72', sort=False, ax=ax_staff)

for x, y in zip(range(len(resumen_staff)), resumen_staff['Recompra_Mediana']):
    if resumen_staff.loc[x, 'Casos'] > 0: 
        ax_staff.text(x, y + 0.001, f'{y*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax_staff.set_title('Impacto de la Experiencia en la Recompra', fontsize=15, fontweight='bold')
ax_staff.set_ylabel('Tasa de Recompra Mediana (%)')
ax_staff.set_xlabel('Años de Experiencia del Equipo en Tienda')
ax_staff.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

st.pyplot(fig_staff)

# Mostrar Tabla Maestra 
tabla_mostrar = resumen_staff.copy()
tabla_mostrar.columns = ['Rango Experiencia', 'Casos', 'Seniority Mediano (Meses)', 'Recompra Mediana', 'Prob. Éxito (%)', 'Ticket Mediano']

formato_columnas = {
    'Seniority Mediano (Meses)': "{:.1f}",
    'Recompra Mediana': "{:.2%}",
    'Prob. Éxito (%)': "{:.1f}%",
    'Ticket Mediano': "${:,.0f}"
}

st.dataframe(tabla_mostrar.style.format(formato_columnas), use_container_width=True)
