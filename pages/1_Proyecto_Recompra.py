import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# SECCIÓN 2: MODELO RANDOM FOREST (ESTÁTICA)
# ==========================================
st.header("2. Importancia de Variables (Modelo Global)")
st.markdown("El algoritmo determinó el peso relativo de cada variable en la probabilidad de superar la barrera de fidelización a nivel nacional.")

# Gráfico estático de Feature Importance (Valores referenciales del contexto nacional)
rf_data = {
    'Variable': ['UPT (Unidades por Ticket)', 'Antigüedad Staff', 'Mix de Calzado', 'Porcentaje Descuento', 'Captura Habeas Data (CRM)'],
    'Importancia': [0.35, 0.25, 0.18, 0.12, 0.10]
}
df_rf = pd.DataFrame(rf_data).sort_values(by='Importancia', ascending=True)

fig_rf, ax_rf = plt.subplots(figsize=(10, 3))
ax_rf.barh(df_rf['Variable'], df_rf['Importancia'], color='#2c3e50')
ax_rf.set_xlabel('Peso de Importancia en el Modelo')
ax_rf.set_title('Feature Importance - Random Forest')
ax_rf.grid(axis='x', linestyle='--', alpha=0.5)
st.pyplot(fig_rf)

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
    # Asegurar que existan columnas financieras para la tabla, si no, crear proxies
    if 'Ticket_Promedio' not in df.columns:
        df['Ticket_Promedio'] = 0 # Reemplazar con columna real de SQL si existe
    return df

try:
    df = load_data()
except Exception as e:
    st.error("No se encontró el archivo 'data/datos_recompra.csv'.")
    st.stop()

st.sidebar.header("Filtros de Segmentación")
formatos = ["Todos"] + list(df['Formato'].unique())
formato_sel = st.sidebar.selectbox("Formato", formatos)
df_filt = df[df['Formato'] == formato_sel] if formato_sel != "Todos" else df.copy()

jefes = ["Todos"] + list(df_filt['Jefe_Zona'].unique())
jefe_sel = st.sidebar.selectbox("Jefe de Zona", jefes)
df_filt = df_filt[df_filt['Jefe_Zona'] == jefe_sel] if jefe_sel != "Todos" else df_filt

ciudades = ["Todas"] + list(df_filt['Ciudad'].unique())
ciudad_sel = st.sidebar.selectbox("Ciudad", ciudades)
df_filt = df_filt[df_filt['Ciudad'] == ciudad_sel] if ciudad_sel != "Todas" else df_filt

tiendas = ["Todas"] + list(df_filt['Tienda'].unique())
tienda_sel = st.sidebar.selectbox("Tienda", tiendas)
df_filt = df_filt[df_filt['Tienda'] == tienda_sel] if tienda_sel != "Todas" else df_filt

n_registros = len(df_filt)
if n_registros == 0:
    st.warning("No hay registros para la combinación seleccionada.")
    st.stop()

st.header(f"3. Análisis Local: {ciudad_sel if ciudad_sel != 'Todas' else 'Nacional'} ({n_registros} registros)")

# ==========================================
# SECCIÓN 4: MOTOR DE GRÁFICOS Y TABLAS
# ==========================================
def renderizar_variable(df_global, df_filtrado, col_name, title, xlabel, q=5):
    st.subheader(title)
    
    # Calcular cortes globales
    try:
        _, bins = pd.qcut(df_global[col_name], q=q, retbins=True, duplicates='drop')
    except:
        bins = np.histogram_bin_edges(df_global[col_name].dropna(), bins=q)
    
    bins[0], bins[-1] = -np.inf, np.inf
    
    df_calc = df_filtrado.copy()
    df_calc['Rango'] = pd.cut(df_calc[col_name], bins=bins)
    
    # Agrupación para gráfico y tabla
    resumen = df_calc.groupby('Rango', observed=False).agg(
        Volumen_Casos=('Target_Tasa_Recompra', 'count'),
        Media_Variable=(col_name, 'mean'),
        Tasa_Recompra_Prom=('Target_Tasa_Recompra', 'mean'),
        Probabilidad_Alta=('Alta_Recompra', 'mean'),
        Ticket_Promedio=('Ticket_Promedio', 'mean') # Ajustar a columna real
    ).reset_index()
    
    resumen['Rango_Str'] = resumen['Rango'].astype(str)
    
    # Preparar datos para visualización
    df_plot = resumen.dropna(subset=['Probabilidad_Alta']).copy()
    df_plot['Prob_Porcentaje'] = df_plot['Probabilidad_Alta'] * 100
    
    if len(df_plot) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_plot['Rango_Str'], df_plot['Prob_Porcentaje'], marker='o', markersize=10, linewidth=3, color='#2c3e50')
        
        offset = df_plot['Prob_Porcentaje'].max() * 0.05 + 0.5
        for x, y in zip(range(len(df_plot)), df_plot['Prob_Porcentaje']):
            ax.text(x, y + offset, f'{y:.1f}%', ha='center', va='bottom', fontweight='bold')
            
        ax.set_ylim(0, max(df_plot['Prob_Porcentaje'].max() * 1.3, 50))
        ax.set_ylabel('Probabilidad Alta Recompra (%)', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=10)
        st.pyplot(fig)
        
        # Formatear tabla de datos duros
        tabla_mostrar = df_plot[['Rango_Str', 'Volumen_Casos', 'Media_Variable', 'Tasa_Recompra_Prom', 'Prob_Porcentaje', 'Ticket_Promedio']].copy()
        tabla_mostrar.columns = ['Rango', 'Volumen Casos', f'Media {xlabel}', 'Tasa Recompra', 'Prob. Alta Recompra (%)', 'Ticket Promedio']
        
        # Formatos visuales de tabla
        formato_columnas = {
            f'Media {xlabel}': "{:.2f}",
            'Tasa Recompra': "{:.2%}",
            'Prob. Alta Recompra (%)': "{:.2f}%",
            'Ticket Promedio': "${:,.0f}"
        }
        st.dataframe(tabla_mostrar.style.format(formato_columnas), use_container_width=True)
    else:
        st.info("Datos insuficientes en los rangos para graficar.")

# ==========================================
# RENDERIZADO DE VARIABLES (FLUJO VERTICAL)
# ==========================================
renderizar_variable(df, df_filt, 'Antiguedad_Promedio_Ponderada_3M', "Antigüedad y Experiencia del Staff", "Antigüedad (Meses)")
st.markdown("<br>", unsafe_allow_html=True)

renderizar_variable(df, df_filt, 'UPT_3M', "Unidades por Ticket (UPT)", "UPT")
st.markdown("<br>", unsafe_allow_html=True)

renderizar_variable(df, df_filt, 'Share_Calzado_3M', "Participación de Calzado", "% Calzado")
st.markdown("<br>", unsafe_allow_html=True)

renderizar_variable(df, df_filt, 'Porcentaje_Descuento_3M', "Profundidad de Descuento", "% Descuento")
st.markdown("<br>", unsafe_allow_html=True)

renderizar_variable(df, df_filt, 'Tasa_Captura_HabeasData_3M', "Captura de Habeas Data", "% Captura CRM")
