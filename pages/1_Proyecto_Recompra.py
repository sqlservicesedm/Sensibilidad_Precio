import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Proyecto Recompra", page_icon="🔄", layout="wide")

st.title("🔄 Radiografía de Recompra por Ciudad")
st.markdown("Análisis de las 6 palancas clave para escalar la **Probabilidad de Alta Recompra (>13.7%)**.")

# 1. CARGA DE DATOS
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

# 2. FILTROS EN CASCADA
st.sidebar.header("🎯 Filtros Estratégicos")
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

# 3. ADVERTENCIA DE MUESTRA
n_registros = len(df_filt)
if n_registros == 0:
    st.warning("No hay datos para la selección actual.")
    st.stop()
elif n_registros < 30:
    st.warning(f"⚠️ Atención: Esta selección tiene solo **{n_registros} registros**. Los promedios deben interpretarse con cautela estadística.")
else:
    st.success(f"✅ Analizando **{n_registros}** casos de estudio bajo los filtros seleccionados.")

st.markdown("---")

# 4. FUNCIÓN GENERADORA ESTILO MATPLOTLIB (Tu estilo original)
def plot_palanca_mpl(df_global, df_filtrado, col_name, title, xlabel, q=5):
    # Cortes sobre el global para respetar los bins
    try:
        _, bins = pd.qcut(df_global[col_name], q=q, retbins=True, duplicates='drop')
    except:
        bins = np.histogram_bin_edges(df_global[col_name].dropna(), bins=q)
    
    bins[0], bins[-1] = -np.inf, np.inf
    
    # Calcular métricas en data filtrada
    df_calc = df_filtrado.copy()
    df_calc['Rango'] = pd.cut(df_calc[col_name], bins=bins)
    
    resumen = df_calc.groupby('Rango', observed=False).agg(
        Probabilidad=('Alta_Recompra', 'mean'),
        Volumen=('Target_Tasa_Recompra', 'count')
    ).reset_index()
    
    # Limpieza para el gráfico
    resumen['Rango_Str'] = resumen['Rango'].astype(str)
    resumen['Probabilidad'] = resumen['Probabilidad'] * 100
    resumen.dropna(subset=['Probabilidad'], inplace=True)
    
    # Construcción de la visual
    fig, ax = plt.subplots(figsize=(10, 6))
    if len(resumen) > 0:
        ax.plot(resumen['Rango_Str'], resumen['Probabilidad'], marker='o', markersize=10, linewidth=3, color='#2c3e50')
        
        # Etiquetas de texto
        offset = resumen['Probabilidad'].max() * 0.05 + 0.5
        for x, y in zip(range(len(resumen)), resumen['Probabilidad']):
            ax.text(x, y + offset, f'{y:.1f}%', ha='center', va='bottom', fontweight='bold')
            
        ax.set_ylim(0, max(resumen['Probabilidad'].max() * 1.3, 50))
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Probabilidad Alta Recompra (%)', fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    return fig

# 5. CREACIÓN DE GRÁFICOS
fig_upt = plot_palanca_mpl(df, df_filt, 'UPT_3M', "1. Impacto del UPT", "Unidades por Ticket")
fig_calzado = plot_palanca_mpl(df, df_filt, 'Share_Calzado_3M', "2. Mix de Calzado (La Curva de Oro)", "Participación Calzado")
fig_desc = plot_palanca_mpl(df, df_filt, 'Porcentaje_Descuento_3M', "3. Techo de Descuento vs Fidelización", "% Descuento")
fig_staff = plot_palanca_mpl(df, df_filt, 'Antiguedad_Promedio_Ponderada_3M', "4. Experiencia del Staff", "Antigüedad (Meses)")
fig_crm = plot_palanca_mpl(df, df_filt, 'Tasa_Captura_HabeasData_3M', "5. CRM y Client Book (Habeas Data)", "Tasa de Captura (%)")

# Gráfico Especial: Geografía (Scatter estático)
if ciudad_sel == "Todas":
    df_geo = df_filt.groupby('Ciudad').agg(
        Probabilidad=('Alta_Recompra', 'mean'),
        Volumen=('Target_Tasa_Recompra', 'count')
    ).reset_index()
    text_col = 'Ciudad'
    title_geo = "Efecto Monopolio vs Competencia (Por Ciudad)"
else:
    df_geo = df_filt.groupby('Tienda').agg(
        Probabilidad=('Alta_Recompra', 'mean'),
        Volumen=('Target_Tasa_Recompra', 'count')
    ).reset_index()
    text_col = 'Tienda'
    title_geo = f"Rendimiento Interno en {ciudad_sel} (Por Tienda)"

df_geo['Probabilidad'] = df_geo['Probabilidad'] * 100

fig_geo, ax_geo = plt.subplots(figsize=(10, 6))
if len(df_geo) > 0:
    ax_geo.scatter(df_geo['Volumen'], df_geo['Probabilidad'], s=df_geo['Volumen'].clip(lower=20)*2, alpha=0.6, color='#2c3e50')
    for i, txt in enumerate(df_geo[text_col]):
        ax_geo.annotate(txt, (df_geo['Volumen'].iloc[i], df_geo['Probabilidad'].iloc[i]), 
                        ha='center', va='bottom', fontsize=9, xytext=(0, 5), textcoords='offset points')
    
ax_geo.set_title(title_geo, fontsize=14, fontweight='bold')
ax_geo.set_ylabel('Probabilidad Alta Recompra (%)', fontsize=12)
ax_geo.set_xlabel('Volumen de Casos', fontsize=12)
ax_geo.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 6. RENDERIZADO (2 Columnas)
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig_upt)
    st.pyplot(fig_desc)
    st.pyplot(fig_staff)
with col2:
    st.pyplot(fig_calzado)
    st.pyplot(fig_crm)
    st.pyplot(fig_geo)