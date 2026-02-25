import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Proyecto Recompra", page_icon="🔄", layout="wide")

st.title("🔄 Radiografía de Recompra por Ciudad")
st.markdown("Análisis de las 6 palancas clave para escalar la **Probabilidad de Alta Recompra (>13.7%)**.")

# 1. CARGA DE DATOS
@st.cache_data
def load_data():
    # Cargamos el CSV extraído de SQL
    df = pd.read_csv("data/datos_recompra.csv")
    
    # Crear variable objetivo (Probabilidad de Alta Recompra)
    df['Alta_Recompra'] = (df['Target_Tasa_Recompra'] > 0.137).astype(int)
    
    # Rellenar nulos en los filtros para evitar errores
    for col in ['Formato', 'Jefe_Zona', 'Ciudad', 'Tienda']:
        if col in df.columns:
            df[col] = df[col].fillna("Sin Asignar")
            
    return df

try:
    df = load_data()
except Exception as e:
    st.error("No se encontró el archivo 'data/datos_recompra.csv'. Asegúrate de exportarlo desde el Notebook.")
    st.stop()

# 2. FILTROS EN CASCADA
st.sidebar.header("🎯 Filtros Estratégicos")

# Formato
formatos = ["Todos"] + list(df['Formato'].unique())
formato_sel = st.sidebar.selectbox("Formato", formatos)
df_filt = df[df['Formato'] == formato_sel] if formato_sel != "Todos" else df.copy()

# Jefe de Zona
jefes = ["Todos"] + list(df_filt['Jefe_Zona'].unique())
jefe_sel = st.sidebar.selectbox("Jefe de Zona", jefes)
df_filt = df_filt[df_filt['Jefe_Zona'] == jefe_sel] if jefe_sel != "Todos" else df_filt

# Ciudad
ciudades = ["Todas"] + list(df_filt['Ciudad'].unique())
ciudad_sel = st.sidebar.selectbox("Ciudad", ciudades)
df_filt = df_filt[df_filt['Ciudad'] == ciudad_sel] if ciudad_sel != "Todas" else df_filt

# Tienda
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

# 4. FUNCIÓN GENERADORA DE GRÁFICOS (Mantiene los cortes del nivel nacional)
def plot_palanca(df_global, df_filtrado, col_name, title, q=5):
    # Cortes sobre el global para evitar distorsiones por ciudad
    try:
        _, bins = pd.qcut(df_global[col_name], q=q, retbins=True, duplicates='drop')
    except:
        bins = np.histogram_bin_edges(df_global[col_name].dropna(), bins=q)
    
    bins[0], bins[-1] = -np.inf, np.inf # Extender límites
    
    # Calcular métricas en la data filtrada usando los cortes globales
    df_calc = df_filtrado.copy()
    df_calc['Rango'] = pd.cut(df_calc[col_name], bins=bins)
    
    resumen = df_calc.groupby('Rango', observed=False).agg(
        Probabilidad=('Alta_Recompra', 'mean'),
        Volumen=('Target_Tasa_Recompra', 'count')
    ).reset_index()
    
    # Formateo
    resumen['Rango'] = resumen['Rango'].astype(str)
    resumen['Probabilidad'] = resumen['Probabilidad'] * 100
    resumen.dropna(subset=['Probabilidad'], inplace=True)
    
    fig = px.line(resumen, x='Rango', y='Probabilidad', markers=True, 
                  title=title, text='Probabilidad')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='top center', 
                      line=dict(width=3, color='#2c3e50'), marker=dict(size=8))
    fig.update_layout(yaxis_title="Probabilidad Alta Recompra (%)", xaxis_title="")
    if len(resumen) > 0:
        fig.update_yaxes(range=[0, max(resumen['Probabilidad'].max() * 1.2, 50)])
        
    return fig

# 5. CREACIÓN DE LOS GRÁFICOS
fig_upt = plot_palanca(df, df_filt, 'UPT_3M', "1. Impacto del UPT (Unidades por Ticket)")
fig_calzado = plot_palanca(df, df_filt, 'Share_Calzado_3M', "2. Mix de Calzado (La Curva de Oro)")
fig_desc = plot_palanca(df, df_filt, 'Porcentaje_Descuento_3M', "3. Techo de Descuento vs Fidelización")
fig_staff = plot_palanca(df, df_filt, 'Antiguedad_Promedio_Ponderada_3M', "4. Experiencia del Staff (Meses)")
fig_crm = plot_palanca(df, df_filt, 'Tasa_Captura_HabeasData_3M', "5. CRM y Client Book (Habeas Data)")

# Gráfico Especial: Geografía (Scatter)
if ciudad_sel == "Todas":
    df_geo = df_filt.groupby('Ciudad').agg(
        Recompra_Mediana=('Target_Tasa_Recompra', 'median'),
        Volumen=('Target_Tasa_Recompra', 'count')
    ).reset_index()
    fig_geo = px.scatter(df_geo, x='Volumen', y='Recompra_Mediana', text='Ciudad', 
                         title="Efecto Monopolio vs Competencia (Por Ciudad)", size='Volumen')
else:
    df_geo = df_filt.groupby('Tienda').agg(
        Recompra_Mediana=('Target_Tasa_Recompra', 'median'),
        Volumen=('Target_Tasa_Recompra', 'count')
    ).reset_index()
    fig_geo = px.scatter(df_geo, x='Volumen', y='Recompra_Mediana', text='Tienda', 
                         title=f"Rendimiento Interno en {ciudad_sel} (Por Tienda)", size='Volumen')

fig_geo.update_traces(textposition='top center')
fig_geo.update_layout(yaxis_title="Tasa de Recompra (Mediana)", xaxis_title="Volumen de Casos")

# 6. RENDERIZADO EN PANTALLA (2 Columnas)
col1, col2 = st.columns(2)

with col1:
    if fig_upt: st.plotly_chart(fig_upt, use_container_width=True)
    if fig_desc: st.plotly_chart(fig_desc, use_container_width=True)
    if fig_staff: st.plotly_chart(fig_staff, use_container_width=True)

with col2:
    if fig_calzado: st.plotly_chart(fig_calzado, use_container_width=True)
    if fig_crm: st.plotly_chart(fig_crm, use_container_width=True)
    if fig_geo: st.plotly_chart(fig_geo, use_container_width=True)