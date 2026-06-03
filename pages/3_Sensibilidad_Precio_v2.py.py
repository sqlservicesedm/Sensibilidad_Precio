"""
sensibilidad_precio.py

Pagina de Streamlit para el analisis de curvas de densidad y optimizacion
de inventario por rango de precio.

Muestra tres curvas de densidad ponderadas (ventas, inventario, despacho)
y la curva propuesta calculada a partir del porcentaje de ventas por intervalo.
Incluye la tabla de optimizacion de inventario y descarga del CSV completo.

Dependencias de datos (carpeta data/):
    ventas_para_elasticidad.csv
    inventario_para_elasticidad.csv
    despacho_para_elasticidad.csv
    recomendacion_rango_precios.csv
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

st.title('Análisis de Sensibilidad al Precio: Curvas de Densidad y Optimización de Inventario')


# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data
def cargar_datos():
    """Carga los cuatro archivos de entrada y los mantiene en cache."""
    ventas       = pd.read_csv('data/ventas_para_elasticidad.csv')
    inventario   = pd.read_csv('data/inventario_para_elasticidad.csv')
    despacho     = pd.read_csv('data/despacho_para_elasticidad.csv')
    recomendacion = pd.read_csv('data/recomendacion_rango_precios.csv')
    return ventas, inventario, despacho, recomendacion


df_ventas, df_inventario, df_despacho, df_recomendacion = cargar_datos()

TIENDAS_EXCLUIDAS = {
    'COBRO A TRANSPORTADORAS',
    'MUESTRAS FISICAS PROVEEDORES SERVICIOS',
    'NOVEDADES TIENDAS ONLINE',
    'PILATOS ACCESORIOS SAN SILVIESTRE',
    'PRESTAMO EMPLEADOS',
    'TEMPORAL 5',
    'PILATOS GRAN PLAZA ALCARAVAN YOPAL',
    'PLANTA PILOTO COSTURA',
    'GARANTIAS EDM',
    'FALTANTES DESPACHOS CEDI',
    'TEMPORAL 2',
    'BODEGA DEVOLUCIONES ONLINE',
    'RECUPERACION DE ROPA',
    'NOVEDADES EN IMPORTACION',
    'BODEGA DE SEGUNDAS OBSOLETAS',
    'CONFE CONCILIACION',
    'SEGUNDAS',
    'MOVIL EVENTO 2',
    'MERCANCIA CONSIGNACION RECIBIDA',
}


# =============================================================================
# SELECTORES
# =============================================================================

marca  = st.selectbox('Selecciona la Marca:',   df_ventas['Marca'].unique())
genero = st.selectbox('Selecciona el Género:',  df_ventas['Genero'].unique())
tipo   = st.selectbox('Selecciona el Tipo:',    df_ventas['Tipo'].unique())
tienda_options = np.append(df_ventas['Tienda'].unique(), 'Todas las tiendas')
tienda = st.selectbox('Selecciona la Tienda:',  tienda_options)


# =============================================================================
# FILTRADO
# =============================================================================

def filtrar(df: pd.DataFrame, col_tienda: str) -> pd.DataFrame:
    """Aplica los filtros de marca, genero, tipo y tienda."""
    mask = (
        (df['Marca']  == marca) &
        (df['Genero'] == genero) &
        (df['Tipo']   == tipo)
    )
    if tienda == 'Todas las tiendas':
        mask &= ~df[col_tienda].isin(TIENDAS_EXCLUIDAS)
    else:
        mask &= df[col_tienda] == tienda
    return df[mask].copy()


df_ventas_f    = filtrar(df_ventas,    'Tienda')
df_inventario_f = filtrar(df_inventario, 'Tienda')
df_despacho_f  = filtrar(df_despacho,  'nombrealmacen')

# Agrupacion por precio.
df_ventas_g = (
    df_ventas_f
    .groupby('Precio_unitario_promedio')['Cantidad']
    .sum()
    .reset_index(name='Cantidad_Ventas')
)
df_inv_g = (
    df_inventario_f
    .groupby('f126_precio')['Cantidad_Inventario']
    .sum()
    .reset_index()
)
df_despacho_g = (
    df_despacho_f
    .groupby('f126_precio')['Despacho']
    .sum()
    .reset_index(name='Cantidad_Despacho')
)

# Validacion: si algun grupo esta vacio no se puede graficar.
datos_insuficientes = (
    df_ventas_g.empty or
    df_inv_g.empty or
    df_despacho_g.empty or
    df_ventas_g['Precio_unitario_promedio'].nunique() < 2 or
    df_inv_g['f126_precio'].nunique() < 2
)

if datos_insuficientes:
    st.warning('No hay datos suficientes para graficar con la selección actual.')
    st.stop()


# =============================================================================
# RANGO DE PRECIO CON FENCE IQR
# =============================================================================

def calcular_fence(s: pd.Series) -> tuple[float, float]:
    """Calcula los limites inferior y superior con criterio IQR."""
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    iqr = q75 - q25
    return max(s.min(), q25 - 1.5 * iqr), min(s.max(), q75 + 1.5 * iqr)


v_low,  v_high  = calcular_fence(df_ventas_g['Precio_unitario_promedio'])
i_low,  i_high  = calcular_fence(df_inv_g['f126_precio'])

precio_min = max(v_low,  i_low)
precio_max = min(v_high, i_high)

if precio_min >= precio_max:
    precio_min = min(df_ventas_g['Precio_unitario_promedio'].min(), df_inv_g['f126_precio'].min())
    precio_max = max(df_ventas_g['Precio_unitario_promedio'].max(), df_inv_g['f126_precio'].max())

precio_unitario_range = np.linspace(precio_min, precio_max, 1000)


# =============================================================================
# CURVAS DE DENSIDAD
# =============================================================================

kde_ventas    = gaussian_kde(df_ventas_g['Precio_unitario_promedio'],
                              weights=df_ventas_g['Cantidad_Ventas'])
kde_inventario = gaussian_kde(df_inv_g['f126_precio'],
                               weights=df_inv_g['Cantidad_Inventario'])
kde_despacho  = gaussian_kde(df_despacho_g['f126_precio'],
                              weights=df_despacho_g['Cantidad_Despacho'])

density_ventas    = kde_ventas(precio_unitario_range)
density_inventario = kde_inventario(precio_unitario_range)
density_despacho  = kde_despacho(precio_unitario_range)


# =============================================================================
# BINS Y PORCENTAJES POR INTERVALO
# =============================================================================

bins_cat = pd.cut(precio_unitario_range, bins=10, right=False).categories

df_ventas_g['Precio_Intervalo']    = pd.cut(df_ventas_g['Precio_unitario_promedio'],
                                            bins=bins_cat, right=False)
df_inv_g['Precio_Intervalo']       = pd.cut(df_inv_g['f126_precio'],
                                            bins=bins_cat, right=False)
df_despacho_g['Precio_Intervalo']  = pd.cut(df_despacho_g['f126_precio'],
                                            bins=bins_cat, right=False)

ventas_por_bin = (
    df_ventas_g.groupby('Precio_Intervalo')['Cantidad_Ventas']
    .sum()
    .reset_index()
)
inv_por_bin = (
    df_inv_g.groupby('Precio_Intervalo')['Cantidad_Inventario']
    .sum()
    .reset_index()
)

total_ventas = ventas_por_bin['Cantidad_Ventas'].sum()
ventas_por_bin['Porcentaje_Ventas'] = ventas_por_bin['Cantidad_Ventas'] / total_ventas

# Punto medio de cada intervalo para la KDE propuesta.
ventas_por_bin['Interval_Midpoint'] = ventas_por_bin['Precio_Intervalo'].apply(
    lambda x: (x.left + x.right) / 2
)

kde_propuesta = gaussian_kde(ventas_por_bin['Interval_Midpoint'],
                              weights=ventas_por_bin['Porcentaje_Ventas'])
density_propuesta = kde_propuesta(precio_unitario_range)


# =============================================================================
# GRAFICO DE CURVAS
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(precio_unitario_range, density_ventas,     color='red',   label='Curva de Densidad de Ventas')
ax.plot(precio_unitario_range, density_inventario, color='blue',  label='Curva de Densidad de Inventario')
ax.plot(precio_unitario_range, density_despacho,   color='black', label='Curva de Densidad de Despacho')
ax.plot(precio_unitario_range, density_propuesta,  color='green', label='Curva de Inventario Propuesta')
ax.set_xlabel('Precio Unitario Promedio')
ax.set_ylabel('Densidad')
ax.set_title('Curvas de Densidad: Ventas, Inventario, Despacho y Propuesta')
ax.legend()
ax.grid(True)
st.pyplot(fig)


# =============================================================================
# INVENTARIO ACTUAL
# =============================================================================

invt_actual = (
    df_inventario_f
    .groupby(['Marca', 'Genero', 'Tipo', 'Tienda'])['Cantidad_Inventario']
    .sum()
    .reset_index()
)
st.write('Inventario Actual:')
st.dataframe(invt_actual)


# =============================================================================
# TABLA DE OPTIMIZACION
# =============================================================================

inv_por_bin['Precio_Intervalo']     = inv_por_bin['Precio_Intervalo'].astype(str)
ventas_por_bin['Precio_Intervalo']  = ventas_por_bin['Precio_Intervalo'].astype(str)

# Normaliza el formato de los strings de intervalo para el merge.
for df_tmp in [inv_por_bin, ventas_por_bin]:
    df_tmp['Precio_Intervalo'] = df_tmp['Precio_Intervalo'].str.replace('.00', '.0', regex=False)

df_combined = pd.merge(
    inv_por_bin,
    ventas_por_bin[['Precio_Intervalo', 'Porcentaje_Ventas']],
    on='Precio_Intervalo',
    how='inner',
)

total_inventario_actual = df_combined['Cantidad_Inventario'].sum()
df_combined['Porcentaje_Actual']             = df_combined['Cantidad_Inventario'] / total_inventario_actual
df_combined['Cantidad_Propuesta_Inventario'] = (df_combined['Porcentaje_Ventas'] * total_inventario_actual).round()

df_final = df_combined[[
    'Precio_Intervalo', 'Cantidad_Inventario',
    'Porcentaje_Actual', 'Porcentaje_Ventas',
    'Cantidad_Propuesta_Inventario',
]]

st.write('Optimización de Inventario:')
st.dataframe(df_final)


# =============================================================================
# DESCARGA
# =============================================================================

csv = df_recomendacion.to_csv(index=False).encode('utf-8')
st.download_button(
    label='Descargar CSV Completo',
    data=csv,
    file_name='recomendaciones_rango_precios.csv',
    mime='text/csv',
)
