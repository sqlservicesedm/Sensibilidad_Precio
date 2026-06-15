"""
4_Sensibilidad_Precio_v3.py

Pagina de Streamlit para el analisis de alineacion oferta vs. demanda
por rango de precio. Muestra cuatro curvas de densidad KDE ponderadas
que cuentan la historia completa del inventario en el periodo:

- Negro:  Despacho YTD      — lo que entro
- Rojo:   Ventas YTD        — lo que se vendio
- Azul:   Inventario actual — lo que quedo
- Verde:  Inventario propuesto — como deberia estar si siguiera el patron de ventas

Incluye tabla de optimizacion por intervalo de precio y descarga del
detalle a nivel SKU filtrado por la seleccion actual.

Dependencias de datos (carpeta data/v2/):
    grupos_kde.csv
    inventario_ventas_sku.csv
    despacho_sku.csv

Notas de interpretacion:
    Las curvas muestran concentracion relativa, no cantidades absolutas.
    Un pico alto significa que ahi ocurre la mayor parte de la actividad
    en ese dataset. Lo importante es comparar la forma de las curvas —
    cuando estan desalineadas, el inventario no esta donde ha estado
    la demanda historica.

    El inventario propuesto redistribuye el stock actual siguiendo el
    patron de ventas historicas. Es una referencia, no una prediccion
    de demanda futura.

Ultima actualizacion: 2026-06-13
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

st.set_page_config(page_title='Sensibilidad al Precio v3', layout='wide')
st.title('Análisis de Oferta vs. Demanda por Rango de Precio')
st.caption(
    'Las curvas muestran concentración relativa por rango de precio, no cantidades absolutas. '
    'Compara la forma de cada curva para identificar desalineaciones entre '
    'lo que entró, lo que se vendió y lo que quedó en inventario.'
)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data
def cargar_datos():
    """Carga los tres archivos de data/v2/ y los mantiene en cache."""
    grupos    = pd.read_csv('data/v2/grupos_kde.csv')
    sku       = pd.read_csv('data/v2/inventario_ventas_sku.csv')
    despacho  = pd.read_csv('data/v2/despacho_sku.csv')
    return grupos, sku, despacho


grupos, sku, despacho = cargar_datos()


# =============================================================================
# FILTROS EN CASCADA (SIDEBAR)
# =============================================================================

st.sidebar.header('Filtros')

canal = st.sidebar.selectbox(
    'Canal',
    sorted(grupos['Canal'].dropna().unique())
)

formatos_disp = sorted(grupos[grupos['Canal'] == canal]['Formato'].dropna().unique())
formato = st.sidebar.selectbox('Formato', formatos_disp)

tiendas_disp = sorted(
    grupos[(grupos['Canal'] == canal) & (grupos['Formato'] == formato)]
    ['Tienda'].dropna().unique()
)
tienda = st.sidebar.selectbox('Tienda', tiendas_disp)

marcas_disp = sorted(
    grupos[
        (grupos['Canal'] == canal) &
        (grupos['Formato'] == formato) &
        (grupos['Tienda'] == tienda)
    ]['Marca'].dropna().unique()
)
marca = st.sidebar.selectbox('Marca', marcas_disp)

generos_disp = sorted(
    grupos[
        (grupos['Canal'] == canal) &
        (grupos['Formato'] == formato) &
        (grupos['Tienda'] == tienda) &
        (grupos['Marca'] == marca)
    ]['Genero'].dropna().unique()
)
genero = st.sidebar.selectbox('Género', generos_disp)

tipos_disp = sorted(
    grupos[
        (grupos['Canal'] == canal) &
        (grupos['Formato'] == formato) &
        (grupos['Tienda'] == tienda) &
        (grupos['Marca'] == marca) &
        (grupos['Genero'] == genero)
    ]['Tipo'].dropna().unique()
)
tipo = st.sidebar.selectbox('Tipo', tipos_disp)


# =============================================================================
# FILTRADO
# =============================================================================

grupo_sel = grupos[
    (grupos['Canal']   == canal) &
    (grupos['Formato'] == formato) &
    (grupos['Tienda']  == tienda) &
    (grupos['Marca']   == marca) &
    (grupos['Genero']  == genero) &
    (grupos['Tipo']    == tipo)
]

sku_sel = sku[
    (sku['Canal']   == canal) &
    (sku['Formato'] == formato) &
    (sku['Tienda']  == tienda) &
    (sku['Marca']   == marca) &
    (sku['Genero']  == genero) &
    (sku['Tipo']    == tipo)
]

desp_sel = despacho[
    (despacho['Canal']   == canal) &
    (despacho['Formato'] == formato) &
    (despacho['Tienda']  == tienda) &
    (despacho['Marca']   == marca) &
    (despacho['Genero']  == genero) &
    (despacho['Tipo']    == tipo)
]

# Validacion de datos suficientes.
if grupo_sel.empty or sku_sel.empty:
    st.warning('No hay datos para la selección actual.')
    st.stop()

if not grupo_sel['flag_confiable'].values[0]:
    st.warning(
        f'⚠️ Este grupo tiene solo {grupo_sel["n_registros_inv"].values[0]} SKUs en inventario. '
        f'Se recomienda al menos {10} para que las curvas sean estadísticamente confiables. '
        f'Interpreta los resultados con cautela.'
    )

# Metricas del grupo
col1, col2, col3, col4 = st.columns(4)
col1.metric('Unidades en inventario', f'{grupo_sel["Cantidad_Inventario"].values[0]:,.0f}')
col2.metric('Unidades vendidas YTD',  f'{grupo_sel["Cantidad_Ventas"].values[0]:,.0f}')
col3.metric('SKUs con ventas',        f'{grupo_sel["n_skus_con_ventas"].values[0]:,} de {grupo_sel["n_skus_inventario"].values[0]:,}')
col4.metric('SKUs sin ventas',        f'{grupo_sel["n_skus_inventario"].values[0] - grupo_sel["n_skus_con_ventas"].values[0]:,}')


# =============================================================================
# PREPARACION DE DATOS PARA KDE
# =============================================================================

def calcular_fence(s: pd.Series) -> tuple[float, float]:
    """Calcula los limites de rango visible con criterio IQR."""
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    iqr = q75 - q25
    return max(s.min(), q25 - 1.5 * iqr), min(s.max(), q75 + 1.5 * iqr)


# Agrega por precio para KDE.
inv_agg  = sku_sel.groupby('Precio_Lista')['Cantidad_Inventario'].sum().reset_index()
desp_agg = desp_sel.groupby('Precio_Lista')['Despacho'].sum().reset_index()

# Ventas raw desde sku (ya filtrado).
ventas_agg = (
    sku_sel[sku_sel['flag_tiene_ventas']]
    .groupby('Precio_Lista')['Cantidad_Ventas']
    .sum()
    .reset_index()
)

datos_insuficientes = (
    inv_agg.empty or
    inv_agg['Precio_Lista'].nunique() < 2 or
    ventas_agg.empty or
    ventas_agg['Precio_Lista'].nunique() < 2
)

if datos_insuficientes:
    st.warning('No hay suficientes datos de precio para construir las curvas.')
    st.stop()

# Rango de precio con fence IQR.
todos_precios = pd.concat([
    inv_agg['Precio_Lista'],
    ventas_agg['Precio_Lista'],
] + ([desp_agg['Precio_Lista']] if not desp_agg.empty else [])).dropna()

p_min, p_max = calcular_fence(todos_precios)
if p_min >= p_max:
    p_min, p_max = todos_precios.min(), todos_precios.max()

rango = np.linspace(p_min, p_max, 1000)


# =============================================================================
# CURVAS KDE
# =============================================================================

st.subheader('Curvas de Densidad: Lo que entró, lo que se vendió y lo que quedó')

fig, ax = plt.subplots(figsize=(12, 6))

# Curva inventario actual (azul).
kde_inv = gaussian_kde(inv_agg['Precio_Lista'], weights=inv_agg['Cantidad_Inventario'])
ax.plot(rango, kde_inv(rango), color='#1f77b4', linewidth=2.5, label='Inventario actual — lo que quedó')

# Curva ventas (rojo).
kde_ventas = gaussian_kde(ventas_agg['Precio_Lista'], weights=ventas_agg['Cantidad_Ventas'])
ax.plot(rango, kde_ventas(rango), color='#d62728', linewidth=2.5, label='Ventas YTD — lo que se vendió')

# Curva despacho (negro) — solo si hay datos suficientes.
if not desp_agg.empty and desp_agg['Precio_Lista'].nunique() >= 2:
    kde_desp = gaussian_kde(desp_agg['Precio_Lista'], weights=desp_agg['Despacho'])
    ax.plot(rango, kde_desp(rango), color='#2ca02c', linewidth=2, linestyle='--', label='Despacho YTD — lo que entró')

# Curva propuesta (verde): redistribucion del inventario segun patron de ventas.
ventas_agg['Intervalo'] = pd.qcut(ventas_agg['Precio_Lista'], q=10, duplicates='drop')
vpb = (
    ventas_agg
    .groupby('Intervalo', observed=True)['Cantidad_Ventas']
    .sum()
    .reset_index()
)
if len(vpb) >= 2:
    vpb['pct']      = vpb['Cantidad_Ventas'] / vpb['Cantidad_Ventas'].sum()
    vpb['midpoint'] = vpb['Intervalo'].apply(lambda x: (x.left + x.right) / 2)
    kde_prop = gaussian_kde(vpb['midpoint'], weights=vpb['pct'])
    ax.plot(rango, kde_prop(rango), color='#ff7f0e', linewidth=2, linestyle=':', label='Inventario propuesto — referencia')

ax.set_xlabel('Precio Lista (COP)', fontsize=11)
ax.set_ylabel('Densidad relativa', fontsize=11)
ax.set_title(
    f'Oferta vs. Demanda por Rango de Precio\n{marca} | {genero} | {tipo} | {tienda}',
    fontsize=13, fontweight='bold'
)
ax.set_yticklabels([])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
st.pyplot(fig)


# =============================================================================
# TABLA DE OPTIMIZACION
# =============================================================================

st.subheader('Optimización por Intervalo de Precio')

if not grupo_sel['flag_confiable'].values[0]:
    st.info('Tabla no disponible — grupo con menos de 10 SKUs en inventario.')
else:
    try:
        inv_agg['Intervalo'] = pd.qcut(inv_agg['Precio_Lista'], q=10, duplicates='drop')
        ventas_agg2 = sku_sel[sku_sel['flag_tiene_ventas']].groupby('Precio_Lista')['Cantidad_Ventas'].sum().reset_index()
        ventas_agg2['Intervalo'] = pd.qcut(ventas_agg2['Precio_Lista'], q=10, duplicates='drop')

        inv_bin = (
            inv_agg.groupby('Intervalo', observed=True)['Cantidad_Inventario']
            .sum().reset_index()
        )
        ven_bin = (
            ventas_agg2.groupby('Intervalo', observed=True)['Cantidad_Ventas']
            .sum().reset_index()
        )

        df_tabla = inv_bin.merge(ven_bin, on='Intervalo', how='left')
        df_tabla['Cantidad_Ventas'] = df_tabla['Cantidad_Ventas'].fillna(0)

        total_inv    = df_tabla['Cantidad_Inventario'].sum()
        total_ventas = df_tabla['Cantidad_Ventas'].sum()

        df_tabla['% Inventario actual'] = (df_tabla['Cantidad_Inventario'] / total_inv * 100).round(1)
        df_tabla['% Ventas YTD']        = (df_tabla['Cantidad_Ventas'] / total_ventas * 100).round(1) if total_ventas > 0 else 0
        df_tabla['Inventario propuesto'] = (df_tabla['% Ventas YTD'] / 100 * total_inv).round(0).astype(int)
        df_tabla['Diferencia']           = df_tabla['Inventario propuesto'] - df_tabla['Cantidad_Inventario'].astype(int)

        # Validacion: porcentajes suman 100%.
        suma_pct_inv    = df_tabla['% Inventario actual'].sum()
        suma_pct_ventas = df_tabla['% Ventas YTD'].sum()
        if abs(suma_pct_inv - 100) > 1 or abs(suma_pct_ventas - 100) > 1:
            st.warning('⚠️ Los porcentajes no suman 100% — puede haber intervalos sin datos en alguna fuente.')

        df_tabla['Intervalo'] = df_tabla['Intervalo'].astype(str)
        df_tabla = df_tabla.rename(columns={
            'Intervalo':            'Rango de Precio',
            'Cantidad_Inventario':  'Inventario actual',
            'Cantidad_Ventas':      'Ventas YTD',
        })

        st.dataframe(
            df_tabla[[
                'Rango de Precio', 'Inventario actual', '% Inventario actual',
                'Ventas YTD', '% Ventas YTD', 'Inventario propuesto', 'Diferencia'
            ]],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.warning(f'No se pudo construir la tabla de optimización: {e}')


# =============================================================================
# DESCARGA A NIVEL SKU
# =============================================================================

st.subheader('Detalle a Nivel SKU')

st.dataframe(
    sku_sel[[
        'Codigo_Barras', 'Referencia', 'Talla', 'Color', 'Precio_Lista',
        'Cantidad_Inventario', 'Cantidad_Ventas', 'Valor_Pagado',
        'Descuento_Total', 'flag_tiene_ventas',
    ]].sort_values('Cantidad_Inventario', ascending=False),
    use_container_width=True,
    hide_index=True,
)

csv = sku_sel.to_csv(index=False).encode('utf-8')
st.download_button(
    label='⬇️ Descargar detalle SKU (CSV)',
    data=csv,
    file_name=f'sku_{canal}_{tienda}_{marca}_{genero}_{tipo}.csv'.replace(' ', '_'),
    mime='text/csv',
)
