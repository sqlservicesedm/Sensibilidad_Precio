"""
4_Sensibilidad_Precio_v3.py

Pagina de Streamlit para el analisis de alineacion oferta vs. demanda
por rango de precio. Muestra cuatro curvas de densidad KDE ponderadas
que cuentan la historia completa del inventario en el periodo:

- Azul:     Inventario actual  — lo que quedo
- Rojo:     Ventas YTD         — lo que se vendio
- Verde:    Despacho YTD       — lo que entro
- Naranja:  Inventario propuesto — referencia si el stock siguiera el patron de ventas

Incluye tabla de optimizacion por intervalo de precio y descarga del
detalle a nivel SKU filtrado por la seleccion actual.

Dependencias de datos (carpeta data/v2/):
    grupos_kde.csv
    inventario_ventas_sku.csv.gz
    despacho_sku.csv.gz

Nomenclatura:
    Formato: marca o canal comercial (Pilatos, Diesel, Superdry, etc.)
    Canal:   tipo de operacion (Lineal, Outlet, Franquicia, Online, etc.)

Notas de interpretacion:
    Las curvas muestran concentracion relativa por rango de precio,
    no cantidades absolutas. Un pico alto significa que ahi ocurre
    la mayor parte de la actividad. Lo importante es comparar la forma
    de las curvas — cuando estan desalineadas, el inventario no esta
    donde ha estado la demanda historica.

    El inventario propuesto redistribuye el stock actual siguiendo el
    patron de ventas historicas. Es una referencia orientativa, no una
    prediccion de demanda futura.

Ultima actualizacion: 2026-06-13
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

st.set_page_config(page_title='Sensibilidad al Precio v3', layout='wide')


# =============================================================================
# CARGA DE DATOS
# =============================================================================

@st.cache_data
def cargar_datos():
    """Carga los archivos de data/v2/ y los mantiene en cache."""
    grupos   = pd.read_csv('data/v2/grupos_kde.csv')
    sku      = pd.read_csv('data/v2/inventario_ventas_sku.csv.gz', compression='gzip')
    despacho = pd.read_csv('data/v2/despacho_sku.csv.gz', compression='gzip')
    return grupos, sku, despacho


grupos, sku, despacho = cargar_datos()


# =============================================================================
# ENCABEZADO Y CONTEXTO DE DATOS
# =============================================================================

st.title('Análisis de Oferta vs. Demanda por Rango de Precio')

# Infiere fechas dinamicamente desde los datos.
periodo_inv = str(grupos['Formato'].iloc[0])  # placeholder — se lee de grupos

# Periodo de inventario desde grupos_kde (campo Precio_Lista_Mediana existe — inferimos desde sku).
periodo_inv_str = 'mayo 2026'  # se sobreescribe abajo si hay columna Periodo

if 'Periodo' in sku.columns:
    periodo_raw = sku['Periodo'].dropna().astype(str).unique()
    if len(periodo_raw) == 1:
        p = periodo_raw[0]
        meses = {
            '01':'enero','02':'febrero','03':'marzo','04':'abril',
            '05':'mayo','06':'junio','07':'julio','08':'agosto',
            '09':'septiembre','10':'octubre','11':'noviembre','12':'diciembre'
        }
        periodo_inv_str = f"{meses.get(p[4:6], p[4:6])} {p[:4]}"

if 'Fecha_Venta' in sku.columns:
    fechas_venta = pd.to_datetime(sku['Fecha_Venta'], errors='coerce').dropna()
    if not fechas_venta.empty:
        fecha_ini = fechas_venta.min().strftime('%d/%m/%Y')
        fecha_fin = fechas_venta.max().strftime('%d/%m/%Y')
        rango_ventas_str = f'{fecha_ini} al {fecha_fin}'
    else:
        rango_ventas_str = 'YTD 2026'
else:
    rango_ventas_str = 'YTD 2026'

with st.expander('ℹ️ ¿Qué representan los datos de esta página?', expanded=True):
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.info(
        f'**Inventario actual**\n\nSnapshot al cierre de **{periodo_inv_str}**. '
        f'Refleja las unidades disponibles en cada tienda en ese momento.'
    )
    col_b.info(
        f'**Ventas YTD**\n\nTransacciones registradas del **{rango_ventas_str}**. '
        f'Representa la demanda realizada en el periodo.'
    )
    col_c.info(
        f'**Despacho YTD**\n\nEntradas de mercancía en tránsito del **{rango_ventas_str}**. '
        f'Aproxima lo que llegó a cada tienda en el periodo.'
    )
    col_d.info(
        f'**Inventario propuesto**\n\nReferencia calculada: redistribuye el stock actual '
        f'siguiendo el patrón de ventas históricas. No es una predicción de demanda futura.'
    )

st.caption(
    'Las curvas muestran concentración relativa por rango de precio, no cantidades absolutas. '
    'Compara la forma de cada curva — cuando están desalineadas, el inventario no está '
    'donde ha estado la demanda histórica.'
)


# =============================================================================
# FILTROS EN CASCADA (SIDEBAR)
# =============================================================================

st.sidebar.header('Filtros')
st.sidebar.caption('Los filtros son dinámicos — cada nivel muestra solo las opciones disponibles según la selección anterior.')

TODOS = '— Todos —'

# Formato (marca comercial: Pilatos, Diesel, etc.)
formatos_disp = [TODOS] + sorted(grupos['Formato'].dropna().unique().tolist())
formato = st.sidebar.multiselect('Formato', options=formatos_disp, default=[TODOS])
if TODOS in formato or not formato:
    mask_formato = pd.Series([True] * len(grupos))
    formato_activo = []
else:
    formato_activo = formato
    mask_formato = grupos['Formato'].isin(formato_activo)

# Canal (tipo de operacion: Lineal, Outlet, etc.)
canales_disp = [TODOS] + sorted(grupos[mask_formato]['Canal'].dropna().unique().tolist())
canal = st.sidebar.multiselect('Canal', options=canales_disp, default=[TODOS])
if TODOS in canal or not canal:
    mask_canal = mask_formato
    canal_activo = []
else:
    canal_activo = canal
    mask_canal = mask_formato & grupos['Canal'].isin(canal_activo)

# Tienda
tiendas_disp = [TODOS] + sorted(grupos[mask_canal]['Tienda'].dropna().unique().tolist())
tienda = st.sidebar.multiselect('Tienda', options=tiendas_disp, default=[TODOS])
if TODOS in tienda or not tienda:
    mask_tienda = mask_canal
    tienda_activo = []
else:
    tienda_activo = tienda
    mask_tienda = mask_canal & grupos['Tienda'].isin(tienda_activo)

# Marca
marcas_disp = [TODOS] + sorted(grupos[mask_tienda]['Marca'].dropna().unique().tolist())
marca = st.sidebar.multiselect('Marca', options=marcas_disp, default=[TODOS])
if TODOS in marca or not marca:
    mask_marca = mask_tienda
    marca_activo = []
else:
    marca_activo = marca
    mask_marca = mask_tienda & grupos['Marca'].isin(marca_activo)

# Genero
generos_disp = [TODOS] + sorted(grupos[mask_marca]['Genero'].dropna().unique().tolist())
genero = st.sidebar.multiselect('Género', options=generos_disp, default=[TODOS])
if TODOS in genero or not genero:
    mask_genero = mask_marca
    genero_activo = []
else:
    genero_activo = genero
    mask_genero = mask_marca & grupos['Genero'].isin(genero_activo)

# Tipo
tipos_disp = [TODOS] + sorted(grupos[mask_genero]['Tipo'].dropna().unique().tolist())
tipo = st.sidebar.multiselect('Tipo', options=tipos_disp, default=[TODOS])
if TODOS in tipo or not tipo:
    mask_tipo = mask_genero
    tipo_activo = []
else:
    tipo_activo = tipo
    mask_tipo = mask_genero & grupos['Tipo'].isin(tipo_activo)


# =============================================================================
# FILTRADO DE DATOS
# =============================================================================

def aplicar_filtro(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los filtros activos a cualquier dataframe con las mismas columnas."""
    mask = pd.Series([True] * len(df))
    if formato_activo:
        mask &= df['Formato'].isin(formato_activo)
    if canal_activo:
        mask &= df['Canal'].isin(canal_activo)
    if tienda_activo:
        mask &= df['Tienda'].isin(tienda_activo)
    if marca_activo:
        mask &= df['Marca'].isin(marca_activo)
    if genero_activo:
        mask &= df['Genero'].isin(genero_activo)
    if tipo_activo:
        mask &= df['Tipo'].isin(tipo_activo)
    return df[mask].copy()


grupos_sel  = grupos[mask_tipo].copy()
sku_sel     = aplicar_filtro(sku)
despacho_sel = aplicar_filtro(despacho)

if grupos_sel.empty or sku_sel.empty:
    st.warning('No hay datos para la selección actual. Ajusta los filtros.')
    st.stop()

# Advertencia si algún grupo no es confiable.
n_no_confiables = (~grupos_sel['flag_confiable']).sum()
if n_no_confiables > 0:
    st.warning(
        f'⚠️ {n_no_confiables} de {len(grupos_sel)} grupos tienen menos de 10 SKUs en inventario. '
        f'Las curvas de esos grupos son menos confiables estadísticamente.'
    )


# =============================================================================
# METRICAS RESUMEN
# =============================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric('Unidades en inventario', f'{grupos_sel["Cantidad_Inventario"].sum():,.0f}')
col2.metric('Unidades vendidas YTD',  f'{grupos_sel["Cantidad_Ventas"].sum():,.0f}')
col3.metric('SKUs con ventas',        f'{grupos_sel["n_skus_con_ventas"].sum():,} de {grupos_sel["n_skus_inventario"].sum():,}')
col4.metric('Grupos analizados',      f'{len(grupos_sel):,}')


# =============================================================================
# PREPARACION DE DATOS PARA KDE
# =============================================================================

def calcular_fence(s: pd.Series) -> tuple[float, float]:
    """Calcula los limites de rango visible con criterio IQR."""
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    iqr = q75 - q25
    return max(s.min(), q25 - 1.5 * iqr), min(s.max(), q75 + 1.5 * iqr)


inv_agg   = sku_sel.groupby('Precio_Lista')['Cantidad_Inventario'].sum().reset_index()
ventas_agg = (
    sku_sel[sku_sel['flag_tiene_ventas']]
    .groupby('Precio_Lista')['Cantidad_Ventas']
    .sum()
    .reset_index()
)
desp_agg  = despacho_sel.groupby('Precio_Lista')['Despacho'].sum().reset_index()

datos_insuficientes = (
    inv_agg.empty or inv_agg['Precio_Lista'].nunique() < 2 or
    ventas_agg.empty or ventas_agg['Precio_Lista'].nunique() < 2
)

if datos_insuficientes:
    st.warning('No hay suficientes datos de precio para construir las curvas con la selección actual.')
    st.stop()

# Rango de precio con fence IQR sobre el conjunto combinado.
todos_precios = pd.concat(
    [inv_agg['Precio_Lista'], ventas_agg['Precio_Lista']] +
    ([desp_agg['Precio_Lista']] if not desp_agg.empty else [])
).dropna()

p_min, p_max = calcular_fence(todos_precios)
if p_min >= p_max:
    p_min, p_max = todos_precios.min(), todos_precios.max()

rango = np.linspace(p_min, p_max, 1000)


# =============================================================================
# CURVAS KDE
# =============================================================================

st.subheader('Curvas de Densidad')

fig, ax = plt.subplots(figsize=(12, 6))

kde_inv = gaussian_kde(inv_agg['Precio_Lista'], weights=inv_agg['Cantidad_Inventario'])
ax.plot(rango, kde_inv(rango), color='#1f77b4', linewidth=2.5,
        label='Inventario actual — lo que quedó')

kde_ventas = gaussian_kde(ventas_agg['Precio_Lista'], weights=ventas_agg['Cantidad_Ventas'])
ax.plot(rango, kde_ventas(rango), color='#d62728', linewidth=2.5,
        label='Ventas YTD — lo que se vendió')

if not desp_agg.empty and desp_agg['Precio_Lista'].nunique() >= 2:
    kde_desp = gaussian_kde(desp_agg['Precio_Lista'], weights=desp_agg['Despacho'])
    ax.plot(rango, kde_desp(rango), color='#2ca02c', linewidth=2, linestyle='--',
            label='Despacho YTD — lo que entró')

# Curva propuesta: bins sobre rango combinado para coherencia.
precio_combinado = pd.concat([inv_agg['Precio_Lista'], ventas_agg['Precio_Lista']]).dropna()
try:
    bins_compartidos = pd.qcut(precio_combinado, q=10, duplicates='drop').cat.categories
    ventas_agg['Intervalo'] = pd.cut(ventas_agg['Precio_Lista'], bins=bins_compartidos, include_lowest=True)
    vpb = (
        ventas_agg.groupby('Intervalo', observed=True)['Cantidad_Ventas']
        .sum().reset_index()
    )
    vpb = vpb[vpb['Cantidad_Ventas'] > 0]
    if len(vpb) >= 2:
        vpb['pct']      = vpb['Cantidad_Ventas'] / vpb['Cantidad_Ventas'].sum()
        vpb['midpoint'] = vpb['Intervalo'].apply(lambda x: (x.left + x.right) / 2)
        kde_prop = gaussian_kde(vpb['midpoint'], weights=vpb['pct'])
        ax.plot(rango, kde_prop(rango), color='#ff7f0e', linewidth=2, linestyle=':',
                label='Inventario propuesto — referencia')
except Exception:
    pass

titulo_filtros = ' | '.join(filter(None, [
    ', '.join(formato_activo) if formato_activo else 'Todos los formatos',
    ', '.join(canal_activo)   if canal_activo   else None,
    ', '.join(tienda_activo)  if tienda_activo  else None,
    ', '.join(marca_activo)   if marca_activo   else None,
    ', '.join(genero_activo)  if genero_activo  else None,
    ', '.join(tipo_activo)    if tipo_activo    else None,
]))

ax.set_xlabel('Precio Lista (COP)', fontsize=11)
ax.set_ylabel('Densidad relativa', fontsize=11)
ax.set_title(f'Oferta vs. Demanda por Rango de Precio\n{titulo_filtros}', fontsize=12, fontweight='bold')
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
st.caption(
    'Compara el porcentaje del inventario actual vs. el porcentaje de ventas en cada rango de precio. '
    'El inventario propuesto redistribuye el stock total según el patrón de ventas históricas.'
)

try:
    # Bins calculados sobre el rango combinado — garantiza coherencia entre inventario y ventas.
    precio_combinado_tabla = pd.concat([
        inv_agg['Precio_Lista'], ventas_agg['Precio_Lista']
    ]).dropna()
    bins_tabla = pd.qcut(precio_combinado_tabla, q=10, duplicates='drop').cat.categories

    inv_agg['Intervalo']    = pd.cut(inv_agg['Precio_Lista'],    bins=bins_tabla, include_lowest=True)
    ventas_agg['Intervalo'] = pd.cut(ventas_agg['Precio_Lista'], bins=bins_tabla, include_lowest=True)

    inv_bin = inv_agg.groupby('Intervalo', observed=True)['Cantidad_Inventario'].sum().reset_index()
    ven_bin = ventas_agg.groupby('Intervalo', observed=True)['Cantidad_Ventas'].sum().reset_index()

    df_tabla = inv_bin.merge(ven_bin, on='Intervalo', how='left')
    df_tabla['Cantidad_Ventas'] = df_tabla['Cantidad_Ventas'].fillna(0)

    total_inv    = df_tabla['Cantidad_Inventario'].sum()
    total_ventas = df_tabla['Cantidad_Ventas'].sum()

    df_tabla['% Inventario actual']  = (df_tabla['Cantidad_Inventario'] / total_inv * 100).round(1)
    df_tabla['% Ventas YTD']         = (df_tabla['Cantidad_Ventas'] / total_ventas * 100).round(1) if total_ventas > 0 else 0.0
    df_tabla['Inventario propuesto'] = (df_tabla['% Ventas YTD'] / 100 * total_inv).round(0).astype(int)
    df_tabla['Diferencia']           = df_tabla['Inventario propuesto'] - df_tabla['Cantidad_Inventario'].astype(int)
    df_tabla['Intervalo']            = df_tabla['Intervalo'].astype(str)

    df_tabla = df_tabla.rename(columns={
        'Intervalo':           'Rango de Precio',
        'Cantidad_Inventario': 'Inventario actual',
        'Cantidad_Ventas':     'Ventas YTD',
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
# DETALLE Y DESCARGA A NIVEL SKU
# =============================================================================

st.subheader('Detalle a Nivel SKU')

columnas_mostrar = [
    'Codigo_Barras', 'Referencia', 'Talla', 'Color',
    'Marca', 'MarcaLinea', 'Genero', 'Categoria', 'Tipo',
    'Formato', 'Canal', 'Tienda',
    'Precio_Lista', 'Precio_Unitario',
    'Cantidad_Inventario', 'Cantidad_Ventas',
    'Valor_Pagado', 'Descuento_Total',
    'flag_tiene_ventas',
]

columnas_existentes = [c for c in columnas_mostrar if c in sku_sel.columns]

st.dataframe(
    sku_sel[columnas_existentes].sort_values('Cantidad_Inventario', ascending=False),
    use_container_width=True,
    hide_index=True,
)

csv_bytes = sku_sel[columnas_existentes].to_csv(index=False).encode('utf-8')
nombre_archivo = 'sku_detalle.csv'
if formato_activo:
    nombre_archivo = f'sku_{"_".join(formato_activo)}.csv'.replace(' ', '_')

st.download_button(
    label='⬇️ Descargar detalle SKU (CSV)',
    data=csv_bytes,
    file_name=nombre_archivo,
    mime='text/csv',
)

