"""
4_Sensibilidad_Precio_v3.py

Pagina de Streamlit para el analisis de alineacion entre demanda pasada
e inventario actual por rango de precio.

Muestra cuatro curvas de densidad KDE ponderadas:
- Azul:    Inventario actual   — lo que quedo al cierre del periodo
- Rojo:    Demanda pasada      — lo que se vendio en el periodo YTD
- Verde:   Despacho YTD        — lo que entro al punto de venta en el periodo
- Naranja: Inventario propuesto — referencia si el stock siguiera el patron de demanda

Nomenclatura usada en los filtros:
    Canal:   marca o grupo comercial (Pilatos, Diesel, Superdry, MFG, etc.)
             Columna en datos: Canal
    Formato: tipo de operacion (Lineal, Outlet, Mixta, Franquicia, Online)
             Columna en datos: Formato

Dependencias de datos (carpeta data/v2/):
    grupos_kde.csv              — agregado liviano para curvas y tabla
    inventario_ventas_sku.csv.gz — detalle SKU para descarga
    despacho_sku.csv.gz         — detalle despacho para curva KDE

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
def cargar_grupos():
    """Carga grupos_kde.csv — archivo liviano para curvas y tabla."""
    return pd.read_csv('data/v2/grupos_kde.csv')


@st.cache_data
def cargar_sku():
    """Carga inventario_ventas_sku.csv.gz — solo para descarga."""
    cols = [
        'Codigo_Barras', 'Referencia', 'Talla', 'Color',
        'Marca', 'MarcaLinea', 'Genero', 'Categoria', 'Tipo',
        'Canal', 'Formato', 'Tienda',
        'Precio_Lista', 'Precio_Unitario',
        'Cantidad_Inventario', 'Cantidad_Ventas',
        'Valor_Pagado', 'Descuento_Total', 'flag_tiene_ventas',
    ]
    df = pd.read_csv('data/v2/inventario_ventas_sku.csv.gz', compression='gzip')
    return df[[c for c in cols if c in df.columns]]


@st.cache_data
def cargar_despacho():
    """Carga despacho_sku.csv.gz — solo columnas necesarias para KDE."""
    return pd.read_csv(
        'data/v2/despacho_sku.csv.gz',
        compression='gzip',
        usecols=['Canal', 'Formato', 'Tienda', 'Marca', 'Genero', 'Tipo',
                 'Precio_Lista', 'Despacho'],
    )


grupos   = cargar_grupos()
sku      = cargar_sku()
despacho = cargar_despacho()


# =============================================================================
# ENCABEZADO
# =============================================================================

st.title('Análisis de Demanda Pasada vs. Inventario Actual por Rango de Precio')

# Expander: acerca del reporte
with st.expander('ℹ️ Acerca de este reporte y glosario de términos', expanded=False):
    st.markdown("""
**¿Qué muestra este reporte?**
Compara la distribución del inventario actual con la distribución de la demanda histórica
por rango de precio. El objetivo es identificar si el stock disponible hoy está concentrado
en los rangos de precio donde históricamente ha habido mayor demanda.

**Importante:** se comparan dos períodos distintos.
- La **demanda pasada** refleja lo que se vendió en un período histórico (YTD).
- El **inventario actual** es una foto del stock disponible al cierre del último período.
No se puede concluir que habrá quiebre futuro — solo que hay desalineación histórica.

**Glosario:**
- **Canal:** marca o grupo comercial al que pertenece la tienda (Pilatos, Diesel, Superdry, MFG, etc.)
- **Formato:** tipo de operación de la tienda (Lineal, Outlet, Mixta, Franquicia, Online)
- **Inventario propuesto:** referencia calculada que redistribuye el stock actual siguiendo
  el patrón de demanda histórica. No es una predicción de demanda futura.
- **Densidad relativa:** las curvas muestran concentración proporcional, no cantidades absolutas.
  Un pico alto significa que ahí ocurre la mayor parte de la actividad en ese conjunto de datos.
""")

# Expander: rangos de fechas
with st.expander('📅 Rangos de datos y fechas de actualización', expanded=False):
    # Infiere periodo de inventario desde grupos.
    if 'Precio_Lista_Mediana' in grupos.columns:
        periodo_inv = 'mayo 2026'  # fallback
    else:
        periodo_inv = 'último período disponible'

    # Infiere rango de ventas desde sku si tiene Fecha_Venta.
    if 'Fecha_Venta' in sku.columns:
        fechas = pd.to_datetime(sku['Fecha_Venta'], errors='coerce').dropna()
        if not fechas.empty:
            rango_ventas = f"{fechas.min().strftime('%d/%m/%Y')} — {fechas.max().strftime('%d/%m/%Y')}"
        else:
            rango_ventas = 'YTD 2026'
    else:
        rango_ventas = 'YTD 2026'

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown(f"""
**Inventario actual**
Snapshot al cierre de **{periodo_inv}**.
Refleja las unidades disponibles en cada punto de venta en ese momento.
""")
    with col_f2:
        st.markdown(f"""
**Demanda pasada (ventas y despacho)**
Período analizado: **{rango_ventas}**.
Acumulado desde el inicio del año hasta el último día del mes anterior.
""")
    with col_f3:
        st.markdown("""
**Actualización**
Los datos se actualizan el primer día de cada mes.
Al subir nuevos archivos a la carpeta `data/v2/`, este reporte
se refresca automáticamente con el nuevo período.
""")


# =============================================================================
# FILTROS EN CASCADA (SIDEBAR)
# =============================================================================

st.sidebar.header('Filtros')
st.sidebar.caption(
    'Cada filtro muestra solo las opciones disponibles '
    'según las selecciones anteriores.'
)

TODOS = '— Todos —'


def multiselect_con_todos(label, opciones, key):
    """Selector multiple con opcion Todos."""
    sel = st.sidebar.multiselect(label, [TODOS] + opciones, default=[TODOS], key=key)
    if TODOS in sel or not sel:
        return []
    return sel


# Canal (Pilatos, Diesel, etc.)
canales_disp = sorted(grupos['Canal'].dropna().unique().tolist())
canal_sel = multiselect_con_todos('Canal', canales_disp, 'canal')
mask_canal = grupos['Canal'].isin(canal_sel) if canal_sel else pd.Series([True] * len(grupos))

# Formato (Lineal, Outlet, etc.)
formatos_disp = sorted(grupos[mask_canal]['Formato'].dropna().unique().tolist())
formato_sel = multiselect_con_todos('Formato', formatos_disp, 'formato')
mask_formato = mask_canal & (grupos['Formato'].isin(formato_sel) if formato_sel else pd.Series([True] * len(grupos)))

# Tienda
tiendas_disp = sorted(grupos[mask_formato]['Tienda'].dropna().unique().tolist())
tienda_sel = multiselect_con_todos('Tienda', tiendas_disp, 'tienda')
mask_tienda = mask_formato & (grupos['Tienda'].isin(tienda_sel) if tienda_sel else pd.Series([True] * len(grupos)))

# Marca
marcas_disp = sorted(grupos[mask_tienda]['Marca'].dropna().unique().tolist())
marca_sel = multiselect_con_todos('Marca', marcas_disp, 'marca')
mask_marca = mask_tienda & (grupos['Marca'].isin(marca_sel) if marca_sel else pd.Series([True] * len(grupos)))

# Genero
generos_disp = sorted(grupos[mask_marca]['Genero'].dropna().unique().tolist())
genero_sel = multiselect_con_todos('Género', generos_disp, 'genero')
mask_genero = mask_marca & (grupos['Genero'].isin(genero_sel) if genero_sel else pd.Series([True] * len(grupos)))

# Tipo
tipos_disp = sorted(grupos[mask_genero]['Tipo'].dropna().unique().tolist())
tipo_sel = multiselect_con_todos('Tipo', tipos_disp, 'tipo')
mask_tipo = mask_genero & (grupos['Tipo'].isin(tipo_sel) if tipo_sel else pd.Series([True] * len(grupos)))

grupos_sel = grupos[mask_tipo].copy()


# =============================================================================
# VALIDACION
# =============================================================================

if grupos_sel.empty:
    st.warning('No hay datos para la selección actual. Ajusta los filtros.')
    st.stop()


# =============================================================================
# METRICAS RESUMEN
# =============================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric('Unidades en inventario', f'{grupos_sel["Cantidad_Inventario"].sum():,.0f}')
col2.metric('Unidades demanda pasada', f'{grupos_sel["Cantidad_Ventas"].sum():,.0f}')
col3.metric(
    'SKUs con demanda',
    f'{grupos_sel["n_skus_con_ventas"].sum():,} / {grupos_sel["n_skus_inventario"].sum():,}'
)
col4.metric('SKUs sin demanda en el periodo', f'{(grupos_sel["n_skus_inventario"] - grupos_sel["n_skus_con_ventas"]).sum():,}')


# =============================================================================
# PREPARACION PARA KDE — desde grupos_kde (liviano)
# =============================================================================

def calcular_fence(s: pd.Series) -> tuple[float, float]:
    """Calcula los limites de rango visible con criterio IQR."""
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    iqr = q75 - q25
    return max(s.min(), q25 - 1.5 * iqr), min(s.max(), q75 + 1.5 * iqr)


# Agrega desde grupos_sel usando Precio_Lista_Mediana como precio representativo.
inv_agg = (
    grupos_sel
    .groupby('Precio_Lista_Mediana')
    .agg(Cantidad_Inventario=('Cantidad_Inventario', 'sum'))
    .reset_index()
    .rename(columns={'Precio_Lista_Mediana': 'Precio_Lista'})
)

ventas_agg = (
    grupos_sel[grupos_sel['flag_tiene_ventas']]
    .groupby('Precio_Lista_Mediana')
    .agg(Cantidad_Ventas=('Cantidad_Ventas', 'sum'))
    .reset_index()
    .rename(columns={'Precio_Lista_Mediana': 'Precio_Lista'})
)

# Despacho — filtra desde el CSV de despacho.
def filtrar_df(df):
    mask = pd.Series([True] * len(df))
    if canal_sel:   mask &= df['Canal'].isin(canal_sel)
    if formato_sel: mask &= df['Formato'].isin(formato_sel)
    if tienda_sel:  mask &= df['Tienda'].isin(tienda_sel)
    if marca_sel:   mask &= df['Marca'].isin(marca_sel)
    if genero_sel:  mask &= df['Genero'].isin(genero_sel)
    if tipo_sel:    mask &= df['Tipo'].isin(tipo_sel)
    return df[mask]

desp_sel = filtrar_df(despacho)
desp_agg = (
    desp_sel.groupby('Precio_Lista')['Despacho'].sum().reset_index()
    if not desp_sel.empty else pd.DataFrame(columns=['Precio_Lista', 'Despacho'])
)

datos_insuficientes = (
    inv_agg.empty or inv_agg['Precio_Lista'].nunique() < 2 or
    ventas_agg.empty or ventas_agg['Precio_Lista'].nunique() < 2
)

if datos_insuficientes:
    st.warning(
        'No hay suficientes datos de precio para construir las curvas. '
        'Intenta ampliar los filtros o seleccionar un grupo con más SKUs.'
    )
    st.stop()

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

st.subheader('Curvas de Densidad por Rango de Precio')

# Titulo dinamico segun filtros activos.
partes = []
if canal_sel:   partes.append(', '.join(canal_sel))
if formato_sel: partes.append(', '.join(formato_sel))
if tienda_sel:  partes.append(', '.join(tienda_sel))
if marca_sel:   partes.append(', '.join(marca_sel))
if genero_sel:  partes.append(', '.join(genero_sel))
if tipo_sel:    partes.append(', '.join(tipo_sel))
titulo = ' | '.join(partes) if partes else 'Todos los grupos'

fig, ax = plt.subplots(figsize=(12, 6))

kde_inv = gaussian_kde(inv_agg['Precio_Lista'], weights=inv_agg['Cantidad_Inventario'])
ax.plot(rango, kde_inv(rango), color='#1f77b4', linewidth=2.5,
        label='Inventario actual — lo que quedó')

kde_ventas = gaussian_kde(ventas_agg['Precio_Lista'], weights=ventas_agg['Cantidad_Ventas'])
ax.plot(rango, kde_ventas(rango), color='#d62728', linewidth=2.5,
        label='Demanda pasada — lo que se vendió')

if not desp_agg.empty and desp_agg['Precio_Lista'].nunique() >= 2:
    kde_desp = gaussian_kde(desp_agg['Precio_Lista'], weights=desp_agg['Despacho'])
    ax.plot(rango, kde_desp(rango), color='#2ca02c', linewidth=2, linestyle='--',
            label='Despacho YTD — lo que entró')

# Curva propuesta: bins sobre rango combinado.
precio_combinado = pd.concat([inv_agg['Precio_Lista'], ventas_agg['Precio_Lista']]).dropna()
try:
    bins_prop = pd.qcut(precio_combinado, q=10, duplicates='drop').cat.categories
    ventas_prop = ventas_agg.copy()
    ventas_prop['Intervalo'] = pd.cut(ventas_prop['Precio_Lista'], bins=bins_prop, include_lowest=True)
    vpb = ventas_prop.groupby('Intervalo', observed=True)['Cantidad_Ventas'].sum().reset_index()
    vpb = vpb[vpb['Cantidad_Ventas'] > 0]
    if len(vpb) >= 2:
        vpb['pct']      = vpb['Cantidad_Ventas'] / vpb['Cantidad_Ventas'].sum()
        vpb['midpoint'] = vpb['Intervalo'].apply(lambda x: (x.left + x.right) / 2)
        kde_prop = gaussian_kde(vpb['midpoint'], weights=vpb['pct'])
        ax.plot(rango, kde_prop(rango), color='#ff7f0e', linewidth=2, linestyle=':',
                label='Inventario propuesto — referencia')
except Exception:
    pass

ax.set_xlabel('Precio Lista (COP)', fontsize=11)
ax.set_ylabel('Densidad relativa', fontsize=11)
ax.set_title(f'Demanda Pasada vs. Inventario Actual\n{titulo}', fontsize=12, fontweight='bold')
ax.set_yticklabels([])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
st.pyplot(fig)


# =============================================================================
# TABLA DE OPTIMIZACION
# =============================================================================

st.subheader('Distribución por Rango de Precio')
st.caption(
    'Compara qué porcentaje del inventario actual está en cada rango de precio '
    'versus qué porcentaje de la demanda pasada ocurrió en ese rango. '
    'El inventario propuesto redistribuye el stock total siguiendo el patrón de demanda histórica.'
)

try:
    precio_combinado_tabla = pd.concat([
        inv_agg['Precio_Lista'], ventas_agg['Precio_Lista']
    ]).dropna()
    bins_tabla = pd.qcut(precio_combinado_tabla, q=10, duplicates='drop').cat.categories

    inv_agg['Intervalo']    = pd.cut(inv_agg['Precio_Lista'],    bins=bins_tabla, include_lowest=True)
    ventas_agg['Intervalo'] = pd.cut(ventas_agg['Precio_Lista'], bins=bins_tabla, include_lowest=True)

    inv_bin    = inv_agg.groupby('Intervalo', observed=True)['Cantidad_Inventario'].sum().reset_index()
    ventas_bin = ventas_agg.groupby('Intervalo', observed=True)['Cantidad_Ventas'].sum().reset_index()

    df_tabla = inv_bin.merge(ventas_bin, on='Intervalo', how='left')
    df_tabla['Cantidad_Ventas'] = df_tabla['Cantidad_Ventas'].fillna(0)

    total_inv    = df_tabla['Cantidad_Inventario'].sum()
    total_ventas = df_tabla['Cantidad_Ventas'].sum()

    df_tabla['% Inventario actual']  = (df_tabla['Cantidad_Inventario'] / total_inv * 100).round(1)
    df_tabla['% Demanda pasada']     = (
        (df_tabla['Cantidad_Ventas'] / total_ventas * 100).round(1)
        if total_ventas > 0 else 0.0
    )
    df_tabla['Inventario propuesto'] = (df_tabla['% Demanda pasada'] / 100 * total_inv).round(0).astype(int)
    df_tabla['Diferencia']           = df_tabla['Inventario propuesto'] - df_tabla['Cantidad_Inventario'].astype(int)
    df_tabla['Intervalo']            = df_tabla['Intervalo'].astype(str)

    df_tabla = df_tabla.rename(columns={
        'Intervalo':           'Rango de Precio',
        'Cantidad_Inventario': 'Inventario actual',
        'Cantidad_Ventas':     'Demanda pasada (unidades)',
    })

    st.dataframe(
        df_tabla[[
            'Rango de Precio', 'Inventario actual', '% Inventario actual',
            'Demanda pasada (unidades)', '% Demanda pasada',
            'Inventario propuesto', 'Diferencia',
        ]],
        use_container_width=True,
        hide_index=True,
    )

except Exception as e:
    st.warning(f'No se pudo construir la tabla: {e}')


# =============================================================================
# DESCARGA A NIVEL SKU
# =============================================================================

st.subheader('Detalle a Nivel SKU')
st.caption('La tabla y el CSV incluyen todas las combinaciones de código de barras y tienda con inventario disponible.')

# Filtra sku solo cuando el usuario llega a esta seccion.
sku_sel = filtrar_df(sku)

if sku_sel.empty:
    st.info('No hay detalle SKU para la selección actual.')
else:
    st.dataframe(
        sku_sel.sort_values('Cantidad_Inventario', ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = sku_sel.to_csv(index=False).encode('utf-8')
    partes_nombre = [p.replace(' ', '_') for p in partes[:3]] if partes else ['todos']
    st.download_button(
        label='⬇️ Descargar detalle SKU (CSV)',
        data=csv_bytes,
        file_name=f'sku_{"_".join(partes_nombre)}.csv',
        mime='text/csv',
    )
