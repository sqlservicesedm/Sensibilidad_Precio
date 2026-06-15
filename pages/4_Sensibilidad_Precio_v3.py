"""
4_Sensibilidad_Precio_v3.py

Pagina de Streamlit para el analisis de alineacion entre demanda pasada
e inventario actual por rango de precio.

Estructura:
    - Filtros globales en sidebar con boton Aplicar
    - Tab 1: Curvas KDE
    - Tab 2: Tabla de optimizacion por intervalo de precio
    - Tab 3: Detalle y descarga a nivel SKU

Nomenclatura:
    Canal:   marca o grupo comercial (Pilatos, Diesel, Superdry, MFG, etc.)
    Formato: tipo de operacion (Lineal, Outlet, Mixta, Franquicia, Online)

Dependencias (carpeta data/v2/):
    grupos_kde.csv
    inventario_ventas_sku.csv.gz
    despacho_sku.csv.gz

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
    """Carga grupos_kde.csv — archivo liviano, base para curvas y tabla."""
    return pd.read_csv('data/v2/grupos_kde.csv')


@st.cache_data
def cargar_sku():
    """Carga inventario_ventas_sku.csv.gz — solo para tab de detalle SKU."""
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

st.title('Análisis de Demanda Pasada vs. Inventario Actual')

with st.expander('ℹ️ Acerca de este reporte y glosario de términos', expanded=False):
    st.markdown("""
**¿Qué muestra este reporte?**
Compara cómo está distribuido el inventario actual por rango de precio
versus cómo estuvo distribuida la demanda histórica. El objetivo es
identificar si el stock disponible hoy está en los rangos donde
históricamente ha habido mayor demanda.

**Importante — se comparan dos momentos distintos:**
- La **demanda pasada** refleja lo que se vendió en el período YTD.
- El **inventario actual** es una foto del stock al cierre del último mes.
No se puede concluir que habrá quiebre futuro — solo que existe una
desalineación entre el inventario disponible y los patrones históricos de compra.

**Glosario:**
- **Canal:** marca o grupo comercial de la tienda (Pilatos, Diesel, Superdry, MFG, etc.)
- **Formato:** tipo de operación de la tienda (Lineal, Outlet, Mixta, Franquicia, Online)
- **Inventario propuesto:** redistribuye el stock actual siguiendo el patrón de demanda
  histórica. Es una referencia orientativa, no una predicción de demanda futura.
- **Densidad relativa:** las curvas muestran concentración proporcional, no cantidades
  absolutas. Un pico alto significa que ahí ocurre la mayor parte de la actividad.
""")

with st.expander('📅 Rangos de datos y fechas de actualización', expanded=False):
    if 'Fecha_Venta' in sku.columns:
        fechas = pd.to_datetime(sku['Fecha_Venta'], errors='coerce').dropna()
        rango_ventas = (
            f"{fechas.min().strftime('%d/%m/%Y')} — {fechas.max().strftime('%d/%m/%Y')}"
            if not fechas.empty else 'YTD 2026'
        )
    else:
        rango_ventas = 'YTD 2026'

    c1, c2, c3 = st.columns(3)
    c1.info('**Inventario actual**\nSnapshot al cierre del último mes disponible.')
    c2.info(f'**Demanda pasada y despacho**\nPeríodo: {rango_ventas}')
    c3.info('**Actualización**\nPrimer día de cada mes al subir nuevos archivos a `data/v2/`.')


# =============================================================================
# FILTROS EN SIDEBAR CON BOTON APLICAR
# =============================================================================

st.sidebar.header('Filtros')
st.sidebar.caption('Selecciona los filtros y haz clic en Aplicar.')

def opciones(df_base, columna):
    """Devuelve lista ordenada de valores unicos no nulos."""
    return sorted(df_base[columna].dropna().unique().tolist())


# Canal
canal_opts = opciones(grupos, 'Canal')
canal_sel = st.sidebar.multiselect('Canal', canal_opts, default=[])

g1 = grupos[grupos['Canal'].isin(canal_sel)] if canal_sel else grupos

# Formato
formato_opts = opciones(g1, 'Formato')
formato_sel = st.sidebar.multiselect('Formato', formato_opts, default=[])

g2 = g1[g1['Formato'].isin(formato_sel)] if formato_sel else g1

# Tienda
tienda_opts = opciones(g2, 'Tienda')
tienda_sel = st.sidebar.multiselect('Tienda', tienda_opts, default=[])

g3 = g2[g2['Tienda'].isin(tienda_sel)] if tienda_sel else g2

# Marca
marca_opts = opciones(g3, 'Marca')
marca_sel = st.sidebar.multiselect('Marca', marca_opts, default=[])

g4 = g3[g3['Marca'].isin(marca_sel)] if marca_sel else g3

# Genero
genero_opts = opciones(g4, 'Genero')
genero_sel = st.sidebar.multiselect('Género', genero_opts, default=[])

g5 = g4[g4['Genero'].isin(genero_sel)] if genero_sel else g4

# Tipo
tipo_opts = opciones(g5, 'Tipo')
tipo_sel = st.sidebar.multiselect('Tipo', tipo_opts, default=[])

st.sidebar.divider()
aplicar = st.sidebar.button('✅ Aplicar filtros', use_container_width=True)

# Guarda los filtros en session_state al aplicar.
if aplicar:
    st.session_state['filtros'] = {
        'canal':   canal_sel,
        'formato': formato_sel,
        'tienda':  tienda_sel,
        'marca':   marca_sel,
        'genero':  genero_sel,
        'tipo':    tipo_sel,
    }

# Usa filtros guardados o vacio si aun no se ha aplicado.
filtros = st.session_state.get('filtros', {
    'canal': [], 'formato': [], 'tienda': [],
    'marca': [], 'genero': [], 'tipo': [],
})


# =============================================================================
# APLICAR FILTROS A LOS DATOS
# =============================================================================

def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra un dataframe segun los filtros guardados en session_state."""
    mask = pd.Series([True] * len(df), index=df.index)
    if filtros['canal']:   mask &= df['Canal'].isin(filtros['canal'])
    if filtros['formato']: mask &= df['Formato'].isin(filtros['formato'])
    if filtros['tienda']:  mask &= df['Tienda'].isin(filtros['tienda'])
    if filtros['marca']:   mask &= df['Marca'].isin(filtros['marca'])
    if filtros['genero']:  mask &= df['Genero'].isin(filtros['genero'])
    if filtros['tipo']:    mask &= df['Tipo'].isin(filtros['tipo'])
    return df[mask].copy()


grupos_sel = aplicar_filtros(grupos)

if grupos_sel.empty:
    st.info('Selecciona los filtros en el panel izquierdo y haz clic en **Aplicar filtros**.')
    st.stop()

# Etiqueta del titulo segun filtros activos.
partes_titulo = []
for k in ['canal', 'formato', 'tienda', 'marca', 'genero', 'tipo']:
    if filtros[k]:
        partes_titulo.append(', '.join(filtros[k]))
titulo_filtros = ' | '.join(partes_titulo) if partes_titulo else 'Todos los grupos'


# =============================================================================
# METRICAS RESUMEN
# =============================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric('Unidades en inventario',      f'{grupos_sel["Cantidad_Inventario"].sum():,.0f}')
col2.metric('Unidades demanda pasada',     f'{grupos_sel["Cantidad_Ventas"].sum():,.0f}')
col3.metric('SKUs con demanda / total',    f'{grupos_sel["n_skus_con_ventas"].sum():,} / {grupos_sel["n_skus_inventario"].sum():,}')
col4.metric('SKUs sin demanda',            f'{(grupos_sel["n_skus_inventario"] - grupos_sel["n_skus_con_ventas"]).sum():,}')


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(['📈 Curvas KDE', '📊 Tabla de Optimización', '🔍 Detalle SKU'])


# ---------------------------------------------------------------------------
# TAB 1 — CURVAS KDE
# ---------------------------------------------------------------------------

with tab1:

    st.caption(
        'Las curvas muestran concentración relativa por rango de precio, no cantidades absolutas. '
        'Compara la forma de cada curva para identificar desalineaciones entre '
        'la demanda pasada y el inventario actual.'
    )

    def calcular_fence(s: pd.Series) -> tuple[float, float]:
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        return max(s.min(), q25 - 1.5 * iqr), min(s.max(), q75 + 1.5 * iqr)

    inv_agg = (
        grupos_sel
        .groupby('Precio_Lista_Mediana')['Cantidad_Inventario']
        .sum().reset_index()
        .rename(columns={'Precio_Lista_Mediana': 'Precio_Lista'})
    )

    ventas_agg = (
        grupos_sel[grupos_sel['flag_tiene_ventas']]
        .groupby('Precio_Lista_Mediana')['Cantidad_Ventas']
        .sum().reset_index()
        .rename(columns={'Precio_Lista_Mediana': 'Precio_Lista'})
    )

    desp_sel   = aplicar_filtros(despacho)
    desp_agg   = (
        desp_sel.groupby('Precio_Lista')['Despacho'].sum().reset_index()
        if not desp_sel.empty else pd.DataFrame(columns=['Precio_Lista', 'Despacho'])
    )

    if inv_agg.empty or inv_agg['Precio_Lista'].nunique() < 2:
        st.warning('No hay suficientes datos de inventario para construir las curvas.')
    elif ventas_agg.empty or ventas_agg['Precio_Lista'].nunique() < 2:
        st.warning('No hay suficientes datos de demanda pasada para construir las curvas. Intenta ampliar los filtros.')
    else:
        todos_precios = pd.concat(
            [inv_agg['Precio_Lista'], ventas_agg['Precio_Lista']] +
            ([desp_agg['Precio_Lista']] if not desp_agg.empty else [])
        ).dropna()

        p_min, p_max = calcular_fence(todos_precios)
        if p_min >= p_max:
            p_min, p_max = todos_precios.min(), todos_precios.max()
        rango = np.linspace(p_min, p_max, 1000)

        fig, ax = plt.subplots(figsize=(12, 6))

        kde_inv = gaussian_kde(inv_agg['Precio_Lista'], weights=inv_agg['Cantidad_Inventario'])
        ax.plot(rango, kde_inv(rango), color='#1f77b4', linewidth=2.5,
                label='Inventario actual — lo que quedó')

        kde_v = gaussian_kde(ventas_agg['Precio_Lista'], weights=ventas_agg['Cantidad_Ventas'])
        ax.plot(rango, kde_v(rango), color='#d62728', linewidth=2.5,
                label='Demanda pasada — lo que se vendió')

        if not desp_agg.empty and desp_agg['Precio_Lista'].nunique() >= 2:
            kde_d = gaussian_kde(desp_agg['Precio_Lista'], weights=desp_agg['Despacho'])
            ax.plot(rango, kde_d(rango), color='#2ca02c', linewidth=2, linestyle='--',
                    label='Despacho YTD — lo que entró')

        try:
            precio_comb = pd.concat([inv_agg['Precio_Lista'], ventas_agg['Precio_Lista']]).dropna()
            bins_prop   = pd.qcut(precio_comb, q=10, duplicates='drop').cat.categories
            vp          = ventas_agg.copy()
            vp['Int']   = pd.cut(vp['Precio_Lista'], bins=bins_prop, include_lowest=True)
            vpb         = vp.groupby('Int', observed=True)['Cantidad_Ventas'].sum().reset_index()
            vpb         = vpb[vpb['Cantidad_Ventas'] > 0]
            if len(vpb) >= 2:
                vpb['pct'] = vpb['Cantidad_Ventas'] / vpb['Cantidad_Ventas'].sum()
                vpb['mid'] = vpb['Int'].apply(lambda x: (x.left + x.right) / 2)
                kde_p      = gaussian_kde(vpb['mid'], weights=vpb['pct'])
                ax.plot(rango, kde_p(rango), color='#ff7f0e', linewidth=2, linestyle=':',
                        label='Inventario propuesto — referencia')
        except Exception:
            pass

        ax.set_xlabel('Precio Lista (COP)', fontsize=11)
        ax.set_ylabel('Densidad relativa', fontsize=11)
        ax.set_title(f'Demanda Pasada vs. Inventario Actual\n{titulo_filtros}',
                     fontsize=12, fontweight='bold')
        ax.set_yticklabels([])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        plt.tight_layout()
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# TAB 2 — TABLA DE OPTIMIZACION
# ---------------------------------------------------------------------------

with tab2:

    st.caption(
        'Compara el porcentaje del inventario actual vs. el porcentaje de demanda pasada '
        'en cada rango de precio. El inventario propuesto redistribuye el stock total '
        'siguiendo el patrón histórico de demanda.'
    )

    try:
        inv_t = (
            grupos_sel
            .groupby('Precio_Lista_Mediana')['Cantidad_Inventario']
            .sum().reset_index()
            .rename(columns={'Precio_Lista_Mediana': 'Precio_Lista'})
        )
        ven_t = (
            grupos_sel[grupos_sel['flag_tiene_ventas']]
            .groupby('Precio_Lista_Mediana')['Cantidad_Ventas']
            .sum().reset_index()
            .rename(columns={'Precio_Lista_Mediana': 'Precio_Lista'})
        )

        if inv_t.empty or ven_t.empty:
            st.warning('No hay datos suficientes para construir la tabla.')
        else:
            precio_comb_t = pd.concat([inv_t['Precio_Lista'], ven_t['Precio_Lista']]).dropna()
            bins_t        = pd.qcut(precio_comb_t, q=10, duplicates='drop').cat.categories

            inv_t['Intervalo'] = pd.cut(inv_t['Precio_Lista'], bins=bins_t, include_lowest=True)
            ven_t['Intervalo'] = pd.cut(ven_t['Precio_Lista'], bins=bins_t, include_lowest=True)

            inv_bin = inv_t.groupby('Intervalo', observed=True)['Cantidad_Inventario'].sum().reset_index()
            ven_bin = ven_t.groupby('Intervalo', observed=True)['Cantidad_Ventas'].sum().reset_index()

            df_t = inv_bin.merge(ven_bin, on='Intervalo', how='left')
            df_t['Cantidad_Ventas'] = df_t['Cantidad_Ventas'].fillna(0)

            total_inv = df_t['Cantidad_Inventario'].sum()
            total_ven = df_t['Cantidad_Ventas'].sum()

            df_t['% Inventario actual'] = (df_t['Cantidad_Inventario'] / total_inv * 100).round(1)
            df_t['% Demanda pasada']    = (
                (df_t['Cantidad_Ventas'] / total_ven * 100).round(1)
                if total_ven > 0 else 0.0
            )
            df_t['Inv. propuesto']  = (df_t['% Demanda pasada'] / 100 * total_inv).round(0).astype(int)
            df_t['Diferencia']      = df_t['Inv. propuesto'] - df_t['Cantidad_Inventario'].astype(int)
            df_t['Rango de Precio'] = df_t['Intervalo'].astype(str)

            st.dataframe(
                df_t[[
                    'Rango de Precio',
                    'Cantidad_Inventario', '% Inventario actual',
                    'Cantidad_Ventas', '% Demanda pasada',
                    'Inv. propuesto', 'Diferencia',
                ]].rename(columns={
                    'Cantidad_Inventario': 'Inventario actual',
                    'Cantidad_Ventas':     'Demanda pasada',
                }),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:
        st.warning(f'No se pudo construir la tabla: {e}')


# ---------------------------------------------------------------------------
# TAB 3 — DETALLE SKU
# ---------------------------------------------------------------------------

with tab3:

    st.caption(
        'Detalle a nivel código de barras y tienda. '
        'Incluye todas las combinaciones con inventario disponible según los filtros aplicados.'
    )

    sku_sel = aplicar_filtros(sku)

    if sku_sel.empty:
        st.info('No hay detalle SKU para la selección actual.')
    else:
        st.dataframe(
            sku_sel.sort_values('Cantidad_Inventario', ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = sku_sel.to_csv(index=False).encode('utf-8')
        partes_nombre = '_'.join(
            [p.replace(' ', '_') for filtro in [
                filtros['canal'], filtros['formato'], filtros['tienda']
            ] for p in filtro]
        ) or 'todos'

        st.download_button(
            label='⬇️ Descargar detalle SKU (CSV)',
            data=csv_bytes,
            file_name=f'sku_{partes_nombre}.csv',
            mime='text/csv',
        )

