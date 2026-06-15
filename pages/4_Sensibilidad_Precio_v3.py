"""
4_Sensibilidad_Precio_v3.py

Pagina de Streamlit para el analisis de alineacion entre demanda pasada
e inventario actual por rango de precio.

Estructura:
    - Filtros globales en sidebar con boton Aplicar
    - Tab 1: Curvas KDE
    - Tab 2: Tabla de optimizacion por intervalo de precio
    - Tab 3: Descarga CSV del resumen filtrado

Nota sobre nombres en los CSV vs. negocio:
    Columna 'Canal'   en CSV = Pilatos, Diesel, etc. → se muestra como 'Formato' en la UI
    Columna 'Formato' en CSV = Linea, Outlet, etc.   → se muestra como 'Canal' en la UI

Dependencias (carpeta data/v2/):
    grupos_kde.csv          — metricas resumen por grupo
    precios_streamlit.csv   — cantidades por grupo x precio para KDE y tabla

Ultima actualizacion: 2026-06-13
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

st.set_page_config(page_title='Sensibilidad al Precio v3', layout='wide')


# =============================================================================
# CARGA DE DATOS — solo archivos livianos
# =============================================================================

@st.cache_data
def cargar_datos():
    """Carga los dos archivos livianos de data/v2/."""
    grupos   = pd.read_csv('data/v2/grupos_kde.csv')
    precios  = pd.read_csv('data/v2/precios_streamlit.csv')
    return grupos, precios


grupos, precios = cargar_datos()


# =============================================================================
# ENCABEZADO
# =============================================================================

st.title('Análisis de Demanda Pasada vs. Inventario Actual')

with st.expander('ℹ️ Acerca de este reporte y glosario de términos', expanded=False):
    st.markdown("""
**¿Qué muestra este reporte?**
Compara cómo está distribuido el inventario actual por rango de precio
versus cómo estuvo distribuida la demanda histórica. El objetivo es
identificar si el stock disponible hoy está en los rangos de precio donde
históricamente ha habido mayor demanda.

**Importante — se comparan dos momentos distintos:**
- La **demanda pasada** refleja lo que se vendió en el período YTD analizado.
- El **inventario actual** es una foto del stock al cierre del último mes.
No se puede concluir que habrá quiebre futuro — solo que existe una desalineación
entre el inventario disponible y los patrones históricos de compra.

**Glosario:**
- **Formato:** marca o grupo comercial de la tienda (Pilatos, Diesel, Superdry, MFG, etc.)
- **Canal:** tipo de operación de la tienda (Lineal, Outlet, Mixta, Franquicia, Online)
- **Inventario propuesto:** redistribuye el stock actual siguiendo el patrón de demanda
  histórica. Es una referencia orientativa, no una predicción de demanda futura.
- **Densidad relativa:** las curvas muestran concentración proporcional, no cantidades
  absolutas. Un pico alto significa que ahí ocurre la mayor parte de la actividad.
""")

with st.expander('📅 Rangos de datos y fechas de actualización', expanded=False):
    meta = pd.read_csv('data/v2/metadata.csv').iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.info(f'**Inventario actual**\nSnapshot al cierre de **{meta["periodo_inventario"]}**.')
    c2.info(f'**Demanda pasada y despacho**\nPeríodo: **{meta["fecha_min_venta"]}** al **{meta["fecha_max_venta"]}**.')
    c3.info('**Actualización**\nPrimer día de cada mes al subir nuevos archivos a `data/v2/`.')


# =============================================================================
# FILTROS EN SIDEBAR CON BOTON APLICAR
# =============================================================================

st.sidebar.header('Filtros')
st.sidebar.caption('Selecciona los filtros y haz clic en Aplicar.')

# Nota: columna 'Canal' en CSV = Formato en negocio (Pilatos, Diesel...)
#        columna 'Formato' en CSV = Canal en negocio (Linea, Outlet...)

def opciones(df_base, col):
    return sorted(df_base[col].dropna().unique().tolist())


# Formato en UI → columna Canal en CSV
fmt_opts = opciones(grupos, 'Canal')
fmt_sel  = st.sidebar.multiselect('Formato', fmt_opts, default=[])
g1 = grupos[grupos['Canal'].isin(fmt_sel)] if fmt_sel else grupos

# Canal en UI → columna Formato en CSV
canal_opts = opciones(g1, 'Formato')
canal_sel  = st.sidebar.multiselect('Canal', canal_opts, default=[])
g2 = g1[g1['Formato'].isin(canal_sel)] if canal_sel else g1

# Tienda
tienda_opts = opciones(g2, 'Tienda')
tienda_sel  = st.sidebar.multiselect('Tienda', tienda_opts, default=[])
g3 = g2[g2['Tienda'].isin(tienda_sel)] if tienda_sel else g2

# Marca
marca_opts = opciones(g3, 'Marca')
marca_sel  = st.sidebar.multiselect('Marca', marca_opts, default=[])
g4 = g3[g3['Marca'].isin(marca_sel)] if marca_sel else g3

# Genero
genero_opts = opciones(g4, 'Genero')
genero_sel  = st.sidebar.multiselect('Género', genero_opts, default=[])
g5 = g4[g4['Genero'].isin(genero_sel)] if genero_sel else g4

# Tipo
tipo_opts = opciones(g5, 'Tipo')
tipo_sel  = st.sidebar.multiselect('Tipo', tipo_opts, default=[])

st.sidebar.divider()
aplicar = st.sidebar.button('✅ Aplicar filtros', use_container_width=True)

if aplicar:
    st.session_state['filtros'] = {
        'Canal':   fmt_sel,
        'Formato': canal_sel,
        'Tienda':  tienda_sel,
        'Marca':   marca_sel,
        'Genero':  genero_sel,
        'Tipo':    tipo_sel,
    }

filtros = st.session_state.get('filtros', {
    'Canal': [], 'Formato': [], 'Tienda': [],
    'Marca': [], 'Genero': [], 'Tipo': [],
})


# =============================================================================
# APLICAR FILTROS
# =============================================================================

def filtrar(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra un dataframe segun los filtros en session_state."""
    mask = pd.Series([True] * len(df), index=df.index)
    for col, vals in filtros.items():
        if vals and col in df.columns:
            mask &= df[col].isin(vals)
    return df[mask].copy()


grupos_sel  = filtrar(grupos)
precios_sel = filtrar(precios)

if grupos_sel.empty or precios_sel.empty:
    st.info('Selecciona los filtros en el panel izquierdo y haz clic en **✅ Aplicar filtros**.')
    st.stop()

# Titulo dinamico.
partes = []
if filtros['Canal']:   partes.append(', '.join(filtros['Canal']))
if filtros['Formato']: partes.append(', '.join(filtros['Formato']))
if filtros['Tienda']:  partes.append(', '.join(filtros['Tienda']))
if filtros['Marca']:   partes.append(', '.join(filtros['Marca']))
if filtros['Genero']:  partes.append(', '.join(filtros['Genero']))
if filtros['Tipo']:    partes.append(', '.join(filtros['Tipo']))
titulo_filtros = ' | '.join(partes) if partes else 'Todos los grupos'


# =============================================================================
# METRICAS RESUMEN
# =============================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric('Unidades en inventario',  f'{grupos_sel["Cantidad_Inventario"].sum():,.0f}')
col2.metric('Demanda pasada',          f'{grupos_sel["Cantidad_Ventas"].sum():,.0f}')
col3.metric('Con demanda / total',     f'{grupos_sel["n_skus_con_ventas"].sum():,} / {grupos_sel["n_skus_inventario"].sum():,}')
col4.metric('Sin demanda en periodo',  f'{(grupos_sel["n_skus_inventario"] - grupos_sel["n_skus_con_ventas"]).sum():,}')


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3 = st.tabs(['📈 Curvas KDE', '📊 Tabla de Optimización', '⬇️ Descarga'])


# ---------------------------------------------------------------------------
# FUNCIONES COMPARTIDAS
# ---------------------------------------------------------------------------

def calcular_fence(s: pd.Series) -> tuple[float, float]:
    """Calcula los limites de rango visible con criterio IQR."""
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    iqr = q75 - q25
    return max(s.min(), q25 - 1.5 * iqr), min(s.max(), q75 + 1.5 * iqr)


inv_agg  = precios_sel.groupby('Precio_Lista')['Cantidad_Inventario'].sum().reset_index()
ven_agg  = precios_sel[precios_sel['Cantidad_Ventas'] > 0].groupby('Precio_Lista')['Cantidad_Ventas'].sum().reset_index()
desp_agg = precios_sel[precios_sel['Despacho'] > 0].groupby('Precio_Lista')['Despacho'].sum().reset_index()

datos_ok = (
    not inv_agg.empty and inv_agg['Precio_Lista'].nunique() >= 2 and
    not ven_agg.empty and ven_agg['Precio_Lista'].nunique() >= 2
)


# ---------------------------------------------------------------------------
# TAB 1 — CURVAS KDE
# ---------------------------------------------------------------------------

with tab1:
    st.caption(
        'Las curvas muestran concentración relativa por rango de precio, no cantidades absolutas. '
        'Compara la forma de cada curva para identificar desalineaciones entre '
        'la demanda pasada y el inventario actual.'
    )

    if not datos_ok:
        st.warning('No hay suficientes datos para construir las curvas. Amplía los filtros.')
    else:
        todos_precios = pd.concat(
            [inv_agg['Precio_Lista'], ven_agg['Precio_Lista']] +
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

        kde_ven = gaussian_kde(ven_agg['Precio_Lista'], weights=ven_agg['Cantidad_Ventas'])
        ax.plot(rango, kde_ven(rango), color='#d62728', linewidth=2.5,
                label='Demanda pasada — lo que se vendió')

        if not desp_agg.empty and desp_agg['Precio_Lista'].nunique() >= 2:
            kde_desp = gaussian_kde(desp_agg['Precio_Lista'], weights=desp_agg['Despacho'])
            ax.plot(rango, kde_desp(rango), color='#2ca02c', linewidth=2, linestyle='--',
                    label='Despacho YTD — lo que entró')

        try:
            precio_comb = pd.concat([inv_agg['Precio_Lista'], ven_agg['Precio_Lista']]).dropna()
            bins_prop   = pd.qcut(precio_comb, q=10, duplicates='drop').cat.categories
            vp          = ven_agg.copy()
            vp['Int']   = pd.cut(vp['Precio_Lista'], bins=bins_prop, include_lowest=True)
            vpb         = vp.groupby('Int', observed=True)['Cantidad_Ventas'].sum().reset_index()
            vpb         = vpb[vpb['Cantidad_Ventas'] > 0]
            if len(vpb) >= 2:
                vpb['pct'] = vpb['Cantidad_Ventas'] / vpb['Cantidad_Ventas'].sum()
                vpb['mid'] = vpb['Int'].apply(lambda x: (x.left + x.right) / 2)
                kde_prop   = gaussian_kde(vpb['mid'], weights=vpb['pct'])
                ax.plot(rango, kde_prop(rango), color='#ff7f0e', linewidth=2, linestyle=':',
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

    if not datos_ok:
        st.warning('No hay suficientes datos para construir la tabla. Amplía los filtros.')
    else:
        try:
            precio_comb_t = pd.concat([inv_agg['Precio_Lista'], ven_agg['Precio_Lista']]).dropna()
            bins_t        = pd.qcut(precio_comb_t, q=10, duplicates='drop').cat.categories

            inv_t = inv_agg.copy()
            ven_t = ven_agg.copy()
            inv_t['Intervalo'] = pd.cut(inv_t['Precio_Lista'], bins=bins_t, include_lowest=True)
            ven_t['Intervalo'] = pd.cut(ven_t['Precio_Lisa'] if 'Precio_Lisa' in ven_t.columns else ven_t['Precio_Lista'], bins=bins_t, include_lowest=True)

            inv_bin = inv_t.groupby('Intervalo', observed=True)['Cantidad_Inventario'].sum().reset_index()
            ven_bin = ven_t.groupby('Intervalo', observed=True)['Cantidad_Ventas'].sum().reset_index()

            df_t = inv_bin.merge(ven_bin, on='Intervalo', how='left')
            df_t['Cantidad_Ventas'] = df_t['Cantidad_Ventas'].fillna(0)

            total_inv = df_t['Cantidad_Inventario'].sum()
            total_ven = df_t['Cantidad_Ventas'].sum()

            df_t['% Inventario actual'] = (df_t['Cantidad_Inventario'] / total_inv * 100).round(1)
            df_t['% Demanda pasada']    = (
                (df_t['Cantidad_Ventas'] / total_ven * 100).round(1) if total_ven > 0 else 0.0
            )
            df_t['Inv. propuesto'] = (df_t['% Demanda pasada'] / 100 * total_inv).round(0).astype(int)
            df_t['Diferencia']     = df_t['Inv. propuesto'] - df_t['Cantidad_Inventario'].astype(int)
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
# TAB 3 — DESCARGA
# ---------------------------------------------------------------------------

with tab3:
    st.caption(
        'Descarga el resumen de precios con las cantidades de inventario, '
        'demanda pasada y despacho para la selección actual.'
    )

    if precios_sel.empty:
        st.info('No hay datos para la selección actual.')
    else:
        st.dataframe(
            precios_sel.sort_values('Cantidad_Inventario', ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = precios_sel.to_csv(index=False).encode('utf-8')
        partes_nombre = '_'.join(
            [v.replace(' ', '_') for k in ['Canal', 'Formato', 'Tienda'] for v in filtros[k]]
        ) or 'todos'

        st.download_button(
            label='⬇️ Descargar resumen (CSV)',
            data=csv_bytes,
            file_name=f'resumen_{partes_nombre}.csv',
            mime='text/csv',
        )

