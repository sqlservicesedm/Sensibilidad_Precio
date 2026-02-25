import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
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
    
    # Meta Objetiva
    META_OBJETIVA = 0.137
    df['Es_Elite'] = df['Target_Tasa_Recompra'] >= META_OBJETIVA
    
    # Bins Globales: Experiencia
    bins_exp = [0, 12, 24, 48, 96, 240]
    labels_exp = [
        '1. Nuevos (<1 Año)',
        '2. En Consolidación (1-2 Años)',
        '3. Estables (2-4 Años)',
        '4. Expertos (4-8 Años)',
        '5. Veteranos (>8 Años)'
    ]
    df['Rango_Experiencia'] = pd.cut(
        df['Antiguedad_Promedio_Ponderada_3M'], 
        bins=bins_exp, 
        labels=labels_exp
    )

    # Bins Globales: UPT
    bins_upt = [0, 1.45, 1.60, 1.80, 2.00, 10]
    labels_upt = [
        'Básico\n(<1.45 Uds)',
        'Transición\n(1.45-1.60)',
        'Estándar\n(1.60-1.80)',
        'Venta Cruzada\n(1.80-2.0)',
        'Outfit Completo\n(>2.0 Uds)'
    ]
    df['Rango_UPT'] = pd.cut(
        df['UPT_3M'], 
        bins=bins_upt, 
        labels=labels_upt
    )
    
    # Bins Globales: Calzado
    bins_calzado = [-0.01, 0.15, 0.30, 0.45, 0.75, 1.01]
    labels_calzado = [
        'Marginal\n(<15%)',
        'Complementario\n(15-30%)',
        'Core\n(30-45%)',
        'Dominante\n(45-75%)',
        'Especialista\n(>75%)'
    ]
    df['Rango_Calzado'] = pd.cut(
        df['Share_Calzado_3M'], 
        bins=bins_calzado, 
        labels=labels_calzado
    )
    # Bins Globales: Descuento
    bins_desc = [-0.01, 0.05, 0.15, 0.25, 0.40, 1.01]
    labels_desc = [
        'Full Price\n(<5%)',
        'Promo Táctica\n(5-15%)',
        'Promo Estructural\n(15-25%)',
        'Liquidación\n(25-40%)',
        'Remate\n(>40%)'
    ]
    df['Rango_Descuento'] = pd.cut(
        df['Porcentaje_Descuento_3M'], 
        bins=bins_desc, 
        labels=labels_desc
    )
    
    # Limpieza de nulos para filtros
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
# SECCIÓN 4: ANÁLISIS DINÁMICO
# ==========================================

# ------------------------------------------
# 4.1 EXPERIENCIA DEL STAFF
# ------------------------------------------
st.subheader("Experiencia del Staff")

df_clean_exp = df_filt[(df_filt['Antiguedad_Promedio_Ponderada_3M'] >= 0) & 
                       (df_filt['Antiguedad_Promedio_Ponderada_3M'] <= 240)].copy()

resumen_staff = df_clean_exp.groupby('Rango_Experiencia', observed=False).agg(
    Casos=('Target_Tasa_Recompra', 'count'),
    Seniority_Mediano_Meses=('Antiguedad_Promedio_Ponderada_3M', 'median'),
    Recompra_Mediana=('Target_Tasa_Recompra', 'median'),
    Prob_Exito_Elite=('Es_Elite', 'mean'),  
    Ticket_Mediano=('Ticket_Promedio_3M', 'median')
).reset_index()

resumen_staff['Prob_Exito_Elite'] = (resumen_staff['Prob_Exito_Elite'] * 100).round(1)
resumen_staff['Recompra_Mediana'] = resumen_staff['Recompra_Mediana'].fillna(0) 

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

tabla_staff = resumen_staff.copy()
tabla_staff.columns = ['Rango Experiencia', 'Casos', 'Seniority Mediano (Meses)', 'Recompra Mediana', 'Prob. Éxito (%)', 'Ticket Mediano']
formato_staff = {'Seniority Mediano (Meses)': "{:.1f}", 'Recompra Mediana': "{:.2%}", 'Prob. Éxito (%)': "{:.1f}%", 'Ticket Mediano': "${:,.0f}"}
st.dataframe(tabla_staff.style.format(formato_staff), use_container_width=True)

st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ------------------------------------------
# 4.2 UPT (UNIDADES POR TICKET)
# ------------------------------------------
st.subheader("Venta Cruzada y Outfit (UPT)")

df_clean_upt = df_filt[(df_filt['UPT_3M'] > 0) & (df_filt['UPT_3M'] <= 10)].copy()

resumen_upt = df_clean_upt.groupby('Rango_UPT', observed=False).agg(
    Casos=('Target_Tasa_Recompra', 'count'),
    UPT_Mediano=('UPT_3M', 'median'),
    Recompra_Mediana=('Target_Tasa_Recompra', 'median'),
    Prob_Exito_Elite=('Es_Elite', 'mean'),
    Ticket_Mediano=('Ticket_Promedio_3M', 'median')
).reset_index()

resumen_upt['Prob_Exito_Elite_Pct'] = (resumen_upt['Prob_Exito_Elite'] * 100).round(1)
resumen_upt['Recompra_Mediana'] = resumen_upt['Recompra_Mediana'].fillna(0)

sns.set_theme(style="whitegrid") 
fig_upt, ax_upt = plt.subplots(figsize=(12, 7))

sns.lineplot(data=resumen_upt, x='Rango_UPT', y='Recompra_Mediana', 
             marker='D', markersize=12, linewidth=3, color='#117864', sort=False, ax=ax_upt) 

for x, y in zip(range(len(resumen_upt)), resumen_upt['Recompra_Mediana']):
    if resumen_upt.loc[x, 'Casos'] > 0:
        label_text = f'{y*100:.1f}%'
        ax_upt.text(x, y + 0.0008, label_text, 
                 ha='center', va='bottom', 
                 fontweight='bold', fontsize=12, color='#0E6251')

ax_upt.set_title('Impacto de la Venta Cruzada (UPT) en la Recompra', fontsize=16, fontweight='bold', pad=20, color='#17202A')
ax_upt.set_ylabel('Tasa de Recompra (Mediana)', fontsize=13)
ax_upt.set_xlabel('Nivel de Unidades por Ticket (UPT)', fontsize=13)

ax_upt.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

y_min = resumen_upt['Recompra_Mediana'].min()
y_max = resumen_upt['Recompra_Mediana'].max()
if y_max > 0:
    ax_upt.set_ylim(y_min * 0.95, y_max * 1.15) 

sns.despine(left=True, bottom=True)
plt.tight_layout()

st.pyplot(fig_upt)

tabla_upt = resumen_upt[['Rango_UPT', 'Casos', 'UPT_Mediano', 'Recompra_Mediana', 'Prob_Exito_Elite_Pct', 'Ticket_Mediano']].copy()
tabla_upt.columns = ['Rango UPT', 'Casos', 'UPT Mediano', 'Recompra Mediana', 'Prob. Éxito (%)', 'Ticket Mediano']
formato_upt = {'UPT Mediano': "{:.2f}", 'Recompra Mediana': "{:.2%}", 'Prob. Éxito (%)': "{:.1f}%", 'Ticket Mediano': "${:,.0f}"}
st.dataframe(tabla_upt.style.format(formato_upt), use_container_width=True)

st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ------------------------------------------
# 4.3 SHARE DE CALZADO
# ------------------------------------------
st.subheader("Mix de Producto: Calzado")

# Filtrar datos de la selección actual
df_clean_calzado = df_filt[(df_filt['Share_Calzado_3M'] >= 0) & 
                           (df_filt['Share_Calzado_3M'] <= 1)].copy()

resumen_calzado = df_clean_calzado.groupby('Rango_Calzado', observed=False).agg(
    Casos=('Target_Tasa_Recompra', 'count'),
    Share_Mediano=('Share_Calzado_3M', 'median'),
    Recompra_Mediana=('Target_Tasa_Recompra', 'median'),
    Prob_Exito_Elite=('Es_Elite', 'mean'),
    Ticket_Mediano=('Ticket_Promedio_3M', 'median')
).reset_index()

resumen_calzado['Prob_Exito_Elite_Pct'] = (resumen_calzado['Prob_Exito_Elite'] * 100).round(1)
resumen_calzado['Recompra_Mediana'] = resumen_calzado['Recompra_Mediana'].fillna(0)

sns.set_theme(style="whitegrid")
fig_calzado, ax_calzado = plt.subplots(figsize=(12, 7))

color_chart = '#D35400' 

sns.lineplot(data=resumen_calzado, x='Rango_Calzado', y='Recompra_Mediana', 
             marker='o', markersize=12, linewidth=3, color=color_chart, sort=False, ax=ax_calzado)

for x, y in zip(range(len(resumen_calzado)), resumen_calzado['Recompra_Mediana']):
    if resumen_calzado.loc[x, 'Casos'] > 0: # Solo si hay datos en ese bin
        label_text = f'{y*100:.1f}%'
        offset = 0.0008
        va_pos = 'bottom'
        
        if y > 0.13: 
            offset = -0.0015
            va_pos = 'top'

        ax_calzado.text(x, y + offset, label_text, 
                 ha='center', va=va_pos, 
                 fontweight='bold', fontsize=12, color=color_chart)

ax_calzado.set_title('¿El Cliente de Calzado es más Fiel? (Impacto del Mix en Recompra)', fontsize=16, fontweight='bold', pad=20, color='#17202A')
ax_calzado.set_ylabel('Tasa de Recompra (Mediana)', fontsize=13)
ax_calzado.set_xlabel('Participación de Calzado en la Venta', fontsize=13)

ax_calzado.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
y_min = resumen_calzado['Recompra_Mediana'].min()
y_max = resumen_calzado['Recompra_Mediana'].max()

if y_max > 0:
    ax_calzado.set_ylim(y_min * 0.95, y_max * 1.15) 

sns.despine(left=True, bottom=True)
plt.tight_layout()

# Renderizar en Streamlit
st.pyplot(fig_calzado)

# Mostrar Tabla
tabla_calzado = resumen_calzado[['Rango_Calzado', 'Casos', 'Share_Mediano', 'Recompra_Mediana', 'Prob_Exito_Elite_Pct', 'Ticket_Mediano']].copy()
tabla_calzado.columns = ['Rango Calzado', 'Casos', 'Share Mediano', 'Recompra Mediana', 'Prob. Éxito (%)', 'Ticket Mediano']
formato_calzado = {'Share Mediano': "{:.1%}", 'Recompra Mediana': "{:.2%}", 'Prob. Éxito (%)': "{:.1f}%", 'Ticket Mediano': "${:,.0f}"}
st.dataframe(tabla_calzado.style.format(formato_calzado), use_container_width=True)

st.markdown("<br><hr><br>", unsafe_allow_html=True)

# =============================================================================
# 4.4 MAPA DE OPORTUNIDAD GEOGRÁFICA (ESTÁTICO - VISTA NACIONAL)
# =============================================================================
st.subheader("Mapa de Oportunidad Geográfica (Nacional)")
st.markdown("💡 *Este mapa mantiene la vista global de todas las ciudades para identificar plazas clave, independientemente de los filtros aplicados arriba.*")

# --- PARCHE PARA SALVAR adjustText EN STREAMLIT ---
import numpy as np
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
# --------------------------------------------------

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False

# Usamos 'df' (dataset global) para congelar la visual
resumen_scatter = df.groupby('Ciudad', observed=False).agg(
    Volumen_Casos=('Target_Tasa_Recompra', 'count'),
    Tasa_Recompra=('Target_Tasa_Recompra', 'median'),
    Prob_Exito_Elite=('Es_Elite', 'mean'),
    Ticket_Promedio=('Ticket_Promedio_3M', 'median')
).reset_index()

# Filtro mínimo para evitar ruido en el gráfico
data_scatter = resumen_scatter[resumen_scatter['Volumen_Casos'] > 20].copy()

if len(data_scatter) > 1:
    sns.set_theme(style="whitegrid")
    fig_scatter, ax_scatter = plt.subplots(figsize=(14, 9))

    sns.scatterplot(
        data=data_scatter, 
        x='Volumen_Casos', 
        y='Tasa_Recompra', 
        size='Ticket_Promedio', 
        sizes=(100, 900),
        alpha=0.6,
        palette='RdYlGn', 
        hue='Prob_Exito_Elite',
        edgecolor='black',
        linewidth=1,
        legend=False,
        ax=ax_scatter
    )

    # Ajuste del Eje X a Logarítmico
    ax_scatter.set_xscale('log')
    x_min = data_scatter['Volumen_Casos'].min() * 0.8
    x_max = data_scatter['Volumen_Casos'].max() * 1.2
    ax_scatter.set_xlim(x_min, x_max) 

    # Etiquetado usando adjustText
    texts = []
    for i in range(data_scatter.shape[0]):
        row = data_scatter.iloc[i]
        # Mantenemos tu regla de etiquetar solo las más relevantes en el gráfico para no saturarlo
        if row['Volumen_Casos'] > 100 or row['Tasa_Recompra'] > 0.13 or row['Tasa_Recompra'] < 0.08:
            label = f"{row['Ciudad']}\n({row['Tasa_Recompra']*100:.1f}%)"
            texts.append(ax_scatter.text(row['Volumen_Casos'], row['Tasa_Recompra'], label, 
                                         fontsize=9, fontweight='bold', color='#2C3E50'))

    if HAS_ADJUST_TEXT:
        adjust_text(texts, ax=ax_scatter, arrowprops=dict(arrowstyle='-', color='gray', lw=0.8))

    sns.regplot(data=data_scatter, x='Volumen_Casos', y='Tasa_Recompra', 
                scatter=False, color='#95a5a6', line_kws={'linestyle':'--', 'linewidth':1.5}, ax=ax_scatter)

    max_recompra = data_scatter['Tasa_Recompra'].max()
    if max_recompra > 0:
        ax_scatter.set_ylim(0, max_recompra * 1.15) 
    ax_scatter.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    ax_scatter.set_title('Mapa de Oportunidad Geográfica: Volumen vs. Recompra', fontsize=18, fontweight='bold', pad=20)
    ax_scatter.set_xlabel('Volumen de Operación (Total Tienda-Mes)', fontsize=12)
    ax_scatter.set_ylabel('Tasa de Recompra (Mediana)', fontsize=12)

    sns.despine(trim=True)
    plt.tight_layout()
    st.pyplot(fig_scatter)

    # ==========================================
    # TABLA COMPLETA CON SCROLL
    # ==========================================
    #st.markdown("**--- DETALLE POR CIUDAD (VISTA NACIONAL COMPLETA) ---**")
    
    # Tomamos todas las ciudades graficadas y las ordenamos
    tabla_ciudades = data_scatter.sort_values('Tasa_Recompra', ascending=False)[['Ciudad', 'Tasa_Recompra', 'Volumen_Casos', 'Ticket_Promedio']]
    tabla_ciudades.columns = ['Ciudad', 'Tasa Recompra Mediana', 'Volumen Casos', 'Ticket Promedio']
    
    # Imprimimos la tabla dándole una altura fija para forzar el scroll
    st.dataframe(
        tabla_ciudades.style.format({'Tasa Recompra Mediana': "{:.2%}", 'Ticket Promedio': "${:,.0f}"}), 
        height=350, 
        width=800
    )
st.markdown("<br><hr><br>", unsafe_allow_html=True)

# =============================================================================
# 4.5 EFECTO MONOPOLIO (CANTIDAD DE TIENDAS VS LEALTAD - ESTÁTICO)
# =============================================================================
st.subheader("Oportunidad Geográfica")
#st.markdown("💡 *Vista estática nacional: ¿Tener una sola tienda exclusiva en una ciudad genera más retención que tener múltiples opciones?*")

# 1. AGRUPACIÓN INTELIGENTE (Usando el 'df' global, no el filtrado)
resumen_ciudad_mono = df.groupby('Ciudad', observed=False).agg(
    Volumen_Casos=('Target_Tasa_Recompra', 'count'),
    Num_Tiendas=('Tienda', 'nunique'), 
    Tasa_Recompra=('Target_Tasa_Recompra', 'median'),
    Prob_Exito_Elite=('Es_Elite', 'mean'),
    Ticket_Promedio=('Ticket_Promedio_3M', 'median')
).reset_index()

# Filtro de ruido
data_mono = resumen_ciudad_mono[resumen_ciudad_mono['Volumen_Casos'] > 20].copy()

if len(data_mono) > 1:
    # 2. VISUALIZACIÓN
    sns.set_theme(style="whitegrid")
    fig_mono, ax_mono = plt.subplots(figsize=(14, 9))

    # Scatter plot con la leyenda activada para poder extraerla después
    sns.scatterplot(
        data=data_mono, 
        x='Volumen_Casos', 
        y='Tasa_Recompra', 
        size='Num_Tiendas',       
        sizes=(100, 1000),        
        alpha=0.7,
        hue='Num_Tiendas',        
        palette='viridis_r',      
        edgecolor='black',
        linewidth=1,
        legend='brief',
        ax=ax_mono
    )

    ax_mono.set_xscale('log')
    x_min_mono = data_mono['Volumen_Casos'].min() * 0.8
    x_max_mono = data_mono['Volumen_Casos'].max() * 1.2
    ax_mono.set_xlim(x_min_mono, x_max_mono)

    # 3. ETIQUETADO ESTRATÉGICO
    texts_mono = []
    for i in range(data_mono.shape[0]):
        row = data_mono.iloc[i]
        
        is_metro = row['Num_Tiendas'] >= 3
        is_single_jewel = (row['Num_Tiendas'] == 1) and (row['Tasa_Recompra'] > 0.14)
        
        if is_metro or is_single_jewel:
            label = f"{row['Ciudad']}\n({row['Num_Tiendas']} Tdas | {row['Tasa_Recompra']*100:.1f}%)"
            texts_mono.append(ax_mono.text(
                row['Volumen_Casos'], 
                row['Tasa_Recompra'], 
                label, 
                fontsize=9, 
                fontweight='bold', 
                color='#2C3E50'
            ))

    if HAS_ADJUST_TEXT:
        adjust_text(texts_mono, ax=ax_mono, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    max_recompra_mono = data_mono['Tasa_Recompra'].max()
    if max_recompra_mono > 0:
        ax_mono.set_ylim(0, max_recompra_mono * 1.15) 

    ax_mono.set_title('Efecto de la cantidad de tiendas', fontsize=16, fontweight='bold', pad=20)
    ax_mono.set_xlabel('Volumen de Operación (Escala Log)', fontsize=12)
    ax_mono.set_ylabel('Tasa de Recompra Mediana (%)', fontsize=12)
    ax_mono.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    # Ajuste seguro de la leyenda
    handles, labels = ax_mono.get_legend_handles_labels()
    if handles:
        ax_mono.legend(handles, labels, title="Cant. Tiendas", loc='upper right', frameon=True)

    sns.despine(trim=True)
    plt.tight_layout()
    st.pyplot(fig_mono)

    # 4. TABLA COMPARATIVA
    #st.markdown("**--- COMPARATIVO ESTRATÉGICO: TIENDA ÚNICA VS MULTI-TIENDA ---**")
    
    data_mono['Tiendas'] = np.where(data_mono['Num_Tiendas'] == 1, 'Única tienda (1 Tienda)', 'Múltiples tiendas (>1 Tienda)')
    comparativo = data_mono.groupby('Tiendas', observed=False).agg(
        Ciudades=('Ciudad', 'count'),
        Recompra_Promedio=('Tasa_Recompra', 'mean'), 
        Ticket_Promedio=('Ticket_Promedio', 'mean')
    ).reset_index()

    comparativo.columns = ['Tiendas', 'Cantidad de Ciudades', 'Recompra Promedio', 'Ticket Promedio']
    
    formato_comparativo = {
        'Recompra Promedio': "{:.2%}",
        'Ticket Promedio': "${:,.0f}"
    }
    st.dataframe(comparativo.style.format(formato_comparativo), width=700)

else:
    st.warning("Datos insuficientes para renderizar.")
st.markdown("<br><hr><br>", unsafe_allow_html=True)

# ------------------------------------------
# 4.6 ESTRATEGIA DE DESCUENTO (DINÁMICO)
# ------------------------------------------
st.subheader("Estrategia de Descuento")

# Filtrar datos de la selección actual
df_clean_desc = df_filt[(df_filt['Porcentaje_Descuento_3M'] >= 0) & 
                        (df_filt['Porcentaje_Descuento_3M'] <= 1)].copy()

resumen_desc = df_clean_desc.groupby('Rango_Descuento', observed=False).agg(
    Casos=('Target_Tasa_Recompra', 'count'),
    Descuento_Mediano=('Porcentaje_Descuento_3M', 'median'),
    Recompra_Mediana=('Target_Tasa_Recompra', 'median'),
    Prob_Exito_Elite=('Es_Elite', 'mean'),
    Ticket_Mediano=('Ticket_Promedio_3M', 'median')
).reset_index()

resumen_desc['Prob_Exito_Elite_Pct'] = (resumen_desc['Prob_Exito_Elite'] * 100).round(1)
resumen_desc['Recompra_Mediana'] = resumen_desc['Recompra_Mediana'].fillna(0)

sns.set_theme(style="whitegrid")
fig_desc, ax_desc = plt.subplots(figsize=(12, 7))
color_chart = '#922B21'

# Tu lineplot exacto
sns.lineplot(data=resumen_desc, x='Rango_Descuento', y='Recompra_Mediana', 
             marker='s', markersize=12, linewidth=3, color=color_chart, sort=False, ax=ax_desc)

for x, y in zip(range(len(resumen_desc)), resumen_desc['Recompra_Mediana']):
    if resumen_desc.loc[x, 'Casos'] > 0:
        label_text = f'{y*100:.1f}%'
        ax_desc.text(x, y + 0.0008, label_text, 
                 ha='center', va='bottom', 
                 fontweight='bold', fontsize=12, color=color_chart)

ax_desc.set_title('¿El Descuento "Compra" Lealtad? Impacto de la Promo en Recompra', fontsize=16, fontweight='bold', pad=20, color='#17202A')
ax_desc.set_ylabel('Tasa de Recompra (Mediana)', fontsize=13)
ax_desc.set_xlabel('Nivel de Agresividad Comercial (% Descuento)', fontsize=13)

ax_desc.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
y_min = resumen_desc['Recompra_Mediana'].min()
y_max = resumen_desc['Recompra_Mediana'].max()

if y_max > 0:
    ax_desc.set_ylim(y_min * 0.9, y_max * 1.2) 

sns.despine(left=True, bottom=True)
plt.tight_layout()
st.pyplot(fig_desc)

# Tabla Maestra de Descuento
tabla_desc = resumen_desc[['Rango_Descuento', 'Casos', 'Descuento_Mediano', 'Recompra_Mediana', 'Prob_Exito_Elite_Pct', 'Ticket_Mediano']].copy()
tabla_desc.columns = ['Rango Descuento', 'Casos', 'Descuento Mediano', 'Recompra Mediana', 'Prob. Éxito (%)', 'Ticket Mediano']
formato_desc = {
    'Descuento Mediano': "{:.1%}", 
    'Recompra Mediana': "{:.2%}", 
    'Prob. Éxito (%)': "{:.1f}%", 
    'Ticket Mediano': "${:,.0f}"
}
st.dataframe(tabla_desc.style.format(formato_desc), width=800)

