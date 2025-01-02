import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Cargar los datos de ventas e inventario
df_ventas = pd.read_csv('ventas_para_elasticidad.csv')
df_inventario = pd.read_csv('inventario_para_elasticidad.csv')
df_despacho = pd.read_csv('despacho_para_elasticidad.csv')

df_recomendacion = pd.read_csv('recomendacion_rango_precios.csv')

# Título de la app
st.title('Análisis de Elasticidad: Curvas de Densidad y Optimización de Inventario')

# Creación de los selectores
marca = st.selectbox('Selecciona la Marca:', df_ventas['Marca'].unique())
genero = st.selectbox('Selecciona el Género:', df_ventas['Genero'].unique())
tipo = st.selectbox('Selecciona el Tipo:', df_ventas['Tipo'].unique())
tienda_options = np.append(df_ventas['Tienda'].unique(), 'Todas las tiendas')
tienda = st.selectbox('Selecciona la Tienda:', tienda_options)
precios = np.array([50000,100000,150000,200000,250000,300000,350000,400000,450000,500000])
excluir = st.selectbox('Selecciona el precio minimo:', precios)


tiendas_excluidas = ['COBRO A TRANSPORTADORAS','MUESTRAS FISICAS PROVEEDORES SERVICIOS',
                     'NOVEDADES TIENDAS ONLINE','PILATOS ACCESORIOS SAN SILVIESTRE',
                     'PRESTAMO EMPLEADOS','TEMPORAL 5','PILATOS GRAN PLAZA ALCARAVAN YOPAL',
                     'PLANTA PILOTO COSTURA','GARANTIAS EDM','FALTANTES DESPACHOS CEDI',
                     'TEMPORAL 2','BODEGA DEVOLUCIONES ONLINE','RECUPERACION DE ROPA',
                     'NOVEDADES EN IMPORTACION','BODEGA DE SEGUNDAS OBSOLETAS',
                     'CONFE CONCILIACION','SEGUNDAS','MOVIL EVENTO 2','MERCANCIA CONSIGNACIÓN RECIBIDA']


# Filtrar y agrupar los datos de ventas
if tienda == 'Todas las tiendas':
    df_ventas_filtered = df_ventas[(df_ventas['Marca'] == marca) & 
                                    (df_ventas['Genero'] == genero) & 
                                    (df_ventas['Tipo'] == tipo) & 
                                    (~df_ventas['Tienda'].isin(tiendas_excluidas))&
                                    (df_ventas['Precio_unitario_promedio']>=excluir)]
else:
    df_ventas_filtered = df_ventas[(df_ventas['Marca'] == marca) & 
                                    (df_ventas['Genero'] == genero) & 
                                    (df_ventas['Tipo'] == tipo) & 
                                    (df_ventas['Tienda'] == tienda) &
                                    (df_ventas['Precio_unitario_promedio']>=excluir)]

df_ventas_grouped = df_ventas_filtered.groupby('Precio_unitario_promedio').agg(
    Cantidad_Ventas=('Cantidad', 'sum')
).reset_index()

# Filtrar y agrupar los datos de inventario
if tienda == 'Todas las tiendas':
    df_inventario_filtered = df_inventario[(df_inventario['Marca'] == marca) & 
                                           (df_inventario['Genero'] == genero) & 
                                           (df_inventario['Tipo'] == tipo) &
                                           (~df_inventario['Descripcion_bodega'].isin(tiendas_excluidas)) &
                                           (df_inventario['f126_precio']>=excluir)]
else:
    df_inventario_filtered = df_inventario[(df_inventario['Marca'] == marca) & 
                                           (df_inventario['Genero'] == genero) & 
                                           (df_inventario['Tipo'] == tipo) & 
                                           (df_inventario['Descripcion_bodega'] == tienda) &
                                           (df_inventario['f126_precio']>=excluir)]

df_inventario_grouped = df_inventario_filtered.groupby('f126_precio').agg(
    Cantidad_Inventario=('Cantidad_Inventario', 'sum')
).reset_index()

# Filtrar y agrupar los datos de despacho
if tienda == 'Todas las tiendas':
    df_despacho_filtered = df_despacho[(df_despacho['Marca'] == marca) & 
                                       (df_despacho['Genero'] == genero) & 
                                       (df_despacho['Tipo'] == tipo)&
                                       (~df_despacho['nombrealmacen'].isin(tiendas_excluidas)) &
                                       (df_despacho['f126_precio']>=excluir)]
else:
    df_despacho_filtered = df_despacho[(df_despacho['Marca'] == marca) & 
                                       (df_despacho['Genero'] == genero) & 
                                       (df_despacho['Tipo'] == tipo) & 
                                       (df_despacho['nombrealmacen'] == tienda) &
                                       (df_despacho['f126_precio']>=excluir)]

df_despacho_grouped = df_despacho_filtered.groupby('f126_precio').agg(
    Cantidad_Despacho=('Despacho', 'sum')
).reset_index()


# Crear un rango continuo para los precios
precio_unitario_range = np.linspace(min( max(df_ventas_grouped['Precio_unitario_promedio'].min(),
                                             df_ventas_grouped['Precio_unitario_promedio'].quantile(0.25) - (1.5)*(df_ventas_grouped['Precio_unitario_promedio'].quantile(0.75) - df_ventas_grouped['Precio_unitario_promedio'].quantile(0.25))),
                                         max(df_inventario_grouped['f126_precio'].min(),
                                             df_inventario_grouped['f126_precio'].quantile(0.25) - (1.5)*(df_inventario_grouped['f126_precio'].quantile(0.75) - df_inventario_grouped['f126_precio'].quantile(0.25)))
                                        ),
                                    min( min(df_ventas_grouped['Precio_unitario_promedio'].max(),
                                             df_ventas_grouped['Precio_unitario_promedio'].quantile(0.75) + (1.5)*(df_ventas_grouped['Precio_unitario_promedio'].quantile(0.75) - df_ventas_grouped['Precio_unitario_promedio'].quantile(0.25))),
                                         min(df_inventario_grouped['f126_precio'].max(),
                                             df_inventario_grouped['f126_precio'].quantile(0.75) + (1.5)*(df_inventario_grouped['f126_precio'].quantile(0.75) - df_inventario_grouped['f126_precio'].quantile(0.25)))
                                        ),
                                    1000)

# Ajustar la curva de densidad para las ventas
kde_ventas = gaussian_kde(df_ventas_grouped['Precio_unitario_promedio'], weights=df_ventas_grouped['Cantidad_Ventas'])
density_ventas = kde_ventas(precio_unitario_range)

# Ajustar la curva de densidad para el inventario
kde_inventario = gaussian_kde(df_inventario_grouped['f126_precio'], weights=df_inventario_grouped['Cantidad_Inventario'])
density_inventario = kde_inventario(precio_unitario_range)

# Ajustar la curva de densidad para el despacho
kde_despacho = gaussian_kde(df_despacho_grouped['f126_precio'], weights=df_despacho_grouped['Cantidad_Despacho'])
density_despacho = kde_despacho(precio_unitario_range)

# Crear los intervalos para df_inventario_grouped
bins = pd.cut(precio_unitario_range, bins=10, right=False)

df_inventario_grouped['Precio_Intervalo'] = pd.cut(
    df_inventario_grouped['f126_precio'], 
    bins=bins.categories,  # Usar las categorías de los bins generados
    right=False
)
# Aplicar los mismos intervalos a df_ventas_grouped
df_ventas_grouped['Precio_Intervalo'] = pd.cut(df_ventas_grouped['Precio_unitario_promedio'], 
                                               bins=bins.categories, 
                                               right=False)
# Aplicar los mismos intervalos a df_ventas_grouped
df_despacho_grouped['Precio_Intervalo'] = pd.cut(df_despacho_grouped['f126_precio'], 
                                               bins=bins.categories, 
                                               right=False)
# Agrupar por intervalos y sumar las ventas
df_ventas_grouped_by_interval = df_ventas_grouped.groupby('Precio_Intervalo').agg(
    Cantidad_Ventas=('Cantidad_Ventas', 'sum')
).reset_index()

df_inventario_grouped_by_interval = df_inventario_grouped.groupby('Precio_Intervalo').agg(
    Cantidad_Inventario=('Cantidad_Inventario', 'sum')
).reset_index()

df_despacho_grouped_by_interval = df_despacho_grouped.groupby('Precio_Intervalo').agg(
     Cantidad_Despacho=('Cantidad_Despacho', 'sum')
).reset_index()

# Calcular el porcentaje de ventas por intervalo
total_ventas = df_ventas_grouped_by_interval['Cantidad_Ventas'].sum()
df_ventas_grouped_by_interval['Porcentaje_Ventas'] = df_ventas_grouped_by_interval['Cantidad_Ventas'] / total_ventas

# Calcular los puntos medios de los intervalos utilizando la columna 'Precio_Intervalo'
df_ventas_grouped_by_interval['Interval_Midpoint'] = df_ventas_grouped_by_interval['Precio_Intervalo'].apply(
    lambda x: (x.left + x.right) / 2
)

# Ajustar la curva de densidad para los porcentajes de ventas
kde_porcentaje_ventas = gaussian_kde(df_ventas_grouped_by_interval['Interval_Midpoint'], 
                                     weights=df_ventas_grouped_by_interval['Porcentaje_Ventas'])
density_porcentaje_ventas = kde_porcentaje_ventas(precio_unitario_range)

# Crear el gráfico con las tres curvas
plt.figure(figsize=(12, 8))

# Curva de densidad de ventas
plt.plot(precio_unitario_range, density_ventas, color='red', label='Curva de Densidad de Ventas')

# Curva de densidad de inventario
plt.plot(precio_unitario_range, density_inventario, color='blue', label='Curva de Densidad de Inventario')

# Curva de densidad de inventario
plt.plot(precio_unitario_range, density_despacho, color='black', label='Curva de Densidad de Despacho')

# Curva de densidad de porcentaje de ventas
plt.plot(precio_unitario_range, density_porcentaje_ventas, color='green', label='Curva de Inventario Propuesta')

# Configurar las etiquetas y leyenda
plt.xlabel('Precio Unitario Promedio')
plt.ylabel('Densidad')
plt.title('Curvas de Densidad para Ventas, Inventario, Despacho y Porcentaje de Ventas')
plt.legend()
plt.grid(True)
st.pyplot(plt)

# Inventario actual:
invt_actual = df_inventario_filtered.groupby(['Marca','Genero','Tipo','Descripcion_bodega']).agg(
    Cantidad_Inventario=('Cantidad_Inventario', 'sum')
).reset_index()

# Mostrar el resultado en Streamlit
st.write('Inventario Actual:')
st.dataframe(invt_actual)

# Calcular la optimización de inventario
# Asegurarse de que 'Precio_Intervalo' esté en formato de cadena
df_inventario_grouped_by_interval['Precio_Intervalo'] = df_inventario_grouped_by_interval['Precio_Intervalo'].astype(str)
df_ventas_grouped_by_interval['Precio_Intervalo'] = df_ventas_grouped_by_interval['Precio_Intervalo'].astype(str)

# Asegurarse de que los intervalos sean exactamente iguales
df_inventario_grouped_by_interval['Precio_Intervalo'] = df_inventario_grouped_by_interval['Precio_Intervalo'].str.replace('.00', '.0', regex=False)
df_ventas_grouped_by_interval['Precio_Intervalo'] = df_ventas_grouped_by_interval['Precio_Intervalo'].str.replace('.00', '.0', regex=False)

# Combinar los DataFrames por 'Precio_Intervalo'
df_combined = pd.merge(df_inventario_grouped_by_interval, 
                       df_ventas_grouped_by_interval[['Precio_Intervalo', 'Porcentaje_Ventas']], 
                       on='Precio_Intervalo', 
                       how='inner')

# Calcular el porcentaje de inventario actual en cada intervalo
total_inventario_actual = df_combined['Cantidad_Inventario'].sum()
df_combined['Porcentaje_Actual'] = df_combined['Cantidad_Inventario'] / total_inventario_actual

# Optimizar el inventario basado en el porcentaje de ventas
df_combined['Cantidad_Propuesta_Inventario'] = df_combined['Porcentaje_Ventas'] * total_inventario_actual

# Mostrar el DataFrame final con las columnas requeridas
df_final = df_combined[['Precio_Intervalo', 'Cantidad_Inventario', 'Porcentaje_Actual', 'Porcentaje_Ventas', 'Cantidad_Propuesta_Inventario']]

# Redondear la cantidad propuesta de inventario a números enteros
df_final['Cantidad_Propuesta_Inventario'] = df_final['Cantidad_Propuesta_Inventario'].round()

# Mostrar el resultado en Streamlit
st.write('Optimización de Inventario:')
st.dataframe(df_final)

# Cargar los datos de probabilidades
probabilidades = pd.read_csv('Probabilidades_Ventas.csv', dtype={'SKU': str, 'cmitems_codbarra_principal': str})

# Filtrar y agrupar los datos de ventas
probabilidades_filtered = probabilidades[(probabilidades['cmitems_MARCA'] == marca) & 
                                (probabilidades['cmitems_GENERO'] == genero) & 
                                (probabilidades['cmitems_TIPO'] == tipo) & 
                                (probabilidades['almacen_descri'] == tienda)].copy()

# Asignar los intervalos de precios a cada SKU en el DataFrame de probabilidades
probabilidades_filtered['Precio_Intervalo'] = pd.cut(probabilidades_filtered['f126_precio'], bins=bins.categories, right=False)

# Convertir 'Precio_Intervalo' a str para asegurar la agrupación correcta
#probabilidades_filtered['Precio_Intervalo'] = probabilidades_filtered['Precio_Intervalo'].astype(str)

# Seleccionar las columnas de interés, después de agregar 'Precio_Intervalo'
columns_of_interest = ['almacen_descri', 'cmitems_MARCA', 'cmitems_GENERO', 'cmitems_TIPO', 
                        'cmitems_MARCALINEA', 'cmitems_Talla', 'cmitems_Extension', 
                        'cmitems_codbarra_principal','f126_precio', 'Precio_Intervalo', 'P_probabilidad_normalizada','Cant_inventario']

# Filtrar el DataFrame para mantener solo las columnas de interés
probabilidades_filtered = probabilidades_filtered[columns_of_interest]

# Agrupar por las columnas de interés y el intervalo de precio
grouped = probabilidades_filtered.groupby(
    [ 'Precio_Intervalo']
)

# Obtener los 20 SKUs con mayor P_probabilidad_normalizada para cada grupo
top_20_skus = grouped.apply(lambda x: x.nlargest(20, 'P_probabilidad_normalizada')).reset_index(drop=True)

# Ordenar top_20_skus por 'Precio_Intervalo'
top_20_skus = top_20_skus.sort_values(by=['Precio_Intervalo', 'P_probabilidad_normalizada'], 
                                      ascending=[True, False]).reset_index(drop=True)


# Mostrar el top 20 SKUs en Streamlit
st.write('Top 20 SKUs por Probabilidad de Venta:')
st.dataframe(top_20_skus)

# 3. Generar un botón para descargar el archivo
csv = df_recomendacion.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar CSV Completo",
    data=csv,
    file_name='recomendaciones_rango_precios.csv',
    mime='text/csv'
)
