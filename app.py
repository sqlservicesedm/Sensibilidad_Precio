import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Plataforma de Análisis Estratégico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título Principal
st.title("Plataforma de Análisis Estratégico: Recompra y Sensibilidad al Precio")

st.markdown("""
Esta infraestructura analítica ha sido diseñada para la monitorización y optimización de indicadores críticos de rendimiento. 
A través del procesamiento de datos históricos y modelos estadísticos, la plataforma permite evaluar la efectividad de las 
estrategias comerciales y operativas de la organización.
""")

st.markdown("---")

# Sección: Proyecto de Recompra
st.header("Análisis de Recompra")
st.markdown("""
Este módulo evalúa la capacidad de la red para generar lealtad en el consumidor final. El análisis se fundamenta en un 
estándar de excelencia, el cual se define a partir del percentil 75 de la operación nacional, estableciendo una tasa de recompra objetivo del **13.7%**.

El estudio permite profundizar en las siguientes dimensiones operativas:
* **Estabilidad del Talento Humano**: Evaluación del impacto de la permanencia del personal en los niveles de fidelización.
* **Indicadores de Transacción (UPT)**: Correlación entre el número de unidades por ticket y la probabilidad de retorno del cliente.
* **Gestión de Captura CRM**: Medición de la eficiencia en el registro de datos (Habeas Data) como pilar de la estrategia de contacto.
""")

st.markdown("---")

# Sección: Proyecto de Sensibilidad al Precio
st.header("Análisis de Sensibilidad al Precio")
st.markdown("""
Este componente analiza la respuesta de la demanda ante variaciones en la estructura de precios y la aplicación de 
descuentos. El objetivo es identificar los rangos de precio óptimos que maximicen el volumen de ventas sin comprometer 
el margen de contribución.

A través de esta herramienta se analizan:
* **Elasticidad de la Demanda**: Comportamiento del consumidor frente a cambios en el precio de venta.
* **Eficiencia Promocional**: Distinción entre descuentos tácticos que movilizan inventario y promociones que erosionan la rentabilidad.
* **Impacto en el Ticket Promedio**: Evaluación de cómo la agresividad comercial afecta el valor percibido de las marcas.
""")

st.sidebar.info("Seleccione un proyecto en el menú lateral para iniciar el análisis detallado.")
