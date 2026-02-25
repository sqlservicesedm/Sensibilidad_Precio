import streamlit as st

# Configuración global de la página (Debe ser la primera línea de código en Streamlit)
st.set_page_config(
    page_title="Analytics Hub",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analytics Hub Comercial")
st.markdown("---")

st.markdown("""
### Bienvenido al Panel de Control de Estrategia

Selecciona un módulo en el menú lateral izquierdo para comenzar:

- **🔄 Proyecto Recompra** — Radiografía de las 6 palancas de fidelización por ciudad, formato y tienda. Evalúa la probabilidad de alta recompra.
- **💰 Sensibilidad Precio** — Curvas de elasticidad, probabilidad de venta por rango y optimización de inventario.

---
*Herramienta interna para la toma de decisiones basada en datos.*
""")