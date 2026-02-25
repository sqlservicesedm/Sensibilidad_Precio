import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# ANÁLISIS FINAL: STAFF CON META OBJETIVA (P75 = 13.7%)
# =============================================================================

# 1. LIMPIEZA Y PREPARACIÓN
df_clean = df[(df['Antiguedad_Promedio_Ponderada_3M'] >= 0) & 
              (df['Antiguedad_Promedio_Ponderada_3M'] <= 240)].copy()

# Definimos la Meta Objetiva (P75 de la red)
META_OBJETIVA = 0.137

# 2. BINS DE SENIORITY REALISTAS
bins_final = [0, 12, 24, 48, 96, 240]
labels_final = [
    '1. Nuevos (<1 Año)',
    '2. En Consolidación (1-2 Años)',
    '3. Estables (2-4 Años)',
    '4. Expertos (4-8 Años)',
    '5. Veteranos (>8 Años)'
]

df_clean['Rango_Experiencia'] = pd.cut(
    df_clean['Antiguedad_Promedio_Ponderada_3M'], 
    bins=bins_final, 
    labels=labels_final
)

# Marcamos quiénes alcanzan la élite (Top 25%)
df_clean['Es_Elite'] = df_clean['Target_Tasa_Recompra'] >= META_OBJETIVA

# 3. TABLA MAESTRA DE STAFF
resumen_staff = df_clean.groupby('Rango_Experiencia', observed=False).agg(
    Casos=('Target_Tasa_Recompra', 'count'),
    Seniority_Mediano_Meses=('Antiguedad_Promedio_Ponderada_3M', 'median'),
    Recompra_Mediana=('Target_Tasa_Recompra', 'median'),
    Prob_Exito_Elite=('Es_Elite', 'mean'),  # % de tiendas que llegan al Top 25%
    Ticket_Mediano=('Ticket_Promedio_3M', 'median')
).reset_index()

# Convertimos a porcentaje para lectura ejecutiva
resumen_staff['Prob_Exito_Elite'] = (resumen_staff['Prob_Exito_Elite'] * 100).round(1)

# 4. EXPORTACIÓN
resumen_staff.to_excel('staff_analisis_final_p75.xlsx', index=False)

# 5. VISUALIZACIÓN DE LA "ESCALERA AL ÉXITO"
plt.figure(figsize=(12, 6))
sns.lineplot(data=resumen_staff, x='Rango_Experiencia', y='Recompra_Mediana', 
             marker='s', markersize=12, linewidth=4, color='#1b4f72', sort=False)

for x, y in zip(range(len(resumen_staff)), resumen_staff['Recompra_Mediana']):
    plt.text(x, y + 0.001, f'{y*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.title('Impacto de la Experiencia en la Recompra', fontsize=15, fontweight='bold')
plt.ylabel('Tasa de Recompra Mediana (%)')
plt.xlabel('Años de Experiencia del Equipo en Tienda')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n--- RESUMEN FINAL STAFF (META OBJETIVA {META_OBJETIVA*100}%) ---")
print(resumen_staff)
