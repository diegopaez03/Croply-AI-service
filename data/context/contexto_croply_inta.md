# Contexto Técnico y Agronómico: Proyecto Croply (Región de Cuyo, Argentina)

## 1. Arquitectura del Servicio de IA en Croply
- **Stack:** Python 3.13 + FastAPI (`analyzer.py` + `recommender.py`) + Ollama (Gemma 4 local).
- **Flujo de datos:**
  1. **Sensores IoT por parcela (`Croply IoT Simulator` -> MQTT -> InfluxDB 3):**
     - `temp_ambiente_c` (°C)
     - `humedad_ambiente_pct` (%)
     - `humedad_suelo_pct` (Moisture %)
     - `lluvia_mm` (Pluviómetro mm)
     - `radiacion_solar_wm2` (W/m²)
     - `ph_suelo` (pH)
  2. **API Climática Externa (Open-Meteo / OpenWeather -> PostgreSQL `weather_records`):**
     - Pronóstico próximas 48-72 hs: Temperatura mín/máx (°C), Precipitación acumulada 24h (mm), Humedad relativa (%), Velocidad del viento (km/h).
  3. **Datos Transaccionales de la Parcela (PostgreSQL 16):**
     - Cultivo activo, Variedad, Fecha de siembra, Días desde la siembra (`dias_desde_siembra`), Etapa fenológica actual, Tipo de suelo, Historial reciente de tareas (último riego, última aplicación de agroquímicos/"curación").

## 2. Cultivos Prioritarios de Cuyo (Extraídos de Calendario INTA y Biblioteca Croply)

| Cultivo | Temporada INTA | Variedades Referencia (INTA) | Forma de Siembra | Meses Siembra | Marco Plantación (cm) | Ciclo Total (Días a cosecha) | Particularidades en Cuyo (Entrevistas/INTA) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tomate** | Primavera-Verano | Redondo / Perita (HD) | Almácigo y Trasplante | Almácigo: Ago-Sep / Trasplante: Oct-Nov | 30-50 x 70 | 80 - 100 días | Muy sensible a heladas tardías (Oct) y golpe de calor (>35°C en floración/cuaje). Alta demanda hídrica en fructificación. |
| **Ajo** | Otoño-Invierno | Colorado / Blanco (Cuyo) | Directa | Mar - Abr | 8-10 x 40 | 150 - 180 días (INTA) / hasta 240-270 días ciclo comercial largo | Debe arraigar antes del frío invernal. Requiere suelo suelto ("sedoso"). Sensible a exceso de humedad previo a cosecha (podredumbre). |
| **Lechuga** | Todo el año | Grand Rapids / Crimor (Primavera), Criolla (Prim-Ver), Gallega (Otoño), Maravilla 4 Est. | Almácigo y Trasplante | Todo el año | 20 x 20 (Criolla) / 25 x 25 (Mantecosa) | 50 días (Primavera) a 85 días (Otoño) | Raíz superficial; sensible a déficit hídrico rápido tras lluvias leves que solo mojan superficie. Golpe de calor induce floración prematura (bolting). |
| **Cebolla** | Otoño-Invierno | Bulbo / Verdeo | Almácigo o Directa | Bulbo: Mar-Abr / Verdeo: Feb-Jun | 10 x 40 (Bulbo) / 5 x 40 (Verdeo) | Bulbo: 270 días / Verdeo: 150 días | Alta relevancia en Mendoza/San Juan. Requiere corte de riego previo a cosecha de bulbo. |
| **Pimiento** | Primavera-Verano | Calahorra / Cuatro cascos | Almácigo y Trasplante | Almácigo: Jul-Ago / Trasplante: Oct | 40 x 70 | 80 - 100 días | Sensible a estrés hídrico y térmico (caída de flores y quemado de sol por alta radiación). |
| **Papa** | Primavera / Verano-Otoño | Spunta / Kennebec | Directa (Tubérculo) | Semitemprana / Tardía | 25-30 x 70-80 | ~120 días (4 meses) | Relevado en finca Guaymallén: requiere humedad específica en siembra, sensible a Tizón (hongos por alta humedad/temp) y heladas. |

*(Nota: Si se incluye **Vid** como cultivo extra por relevancia en bodegas de Cuyo, considerar ciclo perenne con brotación en Septiembre y vendimia en Febrero-Abril).*

## 3. Reglas de Negocio y Condiciones Críticas en Finca (Relevamiento Real)
1. **Error de humedad superficial:** Tras lluvias leves en Mendoza, la superficie parece húmeda pero la raíz sigue seca, o viceversa. El sensor de humedad de suelo (`humedad_suelo_pct`) es el factor decisivo para regar.
2. **Condiciones para "Curar" (Aplicar Agroquímicos):**
   - **Prohibido aplicar** si llueve, hay alta probabilidad de lluvia próxima (lavado de producto), **Viento Fuerte / Viento Zonda** (deriva del producto), o temperaturas extremas al mediodía.
   - **Ventana óptima:** Mañana temprano o atardecer, con viento leve (< 15 km/h), temperatura moderada (15°C - 28°C) y sin lluvias en las próximas 24 hs.
3. **Umbrales de Alertas Climáticas Automáticas (HU-NA-01):**
   - **Helada:** Temperatura mínima pronosticada <= 0°C (Severidad Alta). *Nota agronómica: En etapas sensibles como floración o trasplante de tomate/pimiento, entre 0°C y 2°C ya existe riesgo de helada agronómica.*
   - **Lluvia intensa:** Precipitación acumulada pronosticada >= 30 mm en 24h (Severidad Media).
   - **Viento fuerte / Zonda:** Velocidad de viento pronosticada >= 50 km/h (Severidad Alta). *Para pulverización, >= 15-20 km/h ya impide curar.*
   - **Ola de calor:** Temperatura máxima pronosticada >= 35°C durante >= 2 días consecutivos (Severidad Alta).