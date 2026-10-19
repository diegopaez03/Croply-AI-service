# Documento de Diseño Técnico de Machine Learning — Fase 2
## Arquitectura de Inferencia, Modelo Predictivo y Contratos de Integración de `analyzer.py`

**Proyecto:** Croply — Plataforma Inteligente de Monitoreo y Recomendación Agrícola  
**Módulo:** Servicio de Análisis Agronómico (`analyzer.py`)  
**Región de Operación:** Cuyo, Argentina (Mendoza / San Juan)  
**Versión del Documento:** 1.0.0  
**Fecha:** Octubre 2026  
**Autor:** Equipo de Arquitectura de IA y Machine Learning — Croply  
**Estado:** Aprobado para Implementación  

---

## 1. Arquitectura y Rol de `analyzer.py` en Croply

### 1.1 Diagrama de Flujo de Datos del Sistema

El ecosistema de inteligencia de Croply opera bajo una arquitectura de microservicios asíncronos y desacoplados sobre **FastAPI**. El módulo `analyzer.py` constituye el núcleo computacional analítico y determinista de la plataforma, intermediando entre las fuentes heterogéneas de telemetría y el agente conversacional generativo (`recommender.py`).

```mermaid
flowchart TD
    subgraph FuentesDatos["Fuentes de Datos Heterogéneas"]
        subgraph IoT["Telemetría Parcela"]
            S1["Croply IoT Simulator\n(MQTT Broker)"] --> InfluxDB[("InfluxDB 3 Clustered\nSeries Temporales")]
        end
        subgraph Clima["Servicio Meteorológico"]
            OM["Open-Meteo API / Estación Local"] --> PG_Weather[("PostgreSQL 16\ntabla: weather_records")]
        end
        subgraph Transaccional["Datos de Finca"]
            FincaUI["Gestión de Parcela"] --> PG_Finca[("PostgreSQL 16\ntablas: parcelas, cultivos,\nhistorial_riegos, historial_curas")]
        end
    end

    subgraph AnalyzerModule["FastAPI: analyzer.py (Cerebro Analítico & Determinista)"]
        Ingesta["Capa de Ingesta & Agregación\n(Pandas / Polars Pipeline)"]
        FeatureEng["Motor de Feature Engineering\n(Cálculo de 20 Features Vectorizadas)"]
        
        subgraph MotoresInferencia["Motor Híbrido de Decisión"]
            RulesEngine["Motor de Alertas Deterministas &\nRiesgo Fenológico (Predicción A - HU-NA-01)"]
            ML_LGBM["Clasificador LightGBM Multi-clase\nCalibrado (Predicción B - HU-NA-03)"]
            SHAP_Engine["Explicabilidad Local\n(SHAP TreeExplainer Nativo)"]
        end
        
        JSON_Builder["Generador de Contrato JSON Estructurado\n(Hechos, Probabilidades, Atribución, Guardrails)"]
    end

    subgraph RecommenderModule["FastAPI: recommender.py (Capa de Lenguaje Natural)"]
        PromptEngine["Ensamblador de Prompt Grounded\n(Few-Shot + Restricciones Agronómicas)"]
        OllamaEngine["Inferencia Local LLM\n(Ollama - Gemma 4 / 9B Instruct)"]
        ResponseValidator["Validador Pydantic de Salida"]
    end

    subgraph Consumidores["Capa de Presentación"]
        AppMobile["App Móvil Productor (Flutter)"]
        PushNotif["Servicio de Notificaciones Push"]
        Dashboard["Panel Web de Monitoreo"]
    end

    %% Conexiones
    InfluxDB --> Ingesta
    PG_Weather --> Ingesta
    PG_Finca --> Ingesta
    
    Ingesta --> FeatureEng
    FeatureEng --> RulesEngine
    FeatureEng --> ML_LGBM
    ML_LGBM --> SHAP_Engine
    
    RulesEngine --> JSON_Builder
    ML_LGBM --> JSON_Builder
    SHAP_Engine --> JSON_Builder
    
    JSON_Builder -->|"Payload JSON Estructurado\n(HTTP POST / IPC)"| PromptEngine
    PromptEngine --> OllamaEngine
    OllamaEngine --> ResponseValidator
    ResponseValidator --> AppMobile
    ResponseValidator --> PushNotif
    ResponseValidator --> Dashboard
```

### 1.2 Separación Estricta de Responsabilidades: Decoupled Decision-Generation Architecture

El diseño de Croply implementa una separación formal entre el **análisis cuantitativo y toma de decisiones agronómicas** y la **generación de lenguaje natural adaptada al usuario**:

```
+-----------------------------------------------------------------------------------+
|                           PRINCIPIO ARQUITECTÓNICO                                |
|  "Los modelos de lenguaje redactan; los modelos deterministas y de ML deciden"    |
+-----------------------------------------------------------------------------------+
```

#### Responsabilidades de `analyzer.py` (Cerebro Cuantitativo y Lógico)
1. **Determinismo y Rigor Numérico:** Ejecuta algoritmos reproducibles con latencia sub-10ms. No tolera variabilidad estocástica en umbrales de seguridad (ej. riesgo de congelamiento foliar a $0^\circ\text{C}$).
2. **Clasificación Agronómica:** Predice probabilísticamente la acción óptima mediante un clasificador de ensamble de gradiente (*LightGBM*), entrenado y validado contra bases de conocimiento agronómico de INTA Cuyo y FAO-56.
3. **Atribución Causal Transparente:** Calcula en tiempo real las contribuciones de cada feature a través de valores Shapley (`shap.TreeExplainer`), identificando inequívocamente el `sensor_origen_principal`.
4. **Barreras de Seguridad (Guardrails / Hard Constraints):** Determina qué acciones están taxativamente **prohibidas** bajo el contexto actual (ej. `prohibido_curar: true` por ráfagas de viento $\ge 15\text{ km/h}$, o `prohibido_regar: true` por saturación de perfil hídrico).
5. **Generación de Hechos Inmutables (Ground Truth):** Consolida métricas, predicción, alertas y restricciones en un payload JSON estrictamente tipado.

#### Responsabilidades de `recommender.py` (Capa de Presentación y Empatía Humana)
1. **Recepción Pasiva de Hechos:** Consume el JSON provisto por `analyzer.py`. Tiene prohibido recalcular métricas, desestimar alertas o inferir estados no declarados en el payload.
2. **Blindaje Anti-Alucinaciones:** El system prompt de **Gemma 4** en **Ollama** se inicializa con restricciones duras derivadas directamente del JSON (`Hard Agronomic Constraints`). El LLM no puede recomendar curar si el analyzer marcó `prohibido_curar: true`, eliminando de raíz la alucinación operativa.
3. **Adaptación Lingüística Contextual:** Transforma códigos técnicos (ej. `ALERTA_VIENTO_FUERTE_ZONDA`, `POSTERGAR_CURACION_CLIMA`) en una narrativa agrotécnica empática, clara y adaptada a la terminología cotidiana del productor de Cuyo (referenciando "curar", "zonda", "raíz superficial", "riego por goteo/surco").
4. **Estructuración Conversacional:** Formatea la respuesta con consejos de mitigación práctica, pasos a seguir inmediatos y alertas visibles sin alterar la verdad del dato.

---

## 2. Predicción A — Alertas y Riesgo Climático (HU-NA-01)

El sistema de alertas climáticas tempranas implementa una arquitectura híbrida de dos niveles: **Alertas Deterministas Inmediatas** (basadas en umbrales micrometeorológicos críticos) y la **Matriz de Riesgo Climático Fenológico** (que pondera la susceptibilidad biológica del cultivo en su etapa actual).

### 2.1 Alertas Deterministas de Seguridad Inmediata

Estas alertas se activan mediante reglas deterministas booleanas ejecutadas al ingestarse cada nuevo paquete de telemetría IoT o registro de pronóstico de Open-Meteo (resolución horaria a 48-72 horas):

| Código de Alerta | Condición Lógica Trigger | Severidad Base | Fundamento Agronómico / Físico |
| :--- | :--- | :--- | :--- |
| **`ALERTA_HELADA`** | $T_{min}^{pronostico} \le 0.0^\circ\text{C}$ *(o $T_{min} \le 2.0^\circ\text{C}$ con punto de rocío $\le 0^\circ\text{C}$ en etapas sensibles)* | **ALTA** | Cristalización intracelular del agua en los tejidos vegetales, ruptura de membranas celulares y muerte de meristemas y flores. Frecuente en Cuyo en heladas tardías de primavera (septiembre-octubre). |
| **`ALERTA_LLUVIA_INTENSA`** | $\text{Precip}_{24h}^{pronostico} \ge 30.0\text{ mm}$ | **MEDIA** | Saturación de macroporos del suelo, anoxia radicular temporal, asfixia en cultivos no tolerantes (ajo, cebolla), lixiviación de nitratos hacia napas freáticas e imposibilidad de mecanización agrícola. |
| **`ALERTA_VIENTO_FUERTE_ZONDA`** | $(V_{viento} \ge 50.0\text{ km/h})$ $\lor$ $(V_{viento} \ge 35.0\text{ km/h} \land \text{HR} < 20\% \land \Delta T_{24h} \ge +8.0^\circ\text{C})$ | **ALTA** | El viento Zonda (efecto Föhn en la precordillera andina) provoca una caída drástica de la humedad relativa y una subida violenta de temperatura, acelerando la evapotranspiración a niveles de shock, quebrando ramas y haciendo inviable cualquier aplicación fitosanitaria. |
| **`ALERTA_OLA_CALOR`** | $T_{max}^{pronostico} \ge 35.0^\circ\text{C}$ por $\ge 2$ días consecutivos | **ALTA** | Desnaturalización de proteínas enzimáticas, caída masiva de flores y botones en solanáceas, marchitamiento temporal y cierre estomático prolongado que detiene la fotosíntesis neta. |

### 2.2 Matriz de Riesgo Climático Fenológico

El impacto de un fenómeno meteorológico adverso no es absoluto: depende estrictamente de la vulnerabilidad ontogénica del cultivo en el momento de ocurrencia. El riesgo se formaliza como el producto de la **Amenaza Meteorológica ($A$)** por la **Vulnerabilidad Fenológica ($V$)**:

$$\text{Riesgo} = \mathcal{M}(A, V) \in \{\text{BAJO}, \text{MEDIO}, \text{ALTO}, \text{CRÍTICO}\}$$

```
+-----------------------------+-------------------------------------------------------------------------+
|                             |                           ETAPA FENOLÓGICA                              |
| EVENTO CLIMÁTICO            | Emergencia /    Vegetativo     Floración /    Llenado /    Maduración /|
|                             | Trasplante      Temprano       Cuaje          Fructif.     Reposo      |
+-----------------------------+-------------------------------------------------------------------------+
| Helada (T_min <= 0°C)       | CRÍTICO         ALTO           CRÍTICO        ALTO         BAJO (Ajo)  |
| Lluvia Intensa (>= 30 mm)   | ALTO            MEDIO          MEDIO          MEDIO        CRÍTICO     |
| Viento Fuerte / Zonda       | ALTO            MEDIO          CRÍTICO        MEDIO        BAJO        |
| Ola de Calor (>= 35°C x 2d) | ALTO            MEDIO          CRÍTICO        ALTO         BAJO        |
+-----------------------------+-------------------------------------------------------------------------+
```

#### Ejemplos Concretos por Cultivo en la Región de Cuyo

1. **Tomate (*Solanum lycopersicum*) — Floración vs. Cuaje vs. Maduración:**
   - *Floración ($T_{max} \ge 35^\circ\text{C}$):* **Riesgo CRÍTICO.** El polen se vuelve inviable por encima de $32^\circ\text{C}-35^\circ\text{C}$, provocando aborto floral masivo y pérdida irreversible del primer y segundo piso de frutos.
   - *Floración / Cuaje ($T_{min} \le 1.5^\circ\text{C}$):* **Riesgo CRÍTICO.** Aún sin alcanzar $0^\circ\text{C}$ en termómetro de abrigo meteorológico, la radiación nocturna a nivel de canopia genera daño celular en flores tiernas (helada agronómica).
   - *Maduración / Precosecha ($\text{Precip}_{24h} \ge 30\text{ mm}$):* **Riesgo CRÍTICO.** El influjo hídrico repentino en frutos maduros causa rajado apical/longitudinal (*cracking*) y proliferación fulminante de *Botrytis cinerea* y *Alternaria solani*.

2. **Ajo (*Allium sativum*) — Reposo Invernal vs. Cosecha:**
   - *Reposo / Brotación invernal ($T_{min} \le -3^\circ\text{C}$):* **Riesgo BAJO.** El ajo morado/blanco de Cuyo requiere acumulación de horas de frío ($< 7^\circ\text{C}$) durante el invierno para inducir la bulbificación posterior (vernalización). Tolera heladas sin merma de rendimiento.
   - *Llenado tardío / Precosecha ($\text{Precip}_{24h} \ge 30\text{ mm}$ o riego excesivo en noviembre):* **Riesgo CRÍTICO.** Genera podredumbre blanca de la raíz y bulbo (*Sclerotium cepivorum*) y manchas bacterianas en las túnicas protectoras, descalificando el producto para empaque de exportación.

3. **Lechuga (*Lactuca sativa*) — Etapa Comercial:**
   - *Ola de Calor ($T_{max} \ge 32^\circ\text{C}$):* **Riesgo CRÍTICO.** Estimula la senescencia y la elongación prematura del tallo floral (*bolting* o espigado), tornando las hojas amargas por acumulación de sesquiterpenos lactónicos e inutilizando comercialmente la parcela.

---

## 3. Predicción B — Clasificador de Acción Agronómica (HU-NA-03)

El modelo de clasificación multi-clase de `analyzer.py` predice la **acción agronómica prioritaria** que el productor o administrador de riego debe ejecutar o supervisar en las siguientes 24 a 48 horas.

### 3.1 Definición de las 7 Clases Target y Severidades Asociadas

```
+---------------------------------+---------------+---------------------------------------------+
| CLASE TARGET                    | SEVERIDAD     | DESCRIPCIÓN OPERATIVA SINTÉTICA             |
+---------------------------------+---------------+---------------------------------------------+
| 1. REGAR_URGENTE                | CRÍTICA       | Déficit hídrico severo próximo a PMP        |
| 2. REGAR_PROGRAMADO             | MEDIA         | Entrada regular en umbral de recarga        |
| 3. SUSPENDER_RIEGO              | ALTA / MEDIA  | Perfil saturado o lluvia inminente          |
| 4. VENTANA_OPTIMA_CURAR         | BAJA (INFO)   | Micrometeorología ideal para fitosanitarios |
| 5. POSTERGAR_CURACION_CLIMA     | ALTA          | Viento, calor o lluvia impiden curar        |
| 6. CORREGIR_PH_SUELO            | MEDIA         | Bloqueo nutricional por pH descalibrado     |
| 7. ESTADO_OPTIMO_SIN_ACCION     | INFORMATIVA   | Todos los parámetros en cono de confort     |
+---------------------------------+---------------+---------------------------------------------+
```

### 3.2 Criterios Agronómicos y de Negocio por Clase

#### 1. `REGAR_URGENTE`
- **Severidad:** **CRÍTICA**.
- **Criterios Cuantitativos:**
  - $\text{humedad\_suelo\_pct} \le \text{PMP} + 3\%$ (típicamente $\le 18\%$ en suelos franco-arenosos, o $\le 24\%$ en franco-arcillosos).
  - O $\text{deficit\_hidrico} \ge 40\text{ mm}$ con evapotranspiración diaria prevista $ET_o \cdot K_c \ge 5.0\text{ mm/día}$.
  - Y pronóstico de precipitaciones $\text{precip\_acum\_pronostico\_24h} < 5.0\text{ mm}$.
- **Justificación Agronómica:** El potencial mátrico del suelo ha descendido a niveles donde la raíz no puede vencer la tensión superficial del agua. Si no se irriga en un lapso de $< 6-12$ horas, el cultivo entra en marchitamiento permanente, produciendo necrosis de puntas de crecimiento y aborto irreversible de órganos reproductivos.

#### 2. `REGAR_PROGRAMADO`
- **Severidad:** **MEDIA**.
- **Criterios Cuantitativos:**
  - $\text{humedad\_suelo\_pct}$ se ubica entre el Límite de Agotamiento Permisible (MAD, *Management Allowed Depletion* según FAO-56, típicamente $45\% - 55\%$ de la Capacidad de Campo) y la Capacidad de Campo ($CC$).
  - Precipitaciones pronosticadas en 48h $< 10\text{ mm}$.
  - Días transcurridos desde el último riego $\ge$ ciclo de diseño del sistema (ej. 3-4 días en goteo de tomate, 7-10 días en surco).
- **Justificación Agronómica:** Mantenimiento de la curva de reposición hídrica para sostener la tasa transpiratoria máxima sin generar estrés por falta de agua ni lavado de nutrientes por percolación profunda.

#### 3. `SUSPENDER_RIEGO`
- **Severidad:** **ALTA** (por anoxia o riesgo fitopatológico) o **MEDIA** (por economía de agua).
- **Criterios Cuantitativos:**
  - $\text{humedad\_suelo\_pct} \ge 85\%$ de la capacidad de saturación, O $\text{precip\_acum\_pronostico\_24h} \ge 15.0\text{ mm}$.
  - O etapa fenológica de precosecha en ajo/cebolla ($\text{dias\_desde\_siembra} \ge 220$ o etapa = 4).
- **Justificación Agronómica:** El suelo ya se encuentra en o por encima de Capacidad de Campo. Regar bajo estas condiciones provocaría encharcamiento, anoxia radicular por expulsión del oxígeno del espacio poroso, pudrición de cuello y raíces por *Phytophthora* o *Fusarium*, y desperdicio energético del bombeo.

#### 4. `VENTANA_OPTIMA_CURAR`
- **Severidad:** **INFORMATIVA / BAJA**.
- **Criterios Cuantitativos:**
  - Necesidad técnica de aplicación: $\text{dias\_ultima\_curacion} \ge 14$ días (o alerta endémica de plagas/hongos activa).
  - Viento: $V_{viento} \le 12.0\text{ km/h}$ (óptimo 4 a 8 km/h para dispersión homogénea sin estancamiento ni deriva).
  - Temperatura ambiental: $15.0^\circ\text{C} \le T_{amb} \le 27.0^\circ\text{C}$.
  - Humedad relativa ambiental: $\text{HR} \ge 45\%$.
  - Indicador psicrométrico: $2.0^\circ\text{C} \le \Delta T \le 8.0^\circ\text{C}$.
  - Lluvia: $\text{precip\_acum\_pronostico\_24h} = 0.0\text{ mm}$ (garantía de secado y fijación del caldo en el follaje).
- **Justificación Agronómica:** Ventana ideal donde las gotas asperjadas no sufren deriva por viento, no se evaporan prematuramente antes de contactar el haz foliar y no corren riesgo de lavado pluvial en las 24 horas posteriores a la pulverización.

#### 5. `POSTERGAR_CURACION_CLIMA`
- **Severidad:** **ALTA**.
- **Criterios Cuantitativos:**
  - Existencia de intención o necesidad de curación ($\text{dias\_ultima\_curacion} \ge 14$), PERO al menos una de las siguientes variables está en rango hostil:
    - $V_{viento} \ge 15.0\text{ km/h}$ (deriva de agroquímicos a fincas colindantes o atmósfera).
    - $\text{precip\_acum\_pronostico\_24h} \ge 5.0\text{ mm}$ (lavado activo del producto).
    - $T_{amb} \ge 30.0^\circ\text{C}$ (evaporación súbita de la gota y riesgo severo de fitotoxicidad / quemado de hojas).
    - $\Delta T \ge 9.0^\circ\text{C}$ o alerta de Viento Zonda activa.
- **Justificación Agronómica:** La pulverización en estas condiciones constituye una pérdida económica total del producto fitosanitario, incumple las normas de buenas prácticas agrícolas (BPA) y pone en riesgo legal y ambiental a la explotación por contaminación fuera de blanco.

#### 6. `CORREGIR_PH_SUELO`
- **Severidad:** **MEDIA / ALTA**.
- **Criterios Cuantitativos:**
  - $\text{ph\_suelo} < 5.8$ (acidez crítica) $\lor$ $\text{ph\_suelo} > 7.8$ (alcalinidad típica de suelos calcáreos y salinos de Cuyo).
- **Justificación Agronómica:** Fuera del rango óptimo ($6.2 - 7.2$), se produce insolubilización química de macro y micronutrientes. A pH $> 7.8$, el hierro ($Fe$), fósforo ($P$) y zinc ($Zn$) precipitan como sales insolubles, provocando clorosis férrica y raquitismo aun cuando el suelo esté fertilizado. Requiere aplicación de azufre elemental, ácido fosfórico o enmiendas ácidas en el agua de riego.

#### 7. `ESTADO_OPTIMO_SIN_ACCION`
- **Severidad:** **INFORMATIVA**.
- **Criterios Cuantitativos:**
  - $\text{humedad\_suelo\_pct}$ en banda de confort ($65\% - 80\%$ de CC).
  - $\text{ph\_suelo}$ equilibrado ($6.2 \le \text{pH} \le 7.4$).
  - Sin necesidad inmediata de riego ni curas atrasadas.
  - Ausencia absoluta de alertas climáticas deterministas activas.
- **Justificación Agronómica:** El lote vegeta en balance homeostático; intervenir mecánicamente o agregar insumos constituiría un costo innecesario y posible disturbio del suelo.

---

## 4. Diccionario Completo de Features de Entrada

El pipeline de ingesta procesa y normaliza un vector de **20 features numéricas y categóricas codificadas**, diseñadas para alimentar tanto el modelo `LightGBM` como los árboles de decisión y reglas deterministas de `analyzer.py`.

### 4.1 Tabla de Especificación Técnica de Features

| N° | Nombre Exacto de Feature | Tipo de Dato | Rango / Valores Válidos | Unidades | Fuente Primaria | Fórmula Matemática / Método de Cálculo |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | `temp_ambiente_c` | `float32` | `[-10.0, 50.0]` | °C | InfluxDB 3 (IoT) | Promedio de lecturas térmicas en ventana de 1 hora: $\bar{T} = \frac{1}{N}\sum_{i=1}^N T_i$ |
| **2** | `humedad_ambiente_pct` | `float32` | `[0.0, 100.0]` | % | InfluxDB 3 (IoT) | Humedad relativa instantánea o media móvil de 1h del sensor higrómetro capacitivo |
| **3** | `humedad_suelo_pct` | `float32` | `[0.0, 100.0]` | % (Moisture) | InfluxDB 3 (IoT) | Lectura promedio de sondas FDR/capacitivas situadas en el bulbo húmedo radicular efectivo |
| **4** | `radiacion_solar_wm2` | `float32` | `[0.0, 1400.0]` | W/m² | InfluxDB 3 (IoT) | Flujo radiativo global medido por piranómetro en parcela |
| **5** | `ph_suelo` | `float32` | `[3.0, 11.0]` | Adimensional | InfluxDB 3 / Muestreo | Medición potenciométrica directa de iones de hidrógeno en solución de suelo |
| **6** | `lluvia_acum_1h_mm` | `float32` | `[0.0, 150.0]` | mm | InfluxDB 3 (IoT) | Suma de pulsos del pluviómetro de cazoleta basculante en los últimos 60 minutos |
| **7** | `temp_min_pronostico_48h` | `float32` | `[-15.0, 35.0]` | °C | PostgreSQL (`weather_records`) | $\min_{t \in [t_0, t_0+48h]} T_{pronostico}(t)$ provisto por API Open-Meteo |
| **8** | `temp_max_pronostico_48h` | `float32` | `[5.0, 50.0]` | °C | PostgreSQL (`weather_records`) | $\max_{t \in [t_0, t_0+48h]} T_{pronostico}(t)$ provisto por API Open-Meteo |
| **9** | `precip_acum_pronostico_24h` | `float32` | `[0.0, 200.0]` | mm | PostgreSQL (`weather_records`) | $\sum_{t=t_0}^{t_0+24h} \text{Precip}(t)$ acumulada esperada |
| **10** | `velocidad_viento_max_48h` | `float32` | `[0.0, 150.0]` | km/h | PostgreSQL (`weather_records`) | $\max_{t \in [t_0, t_0+48h]} V_{viento}(t)$ ráfaga/velocidad máxima prevista |
| **11** | `dias_desde_siembra` | `int32` | `[0, 365]` | días | PostgreSQL (`parcelas`) | $\Delta t = \text{FechaActual} - \text{FechaSiembra}$ |
| **12** | `dias_ultimo_riego` | `int32` | `[0, 60]` | días | PostgreSQL (`historial_riegos`) | $\Delta t = \text{FechaActual} - \text{FechaUltimoRiegoRegistrado}$ |
| **13** | `dias_ultima_curacion` | `int32` | `[0, 120]` | días | PostgreSQL (`historial_curas`) | $\Delta t = \text{FechaActual} - \text{FechaUltimaAplicacionFitosanitaria}$ |
| **14** | `etapa_fenologica_idx` | `int32` | `{0, 1, 2, 3, 4}` | Categórico Ordinal | PostgreSQL (`parcelas`) | Mapeo fenológico: `0`: Emergencia/Trasplante, `1`: Vegetativo, `2`: Floración/Cuaje, `3`: Fructificación, `4`: Maduración/Senescencia |
| **15** | `kc_cultivo` | `float32` | `[0.2, 1.35]` | Adimensional | Derivada / FAO-56 | Look-up table polinómico según `cultivo_id` y `etapa_fenologica_idx` (Tablas Allen et al., FAO-56) |
| **16** | `eto_fao56_estimada` | `float32` | `[0.0, 15.0]` | mm/día | Derivada / Calculada | Ecuación simplificada de Hargreaves-Samani / Penman-Monteith horaria según radiación y temperatura |
| **17** | `deficit_hidrico` | `float32` | `[-50.0, 100.0]` | mm | Derivada / Calculada | Balance de agua en el suelo: lámina faltante para alcanzar Capacidad de Campo ($CC - \theta_w$) |
| **18** | `delta_t_psicrometrico` | `float32` | `[0.0, 20.0]` | °C | Derivada / Calculada | Depresión psicrométrica: diferencia termodinámica $\Delta T = T_{seca} - T_{humeda}$ |
| **19** | `humedad_suelo_media_24h` | `float32` | `[0.0, 100.0]` | % | Derivada / InfluxDB 3 | Media móvil integral: $\frac{1}{24}\int_{t-24h}^t \text{humedad\_suelo}(\tau) d\tau$ |
| **20** | `variacion_humedad_suelo_24h` | `float32` | `[-50.0, 50.0]` | % / 24h | Derivada / Calculada | Gradiente temporal de secado/infiltración: $\theta_{suelo}(t) - \theta_{suelo}(t - 24h)$ |

---

### 4.2 Ingeniería de Features Clave

#### A. Déficit Hídrico de la Parcela (`deficit_hidrico`)
El déficit hídrico expresa cuantitativamente el volumen de agua (en milímetros de lámina equivalente) que debe reponerse en el estrato radicular para llevar el contenido volumétrico de agua ($\theta$) desde su valor medido actual hasta la Capacidad de Campo ($\theta_{CC}$), integrando el coeficiente del cultivo:

$$\text{déficit\_hídrico (mm)} = \max\left(0, \, (\theta_{CC} - \theta_{\text{suelo\_actual}}) \times Z_r \times 10\right) + \left(ET_o \times K_c - \text{Lluvia}_{\text{efectiva}}\right)$$

Donde:
- $\theta_{CC}$: Humedad a Capacidad de Campo del tipo de suelo (ej. $0.28\text{ cm}^3/\text{cm}^3$ en suelos francos de Maipú/Luján).
- $\theta_{\text{suelo\_actual}}$: Contenido de agua volumétrico medido por el sensor (`humedad_suelo_pct` $/ 100$).
- $Z_r$: Profundidad efectiva del sistema radicular según cultivo y etapa (en metros, e.g. $0.3\text{ m}$ en lechuga, $0.6-0.8\text{ m}$ en tomate maduro).
- $ET_o \times K_c$: Evapotranspiración del cultivo ($ET_c$) calculada para el ciclo diurno.
- $\text{Lluvia}_{\text{efectiva}}$: Fracción pluvial que realmente infiltra en el bulbo radicular sin escurrir ni evaporarse ($0.8 \times \text{lluvia}$ si $\text{lluvia} > 5\text{ mm}$).

#### B. Delta T Psicrométrico (`delta_t_psicrometrico`)
El parámetro $\Delta T$ es el estándar agronómico mundial adoptado para evaluar la viabilidad física de la pulverización de agroquímicos (Miller & Ellis, 2000). Representa la evaporatividad del aire y se calcula como la diferencia entre la temperatura de bulbo seco ($T_{seco} = \text{temp\_ambiente\_c}$) y la temperatura de bulbo húmedo ($T_{humedo}$):

$$\Delta T = T_{seco} - T_{humedo}$$

La temperatura de bulbo húmedo se deriva numéricamente mediante la aproximación psicrométrica de Stull (2011):

$$T_{humedo} = T \cdot \arctan\left(0.151977 \sqrt{\text{HR} + 8.313659}\right) + \arctan(T + \text{HR}) - \arctan(\text{HR} - 1.676331) + 0.00391838 \cdot \text{HR}^{3/2} \arctan(0.023101 \cdot \text{HR}) - 4.686035$$

##### Interpretación Operativa de $\Delta T$ para Pulverizaciones:
- **$\Delta T < 2.0^\circ\text{C}$ (Zona de Riesgo por Alta Humedad):** El aire está casi saturado. Las gotas pulverizadas no se evaporan pero tampoco secan sobre las hojas, favoreciendo el escurrimiento y la formación de rocío que lava el producto activo.
- **$2.0^\circ\text{C} \le \Delta T \le 8.0^\circ\text{C}$ (Ventana Óptima de Aplicación):** Vida media óptima de la gota en vuelo. Permite que el caldo alcance el estrato inferior de la canopia y se deposite eficazmente.
- **$\Delta T > 8.0^\circ\text{C}$ (Zona de Alta Evaporación y Pérdida):** En condiciones de Zonda o aire seco cuyano, las gotas menores a 150 micrones se evaporan en menos de 2 segundos antes de tocar la planta, generando nubes de deriva de polvo químico y fitotoxicidad grave por sobredemanda osmótica foliar.

#### C. Agregaciones Temporales de Suelo de 24 Horas
Los sensores capacitivos de bajo costo sufren desviaciones dieléctricas causadas por las oscilaciones térmicas diarias del suelo (efecto día/noche en la constante dieléctrica del agua). La ingeniería de features mitiga este ruido mediante dos variables cinéticas:
1. **`humedad_suelo_media_24h`:** Filtra los ciclos circadianos y variaciones diurnas espurias, reflejando el verdadero stock hídrico radicular.
2. **`variacion_humedad_suelo_24h`:** Diferencia entre el valor actual y el de hace 24 horas exactas ($\theta_t - \theta_{t-24h}$). Un gradiente fuertemente negativo indica alta tasa de transpiración/secado; un gradiente positivo sostenido indica infiltración activa tras riego o lluvia.

---

## 5. Comparativa y Selección de Algoritmos de Machine Learning

Para determinar el clasificador de acción agronómica de `analyzer.py`, se evaluaron exhaustivamente cinco familias de algoritmos bajo las restricciones operativas de la infraestructura de Croply.

### 5.1 Matriz Comparativa de Algoritmos

| Criterio de Selección | LightGBM (GOSS) | XGBoost (Exact/Hist) | Random Forest | Decision Tree (CART) | Redes Neuronales (MLP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Latencia Inferencia (SLA < 10ms)** | **$\approx 1.2\text{ ms}$** | $\approx 3.8\text{ ms}$ | $\approx 8.5\text{ ms}$ | $\approx 0.3\text{ ms}$ | $\approx 4.5\text{ ms}$ (CPU) |
| **Consumo de RAM en Runtime** | **$\approx 25\text{ MB}$** | $\approx 85\text{ MB}$ | $\approx 180\text{ MB}$ | $\approx 10\text{ MB}$ | $\approx 150\text{ MB}$ (PyTorch/TF) |
| **Explicabilidad Nativa (SHAP TreeExplainer)** | **Nativo y Óptimo ($O(TLD^2)$)** | Nativo y Rápido | Nativo pero Lento en bosques grandes | Reglas booleanas simples | KernelExplainer lento ($O(M \cdot N)$) |
| **Manejo de Desbalance de Clases** | **Excelente (`is_unbalance` / `focal_loss`)** | Excelente (`scale_pos_weight`) | Regular (Requiere balanceo previo) | Pobre (Sesgo a clases mayoritarias) | Requiere funciones de pérdida complejas |
| **Tolerancia a Valores Faltantes (NaN)** | **Nativo (bifurcación óptima)** | Nativo | No (Requiere imputación forzada) | Requiere imputación | No (Falla ante NaN) |
| **Tamaño de Serialización (.joblib/.txt)** | **$\approx 2.5\text{ MB}$** | $\approx 12.0\text{ MB}$ | $\approx 65.0\text{ MB}$ | $\approx 0.5\text{ MB}$ | $\approx 25.0\text{ MB}$ |
| **Puntuación Global (1 a 10)** | **9.8 / 10** | 8.9 / 10 | 7.2 / 10 | 5.8 / 10 | 6.5 / 10 |

### 5.2 Justificación Técnica Formal de la Elección de LightGBM

Se seleccionó formalmente **LightGBM** (Light Gradient Boosting Machine, Ke et al., 2017) como el motor de clasificación de `analyzer.py` en base a cuatro fundamentos de ingeniería de software y agronomía computacional:

1. **Estrategia de Crecimiento *Leaf-wise* con Límite de Profundidad:**
   A diferencia de los árboles tradicionales que crecen nivel por nivel (*depth-wise/level-wise*), LightGBM divide el nodo que produce la mayor reducción de pérdida cuadrática o entropía cruzada (*leaf-wise*). Al restringir `max_depth` (ej. 6 niveles) y `num_leaves` (ej. 31), el modelo aprende fronteras de decisión no lineales hiperprecisas entre clases complejas (como `VENTANA_OPTIMA_CURAR` vs `POSTERGAR_CURACION_CLIMA`) con mínimas iteraciones de ensamble.

2. **Inferencia de Ultra Baja Latencia y Mínimo Consumo de Memoria:**
   Gracias al agrupamiento de valores de features continuas en histogramas discretos (*Histogram-based algorithm*) y la exclusión de características mutuamente excluyentes (*Exclusive Feature Bundling* - EFB), el modelo compilado pesa menos de $3\text{ MB}$ y evalúa una muestra en $\approx 1.2\text{ milisegundos}$ en un entorno de CPU convencional (FastAPI en contenedor Docker), cumpliendo con holgura el SLA de $< 10\text{ ms}$.

3. **Integración Óptima con SHAP TreeExplainer:**
   El algoritmo *TreeSHAP* (Lundberg et al., 2020) está optimizado nativamente en C++ para la estructura de punteros de nodos de LightGBM. Esto permite calcular los valores Shapley locales exactos para las 20 features en menos de $2.5\text{ ms}$ por inferencia, permitiendo identificar al instante cuál fue el `sensor_origen_principal` y desglosar el impacto porcentual de cada métrica sin ralentizar la respuesta de la API.

4. **Robustez ante Fallas de Sensores de Campo (Valores Faltantes):**
   En entornos agrícolas reales, los sensores IoT periódicamente pierden conexión o envían valores nulos por caídas de batería solar o radioenlace LoRaWAN. LightGBM asigna de manera intrínseca las muestras con valores faltantes al brazo de división que minimiza el error durante el entrenamiento, garantizando que el sistema no colapse ante una eventual caída de un sensor secundario.

---

## 6. Contrato de Interfaz JSON (API Contract)

### 6.1 Esquema JSON de Entrada de `analyzer.py`

Este payload representa la estructura de datos que el endpoint `/api/v1/analyze` de `analyzer.py` recibe tras la consulta consolidada a InfluxDB 3 y PostgreSQL 16:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AnalyzerInputPayload",
  "type": "object",
  "required": [
    "parcela_id",
    "timestamp_utc",
    "cultivo",
    "telemetria_iot",
    "pronostico_meteo",
    "datos_historicos_finca"
  ],
  "properties": {
    "parcela_id": { "type": "string", "format": "uuid" },
    "finca_id": { "type": "string", "format": "uuid" },
    "timestamp_utc": { "type": "string", "format": "date-time" },
    "cultivo": {
      "type": "object",
      "required": ["nombre", "variedad", "etapa_fenologica", "etapa_fenologica_idx", "dias_desde_siembra"],
      "properties": {
        "nombre": { "type": "string", "enum": ["Tomate", "Ajo", "Lechuga", "Cebolla", "Pimiento", "Papa"] },
        "variedad": { "type": "string" },
        "etapa_fenologica": { "type": "string" },
        "etapa_fenologica_idx": { "type": "integer", "minimum": 0, "maximum": 4 },
        "dias_desde_siembra": { "type": "integer", "minimum": 0 }
      }
    },
    "telemetria_iot": {
      "type": "object",
      "required": [
        "temp_ambiente_c",
        "humedad_ambiente_pct",
        "humedad_suelo_pct",
        "radiacion_solar_wm2",
        "ph_suelo",
        "lluvia_acum_1h_mm",
        "humedad_suelo_media_24h",
        "variacion_humedad_suelo_24h"
      ],
      "properties": {
        "temp_ambiente_c": { "type": "number", "minimum": -20.0, "maximum": 60.0 },
        "humedad_ambiente_pct": { "type": "number", "minimum": 0.0, "maximum": 100.0 },
        "humedad_suelo_pct": { "type": "number", "minimum": 0.0, "maximum": 100.0 },
        "radiacion_solar_wm2": { "type": "number", "minimum": 0.0, "maximum": 1500.0 },
        "ph_suelo": { "type": "number", "minimum": 3.0, "maximum": 11.0 },
        "lluvia_acum_1h_mm": { "type": "number", "minimum": 0.0 },
        "humedad_suelo_media_24h": { "type": "number", "minimum": 0.0, "maximum": 100.0 },
        "variacion_humedad_suelo_24h": { "type": "number", "minimum": -100.0, "maximum": 100.0 }
      }
    },
    "pronostico_meteo": {
      "type": "object",
      "required": [
        "temp_min_pronostico_48h",
        "temp_max_pronostico_48h",
        "precip_acum_pronostico_24h",
        "velocidad_viento_max_48h"
      ],
      "properties": {
        "temp_min_pronostico_48h": { "type": "number" },
        "temp_max_pronostico_48h": { "type": "number" },
        "precip_acum_pronostico_24h": { "type": "number", "minimum": 0.0 },
        "velocidad_viento_max_48h": { "type": "number", "minimum": 0.0 }
      }
    },
    "datos_historicos_finca": {
      "type": "object",
      "required": ["dias_ultimo_riego", "dias_ultima_curacion", "tipo_suelo"],
      "properties": {
        "dias_ultimo_riego": { "type": "integer", "minimum": 0 },
        "dias_ultima_curacion": { "type": "integer", "minimum": 0 },
        "tipo_suelo": { "type": "string", "enum": ["Arenoso", "Franco-arenoso", "Franco", "Franco-arcilloso", "Arcilloso"] }
      }
    }
  }
}
```

---

### 6.2 Esquema JSON de Salida de `analyzer.py` hacia `recommender.py`

Este payload es la salida generada por `analyzer.py`. Contiene la predicción calibrada, la descomposición explicable mediante SHAP y los **umbrales agronómicos inviolables** para blindar a Gemma 4 contra alucinaciones durante la generación del mensaje al productor:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AnalyzerOutputPayload",
  "type": "object",
  "required": [
    "analisis_id",
    "timestamp_procesamiento",
    "parcela_contexto",
    "prediccion_ml",
    "explicabilidad_shap",
    "alertas_climaticas",
    "restricciones_duras_llm"
  ],
  "properties": {
    "analisis_id": { "type": "string", "format": "uuid" },
    "timestamp_procesamiento": { "type": "string", "format": "date-time" },
    "latencia_ms": { "type": "number" },
    "parcela_contexto": {
      "type": "object",
      "required": ["parcela_id", "cultivo", "etapa_fenologica", "dias_desde_siembra"],
      "properties": {
        "parcela_id": { "type": "string", "format": "uuid" },
        "cultivo": { "type": "string" },
        "etapa_fenologica": { "type": "string" },
        "dias_desde_siembra": { "type": "integer" }
      }
    },
    "prediccion_ml": {
      "type": "object",
      "required": ["accion_recomendada", "severidad", "confianza_probabilidad", "distribucion_probabilidades"],
      "properties": {
        "accion_recomendada": {
          "type": "string",
          "enum": [
            "REGAR_URGENTE",
            "REGAR_PROGRAMADO",
            "SUSPENDER_RIEGO",
            "VENTANA_OPTIMA_CURAR",
            "POSTERGAR_CURACION_CLIMA",
            "CORREGIR_PH_SUELO",
            "ESTADO_OPTIMO_SIN_ACCION"
          ]
        },
        "severidad": { "type": "string", "enum": ["CRITICA", "ALTA", "MEDIA", "BAJA", "INFORMATIVA"] },
        "confianza_probabilidad": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "distribucion_probabilidades": {
          "type": "object",
          "additionalProperties": { "type": "number" }
        }
      }
    },
    "explicabilidad_shap": {
      "type": "object",
      "required": ["sensor_origen_principal", "valor_sensor_principal", "unidad_sensor_principal", "top_factores_influencia"],
      "properties": {
        "sensor_origen_principal": { "type": "string" },
        "valor_sensor_principal": { "type": "number" },
        "unidad_sensor_principal": { "type": "string" },
        "top_factores_influencia": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["feature", "valor_medido", "shap_value", "impacto"],
            "properties": {
              "feature": { "type": "string" },
              "valor_medido": { "type": "number" },
              "shap_value": { "type": "number" },
              "impacto": { "type": "string", "enum": ["AUMENTA_PROBABILIDAD", "DISMINUYE_PROBABILIDAD"] }
            }
          }
        }
      }
    },
    "alertas_climaticas": {
      "type": "object",
      "required": ["alertas_activas", "riesgo_fenologico_global"],
      "properties": {
        "alertas_activas": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["codigo_alerta", "severidad", "descripcion_evento", "valor_gatillo"],
            "properties": {
              "codigo_alerta": { "type": "string" },
              "severidad": { "type": "string" },
              "descripcion_evento": { "type": "string" },
              "valor_gatillo": { "type": "string" }
            }
          }
        },
        "riesgo_fenologico_global": { "type": "string", "enum": ["BAJO", "MEDIO", "ALTO", "CRITICO"] }
      }
    },
    "restricciones_duras_llm": {
      "type": "object",
      "required": ["prohibido_regar", "prohibido_curar", "justificacion_restricciones", "variables_criticas"],
      "properties": {
        "prohibido_regar": { "type": "boolean" },
        "prohibido_curar": { "type": "boolean" },
        "justificacion_restricciones": { "type": "array", "items": { "type": "string" } },
        "variables_criticas": {
          "type": "object",
          "properties": {
            "deficit_hidrico_mm": { "type": "number" },
            "delta_t_psicrometrico_c": { "type": "number" },
            "horas_proxima_ventana_optima": { "type": ["integer", "null"] }
          }
        }
      }
    }
  }
}
```

---

### 6.3 Ejemplo Completo y Realista de Payload JSON de Salida

**Escenario de Campo Real en Cuyo:** Finca de tomate en Maipú, Mendoza, en plena etapa de floración y cuaje. Se registra una disminución sostenida de la humedad del suelo y un frente pronosticado con ráfagas secas compatibles con viento Zonda y temperaturas elevadas:

```json
{
  "analisis_id": "7f8b9e12-4c56-42d8-9a3e-bfa1e4d3a011",
  "timestamp_procesamiento": "2026-10-15T14:30:00Z",
  "latencia_ms": 3.42,
  "parcela_contexto": {
    "parcela_id": "b1a2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
    "cultivo": "Tomate",
    "etapa_fenologica": "Floración y Cuaje",
    "dias_desde_siembra": 58
  },
  "prediccion_ml": {
    "accion_recomendada": "REGAR_URGENTE",
    "severidad": "CRITICA",
    "confianza_probabilidad": 0.9124,
    "distribucion_probabilidades": {
      "REGAR_URGENTE": 0.9124,
      "REGAR_PROGRAMADO": 0.0512,
      "SUSPENDER_RIEGO": 0.0011,
      "VENTANA_OPTIMA_CURAR": 0.0004,
      "POSTERGAR_CURACION_CLIMA": 0.0210,
      "CORREGIR_PH_SUELO": 0.0089,
      "ESTADO_OPTIMO_SIN_ACCION": 0.0050
    }
  },
  "explicabilidad_shap": {
    "sensor_origen_principal": "humedad_suelo_pct",
    "valor_sensor_principal": 17.4,
    "unidad_sensor_principal": "%",
    "top_factores_influencia": [
      {
        "feature": "humedad_suelo_pct",
        "valor_medido": 17.4,
        "shap_value": 2.4510,
        "impacto": "AUMENTA_PROBABILIDAD"
      },
      {
        "feature": "deficit_hidrico",
        "valor_medido": 38.6,
        "shap_value": 1.8320,
        "impacto": "AUMENTA_PROBABILIDAD"
      },
      {
        "feature": "temp_max_pronostico_48h",
        "valor_medido": 36.5,
        "shap_value": 0.9840,
        "impacto": "AUMENTA_PROBABILIDAD"
      },
      {
        "feature": "velocidad_viento_max_48h",
        "valor_medido": 54.0,
        "shap_value": -0.4120,
        "impacto": "DISMINUYE_PROBABILIDAD"
      }
    ]
  },
  "alertas_climaticas": {
    "alertas_activas": [
      {
        "codigo_alerta": "ALERTA_VIENTO_FUERTE_ZONDA",
        "severidad": "ALTA",
        "descripcion_evento": "Ráfagas secas pronosticadas >= 50 km/h con salto térmico",
        "valor_gatillo": "54.0 km/h (Rachas) - HR 16%"
      },
      {
        "codigo_alerta": "ALERTA_OLA_CALOR",
        "severidad": "ALTA",
        "descripcion_evento": "Temperaturas máximas sostenidas >= 35°C por 48h en floración",
        "valor_gatillo": "36.5°C máx pronosticada"
      }
    ],
    "riesgo_fenologico_global": "CRITICO"
  },
  "restricciones_duras_llm": {
    "prohibido_regar": false,
    "prohibido_curar": true,
    "justificacion_restricciones": [
      "PROHIBIDO FUMIGAR O CURAR: Viento en ráfagas de 54 km/h excede el límite máximo de 15 km/h, produciendo deriva total y evaporación violenta de gotas (Delta T psicrométrico = 10.2°C).",
      "PRIORIDAD ABSOLUTA DE RIEGO: Déficit hídrico de 38.6 mm en plena floración con riesgo crítico de aborto floral por choque térmico y estrés mátrico del suelo."
    ],
    "variables_criticas": {
      "deficit_hidrico_mm": 38.6,
      "delta_t_psicrometrico_c": 10.2,
      "horas_proxima_ventana_optima": null
    }
  }
}
```

---

## 7. Referencias Técnicas y Bibliográficas

1. **Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998).** *Crop evapotranspiration: Guidelines for computing crop water requirements.* FAO Irrigation and Drainage Paper No. 56. Roma, Italia: FAO.
2. **Instituto Nacional de Tecnología Agropecuaria (INTA Cuyo) (2022).** *Guía de Buenas Prácticas Agrícolas y Manejo del Riego en Hortalizas de la Región Cuyana.* Mendoza, Argentina: Ediciones INTA.
3. **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017).** *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 3146–3154.
4. **Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020).** *From local explanations to global understanding with explainable AI for trees.* Nature Machine Intelligence, 2(1), 56–67.
5. **Miller, P. C. H., & Ellis, M. B. (2000).** *Effects of formulation on spray droplet characteristics and drift.* Aspects of Applied Biology, 57, 167–174.
6. **Stull, R. (2011).** *Wet-Bulb Temperature from Relative Humidity and Air Temperature.* Journal of Applied Meteorology and Climatology, 50(11), 2267–2269.
