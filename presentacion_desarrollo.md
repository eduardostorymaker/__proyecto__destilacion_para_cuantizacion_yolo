# Desarrollo de la presentación del primer avance

**Proyecto:** ¿Protege la destilación de conocimiento a los detectores YOLO ligeros de la pérdida por cuantización?
Un estudio controlado en la detección de daños viales (RDD2022).

Este documento desarrolla los 5 temas de [presentacion.md](presentacion.md). Cada tema incluye:
- el objetivo del segmento;
- las diapositivas sugeridas, con su contenido;
- el guion para el expositor;
- las figuras y tablas que se pueden usar (todas existen en el repositorio);
- las preguntas probables del público, con su respuesta.

Todas las cifras salen del repositorio (`results/`), con corte al primer avance del 2026-10-31.

**Convención I / J:** India (**I**) y Japón (**J**) se presentan juntos, en las mismas tablas y figuras, y las
observaciones se etiquetan con I o J. Japón solo aparece donde ya tiene datos. Hasta ahora son los datos (J1), el
maestro (J2), los alumnos base (J3), la verificación de la KD (J4a) y el piloto de λ (J4b), todos en
**validación**. Japón aún no tiene alumnos KD, cuantización ni resultados en prueba, así que el tema 4 es solo de I.

---

## Tema 1. Introducción, estado del arte y dataset

**Objetivo del segmento:** que el público entienda el problema y los conceptos (YOLO, cuantización, destilación,
calibración), por qué la pregunta es nueva y con qué datos se responde.

### Diapositiva 1.1 — El problema

- Inspeccionar carreteras con cámaras de bajo costo (teléfonos, cámaras embarcadas) exige detectar daños **en el
  dispositivo**, sin depender de un servidor.
- Esos dispositivos tienen poca memoria y poco cómputo, así que se necesitan modelos **pequeños y rápidos**.
- Hay dos técnicas habituales para lograrlo:
  - **Cuantización:** pasar el modelo de números de 32 bits (FP32) a enteros de 8 bits (INT8). Es más rápido, pero
    pierde precisión.
  - **Destilación de conocimiento (KD):** un modelo grande (maestro) le enseña a uno pequeño (alumno).
- **Pregunta central:** si un alumno aprende de un maestro, ¿resiste mejor la cuantización?

**Guion:** en 2018, Maeda *et al.* crearon un dataset de daños viales en Japón con teléfonos inteligentes. En 2022 se
publicó RDD2022, con 47 420 imágenes de seis países, lo que impulsó el uso de detectores YOLO para esta tarea. El
cuello de botella ya no es solo la precisión, sino desplegar el modelo en hardware barato.

### Diapositiva 1.2 — Detección de objetos con YOLO

- YOLO detecta en **una sola etapa**: una red predice a la vez las cajas y las clases, lo que permite tiempo real.
- YOLO11 tiene tres partes:
  - **backbone**, que extrae características;
  - **neck**, que fusiona escalas;
  - **cabeza de detección** (Detect).
- Incorpora bloques C3k2 y atención espacial C2PSA.
- Viene en escalas de nano (n) a extra grande (x). En este trabajo:
  - **Alumno:** YOLO11n, con 2.58 M de parámetros y 6.4 GFLOPs.
  - **Maestro:** YOLO11m.
- La cabeza recibe tres mapas de características (**P3, P4 y P5**), uno por escala. Ahí se aplica la destilación.

### Diapositiva 1.3 — Cómo se mide un detector

| Métrica | Qué significa |
|---|---|
| IoU = \|Bp ∩ Bg\| / \|Bp ∪ Bg\| | Solapamiento entre la caja predicha y la real; acierto si supera un umbral |
| Precisión P = TP / (TP + FP) | De lo que detecta, cuánto es correcto (pocas falsas alarmas) |
| Recall R = TP / (TP + FN) | De lo que existe, cuánto detecta (pocos daños omitidos) |
| AP | Área bajo la curva precisión–recall de una clase |
| mAP@0.5 | Promedio del AP de las 4 clases con IoU 0.5 (estilo PASCAL VOC). **Métrica principal** |
| mAP@0.5:0.95 | Promedio sobre 10 umbrales de IoU, de 0.5 a 0.95 (estilo COCO): exige cajas más precisas |

**Guion:** la mAP@0.5 es la métrica principal y la mAP@0.5:0.95 se reporta como complemento. En los baches (D40) se
reporta su AP por separado, porque es la clase de mayor interés práctico.

### Diapositiva 1.4 — Cuantización

- La cuantización uniforme convierte un valor real *x* en un entero *q* con una escala *s* y un punto cero *z*:
  `q = clamp(round(x/s) + z, qmin, qmax)` y se reconstruye como `x̂ = s·(q − z)`.
- **FP32 → INT8:** ocupa 4 veces menos memoria y permite aritmética entera. A cambio, el redondeo y el recorte meten
  ruido.
- La escala puede ser **por tensor** (una para toda la capa) o **por canal** (una por cada filtro). La versión por
  canal tiene menos error.
- **FP16** (media precisión): ocupa la mitad y es habitual en GPU.
- Hay dos enfoques:

| | PTQ (post-entrenamiento) | QAT (consciente del entrenamiento) |
|---|---|---|
| Qué hace | Cuantiza un modelo ya entrenado; estima los rangos con un pequeño conjunto de calibración | Reentrena simulando la cuantización, para que la red compense el error |
| Costo | Minutos, sin reentrenar | Un ajuste fino corto, pero hay que entrenar |
| Precisión | Pierde más | Suele recuperar más (0.8–2.3 puntos de mAP@0.5 frente a la PTQ en COCO, Jabbar *et al.*) |

**Ejemplo para el público:** Jabbar *et al.* midieron una caída de 3.2 puntos de mAP@0.5 al aplicar PTQ INT8 a YOLOv5s.

### Diapositiva 1.5 — Destilación de conocimiento y CWD

- Hinton *et al.*: el alumno imita las probabilidades "suavizadas" del maestro, que contienen información sobre la
  similitud entre clases.
- Tipos de destilación: basada en **respuestas** (salidas), en **características** (activaciones intermedias) o en
  **relaciones**.
- **Brecha de capacidad:** si el maestro es mucho más grande que el alumno, la destilación pierde eficacia
  (Mirzadeh *et al.*).
- **CWD (destilación por canales, Shu *et al.*, ICCV 2021):**
  - Convierte cada canal del mapa de activación en una distribución espacial con softmax y temperatura τ.
  - Minimiza la divergencia KL entre maestro y alumno, canal por canal. Así el alumno aprende **dónde mira** el
    maestro en cada canal.
  - `L_CWD = (τ²/C) Σ_c Σ_i φ(y^T_ci) · log(φ(y^T_ci) / φ(y^S_ci))`
- **Pérdida total del alumno:** `L = L_det + λ·L_CWD`.

### Diapositiva 1.6 — Calibración de la confianza (ECE)

- Un modelo está **bien calibrado** si su confianza coincide con su tasa de aciertos: de las detecciones con
  confianza 0.8, alrededor del 80 % deberían ser correctas.
- Las redes modernas tienden a ser **sobreconfiadas** (Guo *et al.*, 2017).
- **ECE** = Σ_m (\|B_m\|/n) · \|acc(B_m) − conf(B_m)\|, con M = 15 intervalos de confianza.
- En un detector, la confianza decide qué cajas pasan el umbral. Una mala calibración se traduce en **falsas alarmas
  o daños no detectados**.

### Diapositiva 1.7 — ¿Por qué podrían relacionarse KD y cuantización?

- La cuantización es una **perturbación** sobre pesos y activaciones.
- **Hipótesis:** si la KD produce un alumno con salidas más suaves y mejor calibradas, sus decisiones podrían ser más
  estables ante esa perturbación.
- Evidencia previa, solo en clasificación: Polino *et al.* y Mishra y Marr destilan entrenando al alumno directamente
  en baja precisión, lo que se parece más a la QAT que a la PTQ.
- Este trabajo destila **antes** de cuantizar, para separar el efecto de cada técnica. Usa CWD (adecuada para
  detección) e INT8 con herramientas estándar (TensorRT, OpenVINO, ModelOpt).

### Diapositiva 1.8 — Estado del arte (búsqueda PRISMA)

- Búsqueda estructurada en **Scopus** (8 de octubre de 2026), en título, resumen y palabras clave:
  - daños viales **AND** YOLO **AND** (destilación **OR** cuantización);
  - **OR** estudios con RDD2022;
  - artículos y conferencias de 2022 a 2026.
- Se obtuvieron **250 registros** y se incluyeron **10 estudios** tras dos etapas de cribado (PRISMA 2020).
- Figura: `paper/figures/prisma_flujo.pdf`.
- Lo que encontró la búsqueda:
  - **Solo destilación:** Jiang *et al.* (poda + CWD en YOLOv7), Meng *et al.* (maestros asistentes, YOLOv8), Cong
    *et al.* (JS, YOLOv8n), Xiao *et al.*, SBP-YOLO. Todos evalúan la KD **en precisión completa**.
  - **Solo cuantización:** SHE-YOLO (Kendryte K230, ~26 FPS; degradación "comparable a la del modelo base") y
    Graph-MambaRoadDet (INT8, 45 FPS en una Jetson Orin Nano). Ninguno usa QAT.
  - **RDD2022 sin compresión:** Babawarun *et al.* (India, transferencia a Sudáfrica), MSRD-DET (Japón).
  - **Ningún estudio** combina KD y cuantización comparando un alumno destilado con uno sin destilar.

### Diapositiva 1.9 — Antecedente más cercano y brecha

- Fuera del dominio, **Karjol y Hanna (2026, arXiv:2604.26857, preprint)**, en BDD100K (conducción):
  - la KD **no** redujo la pérdida de mAP al cuantizar (0.030 frente a 0.031), aunque conservó más precisión en INT8;
  - una sola corrida y solo PTQ.
- **Brecha:** en daños viales nadie compara un detector destilado con uno sin destilar tras cuantizar, ni evalúa la QAT.
- Además, los daños viales podrían ser **más sensibles** a la cuantización: objetos pequeños y de bajo contraste, clases
  minoritarias (baches) y modelos nano.

### Diapositiva 1.10 — Objetivo y preguntas de investigación

**Objetivo general:** evaluar si la destilación de conocimiento reduce la pérdida de precisión y conserva la
calibración de la confianza en detectores YOLO cuantizados con PTQ y QAT para la detección de daños viales.

| | Pregunta |
|---|---|
| **P1** | ¿La KD reduce la pérdida de mAP por cuantización INT8? |
| **P2** | ¿El alumno destilado conserva mejor la calibración (ECE) tras cuantizar? |
| **P3** | ¿Un alumno destilado con PTQ (barato) alcanza a un alumno base con QAT (costosa)? |

**Objetivos específicos:**
1. Entrenar un YOLO11n con y sin destilación a partir de un maestro YOLO11m, en India y Japón, y comparar su
   desempeño en FP32.
2. Cuantizar ambos alumnos con PTQ y QAT (INT8) y medir la pérdida de mAP, global y por clase, con énfasis en los
   baches.
3. Evaluar si el alumno destilado conserva una mejor calibración (ECE) tras cuantizar.
4. Determinar si el alumno destilado con PTQ no es inferior al alumno base con QAT.
5. Medir la eficiencia de los modelos en GPU y CPU.

### Diapositiva 1.11 — Dataset RDD2022: India (I) y Japón (J)

- Se usa la partición de **entrenamiento** de cada país, porque el test oficial no tiene etiquetas.
- **Por qué dos países:** los resultados de un subconjunto de RDD2022 no se transfieren a otro (Babawarun *et al.*),
  así que se comprueba si los efectos se repiten. Ambos usan el mismo protocolo y el mismo código.

| Clase | Daño | Instancias I | Instancias J |
|---|---|---|---|
| D00 | Grieta longitudinal | 1 555 | 4 049 |
| D10 | Grieta transversal | 68 | 3 979 |
| D20 | Piel de cocodrilo | 2 021 | 6 198 |
| D40 | Bache | 3 187 | 2 243 |
| **Total** | | **6 831** | **16 469** |

| | I | J |
|---|---|---|
| Imágenes | 7 706 (todas de 720 × 720) | 10 506 (de 540, 600 y 1 024 px) |
| Cajas en los XML originales | 8 203 | 24 754 |
| Cajas eliminadas por ser de otras clases (regla 1) | 1 372 (D44, D01, D43, D11, D50 y 1 ambigua) | 8 284 (D44, D50 y D43) |
| Cajas eliminadas por medir menos de 1 px (regla 3) | 0 | 1 |
| Imágenes eliminadas | 0 | 0 |
| Imágenes de fondo | 4 483 (**58.2 %**) | 2 606 (**24.8 %**) |

- **Reglas de limpieza**, aplicadas en orden y contadas, iguales en I y J:
  1. Las cajas de otras clases se eliminan, pero la imagen se conserva.
  2. Las cajas fuera de la imagen se recortan.
  3. Las cajas de menos de 1 px se eliminan.
  4. De los duplicados se deja una caja.
  5. Solo se eliminan imágenes corruptas.
  6. Las imágenes sin cajas se **conservan como fondo**.
- **Observaciones:**
  - **I:** D10 es casi inexistente (68 instancias) y predominan los baches.
  - **J:** las 4 clases están mucho más equilibradas, con D10 abundante y D20 como clase mayoritaria.
  - **I** tiene más del doble de proporción de fondo que **J**.
- Figura: `results/figures/presentacion/distribucion_clases.png` (I y J lado a lado).

### Diapositiva 1.12 — Partición y desafíos

- 70/15/15 con **estratificación iterativa multietiqueta** (Sechidis *et al.*, 2011; semilla 42), que mantiene la
  proporción de cada clase en las tres particiones:

| Partición | Imágenes | D00 | D10 | D20 | D40 | Fondo |
|---|---|---|---|---|---|---|
| I · entrenamiento | 5 396 | 1 097 | 50 | 1 417 | 2 241 | 3 138 |
| I · validación | 1 144 | 231 | 9 | 304 | 482 | 673 |
| I · prueba | 1 166 | 227 | 9 | 300 | 464 | 672 |
| J · entrenamiento | 7 346 | 2 833 | 2 794 | 4 354 | 1 560 | 1 824 |
| J · validación | 1 595 | 611 | 594 | 919 | 322 | 391 |
| J · prueba | 1 565 | 605 | 591 | 925 | 361 | 391 |

Las columnas de clase son instancias; Fondo son imágenes sin daños. En los dos países la estratificación deja entre
el 69 y el 74 % de cada clase en entrenamiento y entre el 13 y el 16 % en validación y en prueba.

- **Desafíos:**
  - **I:** D10 tiene solo 9 instancias en prueba, así que su AP es inestable. Se mantiene (borrarla convertiría las
    grietas transversales en "fondo") y se agrega un análisis **sin D10**.
  - **J:** D10 tiene 591 instancias en prueba, así que el análisis sin D10 es solo una comprobación.
  - **I:** más de la mitad de las imágenes son de fondo (58 %); **J:** una de cada cuatro (25 %).
  - Calibración INT8: 500 imágenes de entrenamiento en cada país. En **I** solo 4 tienen D10; en **J**, 107.
- Fuente: `results/tables/paper/tabla_III_particion.md` y `data_audit.csv` (I: `results/tables/`, J:
  `results/japan/tables/`).

### Preguntas probables (tema 1)

- **¿Por qué YOLO11n y no un modelo más pequeño o más grande?** Es la escala típica de despliegue en borde. El maestro
  YOLO11m es la escala intermedia, que evita una brecha de capacidad excesiva.
- **¿Por qué no se descarta D10 en I?** Si se borran sus etiquetas pero se conservan las imágenes, las grietas
  transversales pasan a ser "fondo" y confunden el aprendizaje de D00.

---

## Tema 2. Flujo de la metodología planteada

**Objetivo del segmento:** explicar **qué** se hace y en qué orden, con un diagrama, y justificar cada decisión de
diseño. Es la vista conceptual; el tema 3 explica cómo se ejecutó.

### Diapositiva 2.1 — Diagrama general

Figura para la diapositiva: `results/figures/presentacion/diagrama_flujo.png`. Abajo, la misma idea en texto.

```
RDD2022 (India / Japón)
   │ limpieza + auditoría + partición 70/15/15 estratificada
   ▼
Maestro YOLO11m (FP32) ─────────────── congelado ───────────────┐
   │                                                             ▼
   ├─► Alumno BASE YOLO11n (solo L_det) × 3 semillas     Alumno KD YOLO11n (L_det + λ·L_CWD) × 3 semillas
   │                                                     (λ elegido en un piloto)
   ▼                                                             ▼
         ┌───────────────── best.pt FP32 de cada alumno ─────────────────┐
         ▼                    ▼                     ▼                     ▼
   TensorRT FP16      TensorRT INT8 (PTQ)     ModelOpt PTQ INT8     QAT INT8 (ModelOpt)
                      OpenVINO INT8 (CPU)     (control)             (ajuste fino 30 épocas)
         └───────────────────────────── ▼ ──────────────────────────────┘
               Evaluación en test: mAP, AP por clase, ECE (mismas predicciones)
                                        ▼
               Estadística preespecificada: bootstrap pareado + Holm (H1a–H3)
                                        ▼
               Eficiencia: latencia, FPS y tamaño (GPU y CPU)
```

**Guion:** la única diferencia entre los dos alumnos es la pérdida de destilación. Todo lo demás (datos,
hiperparámetros, semillas y cuantización) es idéntico, por eso es un **estudio controlado**.

### Diapositiva 2.2 — Paso 1: datos

- Limpieza con reglas explícitas y auditoría con cada conteo registrado.
- Partición estratificada y fija (semilla 42); las listas de imágenes se publican.
- **500 imágenes de calibración** tomadas solo de entrenamiento (estratificadas): **nunca** se calibra con validación
  ni con prueba.

### Diapositiva 2.3 — Paso 2: maestro y alumnos base

- **Maestro YOLO11m**, inicializado con COCO.
- Se probó también un **YOLO11l** con una regla fijada antes del resultado: se usaría solo si superaba al 11m en
  ≥ 0.010 de mAP@0.5:0.95 de validación.
  - Obtuvo 0.144 frente a 0.147, así que **se mantuvo el YOLO11m**.
  - Figura: `results/figures/teachers_m_vs_l.png`.
- **J:** solo YOLO11m (decisión previa, porque en I el 11l no mejoró).
- **Alumnos base YOLO11n:** 3 semillas (0, 1 y 2), entrenados solo con la pérdida de detección.

**Maestro frente a alumnos base, en validación** (best.pt revalidado: batch 8, conf 0.001, IoU 0.7; alumnos: media ±
d.e. de 3 semillas):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | AP@0.5 D40 | mAP@0.5 sin D10 |
|---|---|---|---|---|
| I · Maestro YOLO11m (best.pt, época 148) | 0.345 | 0.147 | 0.437 | 0.449 |
| I · Alumnos base | 0.342 ± 0.022 | 0.145 ± 0.006 | 0.397 ± 0.005 | 0.424 ± 0.020 |
| J · Maestro YOLO11m (best.pt, época 81 de 131) | 0.571 | 0.277 | 0.583 | 0.614 |
| J · Alumnos base | 0.553 ± 0.001 | 0.262 ± 0.001 | 0.570 ± 0.002 | 0.595 ± 0.002 |

- **I:** el maestro apenas supera a los alumnos (+0.003 de mAP@0.5), una diferencia **menor** que la d.e. entre
  semillas (0.022). Sin D10, la ventaja sube a +0.025.
- **J:** el maestro supera a los alumnos en las 4 clases (+0.018 de mAP@0.5), una ventaja **unas 15 veces mayor** que la
  d.e. entre semillas (0.001).
- **J** es un conjunto más "aprendible": más datos, menos fondo y D10 abundante. Su maestro alcanza 0.571 de mAP@0.5,
  frente a 0.345 en **I**.
- Esta diferencia es clave para la réplica: en **J** el maestro sí tiene algo que enseñar.

### Diapositiva 2.4 — Paso 3: destilación CWD

- CWD sobre las **tres entradas de la cabeza** (P3, P4 y P5).
- El maestro está **congelado** (sin gradientes) y recibe **el mismo batch aumentado** que el alumno.
- **Adaptadores 1 × 1:** igualan los canales del alumno (64, 128 y 256) con los del maestro (256, 512 y 512). Se
  entrenan con el alumno y se **descartan** al exportar, así que el alumno destilado tiene exactamente la misma
  arquitectura y el mismo costo que el base.
  - Se verificó con `check_kd`: el best.pt del destilado tiene los mismos 499 tensores que el base y el mismo número
    de parámetros en ONNX (2 624 974).
  - **I:** se verificó en la prueba de humo y en los 3 alumnos KD finales (TODO OK).
  - **J:** se verificó en la prueba de humo con el maestro de Japón (TODO OK). Los adaptadores son los mismos
    (214 272 parámetros que se descartan).
- **λ** se elige con un piloto (siguiente diapositiva).

### Diapositiva 2.5 — Elección de λ (piloto)

- Se probaron λ ∈ {0.5, 1.0, 2.0}, con semilla 0 y 50 épocas. Se elige el de mayor mAP@0.5:0.95 de **validación**
  (nunca prueba).

| λ | I · val mAP@0.5:0.95 | J · val mAP@0.5:0.95 | I · val mAP@0.5 | J · val mAP@0.5 | I · AP D40 | J · AP D40 | Peso de λ·L_CWD (I / J) |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.134 | 0.242 | 0.312 | 0.517 | 0.325 | 0.509 | 4.3 % / 4.2 % |
| **1.0** | **0.144** | **0.250** | 0.328 | 0.526 | 0.375 | 0.525 | 7.6 % / 7.4 % |
| 2.0 | 0.130 | 0.246 | 0.303 | 0.528 | 0.374 | 0.509 | 13.8 % / 13.0 % |

- En **I** y en **J** se eligió **λ = 1.0**, por la regla fijada de antemano (mayor mAP@0.5:0.95 de validación).
- **I:** λ = 1.0 gana con claridad (+0.010 sobre el segundo).
- **J:** las diferencias son menores (+0.004 sobre λ = 2.0, que tiene la mejor mAP@0.5 por 0.002). λ = 1.0 también
  gana en baches.
- En **I** y en **J** la pérdida CWD baja de forma estable (J: de 1.37 a 0.35 con λ = 1.0) y pesa lo mismo frente a la
  pérdida de detección. La escala de la destilación es igual en ambos países.
- Figuras: `results/figures/pilot_lambda.png` (I) y `results/japan/figures/pilot_lambda.png` (J).

### Diapositiva 2.6 — Paso 4: rutas de cuantización

| Ruta | Herramienta | Para qué |
|---|---|---|
| FP16 | TensorRT | Referencia de media precisión |
| INT8 PTQ | TensorRT, con el exportador de Ultralytics | Uso práctico típico (H1a, H2a, H3) |
| INT8 PTQ (CPU) | OpenVINO + NNCF | Despliegue en CPU |
| INT8 PTQ-MO | NVIDIA ModelOpt, sin ajuste fino | **Control:** separa el efecto del esquema de cuantización del efecto del ajuste fino |
| INT8 QAT | ModelOpt + 30 épocas de ajuste fino | H1b, H2b, H3 |

- Todas parten del **mismo best.pt FP32**. FP16 e INT8 se generan por separado, sin encadenar FP32 → FP16 → INT8.
- El maestro también se cuantiza, como referencia.

### Diapositiva 2.7 — Decisiones de diseño (por qué así)

- **¿Por qué destilar antes de cuantizar?** Para separar los efectos. La PTQ produce modelos de solo inferencia, y la
  PTQ y la QAT parten de un modelo FP32.
- **¿Por qué la QAT sin destilación?** Para responder si la ventaja de *haber sido destilado* persiste cuando la red
  aprende a compensar el error de cuantización. Destilar durante la QAT mezclaría ambos efectos.
- **¿Por qué el control PTQ-MO?** La QAT usa ModelOpt, que cuantiza distinto que TensorRT (Q/DQ explícito, otro
  calibrador, otras capas en INT8). Sin el control, no se sabría si la mejora viene del ajuste fino o del esquema.
- **¿Por qué 3 semillas?** El resultado de un solo entrenamiento varía. En validación, la d.e. de los alumnos base
  fue de 0.022 de mAP@0.5 en **I** y de 0.001 en **J**: la variabilidad depende del dataset, y no se sabe de antemano.

### Diapositiva 2.8 — Pasos 5 y 6: evaluación y comparación

- Todas las combinaciones (modelo × semilla × motor × precisión) se evalúan **en prueba** con el mismo procedimiento.
- Se miden mAP@0.5, mAP@0.5:0.95, AP por clase, P, R y ECE. El ECE se calcula con las **mismas predicciones** que la
  mAP.
- Contrastes estadísticos **preespecificados** (H1a–H3, detallados en el tema 3) y eficiencia (latencia, FPS y tamaño).
- **Réplica:** el flujo completo se repite en Japón con el mismo código (`--dataset japan`).

### Preguntas probables (tema 2)

- **¿Los adaptadores no le dan ventaja al alumno destilado?** No en inferencia, porque se descartan. Solo existen
  durante el entrenamiento.
- **¿Por qué el maestro recibe el mismo batch aumentado?** Para que ambos vean exactamente la misma imagen. Si no, la
  comparación de características no tendría sentido.

---

## Tema 3. Metodología de experimentación computacional

**Objetivo del segmento:** mostrar **cómo** se ejecutó para que sea válido y reproducible: entorno, protocolo de
entrenamiento, cuantización, evaluación, estadística y eficiencia.

### Diapositiva 3.1 — Entorno

| Componente | Especificación |
|---|---|
| GPU / CPU | NVIDIA RTX 4060 de 8 GB / Intel Core i7-12700KF |
| RAM | 32 GB (24 GB asignados a WSL2) |
| Sistema | Windows 11 + WSL2 (Ubuntu 26.04) |
| Software | Python 3.13, PyTorch 2.14.0 (CUDA 13.0), Ultralytics 8.4.164 (fijado: define el calibrador INT8), TensorRT 10.16.1, OpenVINO 2026.4.0 (NNCF 3.4.0), ModelOpt 0.47.0; driver 616.92 |

- Los datos están en el disco de Linux (no en `/mnt`) para leer más rápido.
- Se usa TensorRT 10 y no 11, porque la 11 elimina la calibración implícita que usa el exportador de Ultralytics.

### Diapositiva 3.2 — Pruebas de tiempo y decisión de AMP (Fase 1b)

- Antes de entrenar se midió el costo por época (batch 8):

| Caso | s/época | VRAM máx. |
|---|---|---|
| Maestro YOLO11m, AMP | 123 s | 4.2 GB |
| Alumno YOLO11n, AMP | 33 s | 1.5 GB |
| Maestro, FP32 | ~7 veces más lento | satura los 8 GB |

- **Decisión:** precisión mixta (AMP) para todos los modelos.
- AMP **no es cuantización**: los pesos se guardan en FP32, y ese best.pt es el punto de partida de toda la
  cuantización.

### Diapositiva 3.3 — Protocolo de entrenamiento

- **Hiperparámetros comunes:**
  - 640 × 640, 300 épocas;
  - SGD (lr0 0.01, lrf 0.01, momentum 0.937, weight decay 0.0005, 3 épocas de calentamiento);
  - batch 8 con acumulación hasta un batch nominal de 64.
- **Aumentación** por defecto de Ultralytics (mosaico, desactivado en las últimas 10 épocas; HSV; traslación; escala;
  volteo), sin extras.
- **Reproducibilidad:**
  - `deterministic=True`, con semillas fijadas en Python, NumPy y PyTorch;
  - caché en disco (la caché en RAM no es determinista);
  - `args.yaml` guardado en cada corrida.
- **Maestro:** paciencia de 50 épocas (parada temprana). **Alumnos:** 300 épocas completas, para que todos tengan el
  mismo presupuesto.
- **best.pt:** el criterio de Ultralytics 8.4.164, que es la **mAP@0.5:0.95 de validación**.
- Toda la configuración vive en `configs/*.yaml`; no hay valores fijos escondidos en el código.

### Diapositiva 3.3b — Semillas: qué se fija y para qué

Hay dos tipos de semillas: las que **fijan** una parte del experimento para que sea idéntica en todos los modelos, y
las que se **varían** a propósito para medir cuánto cambia el resultado de un entrenamiento a otro.

| Semilla | Valor | Qué controla | Por qué |
|---|---|---|---|
| Partición de datos | 42 (fija) | Qué imágenes van a entrenamiento, validación y prueba | Todos los modelos se entrenan y evalúan sobre exactamente las mismas imágenes |
| Imágenes de calibración | 42 (fija) | Las 500 imágenes de entrenamiento para calibrar el INT8 | Misma calibración para todos los modelos y para la QAT |
| Maestro | 0 (fija) | Entrenamiento del YOLO11m (una sola corrida) | El maestro es un insumo del experimento, no lo que se compara |
| **Alumnos (base y KD)** | **0, 1 y 2 (variadas)** | Inicialización de la cabeza, orden de los lotes y aumentaciones aleatorias | Medir la **variabilidad entre entrenamientos**: un solo entrenamiento podría salir "con suerte" |
| Piloto de λ | 0 | Las 3 corridas cortas del piloto | Comparar los λ con todo lo demás igual |
| QAT | La del alumno de origen | El ajuste fino de cada alumno | Cada modelo QAT hereda la semilla de su alumno |
| Bootstrap | 42 (fija) | Los 1 000 remuestreos de las imágenes de prueba | Mismos remuestreos para todos los modelos; resultado reproducible |
| Sensibilidad a la calibración (pendiente) | 43 y 44 | Dos conjuntos de calibración alternativos | Medir cuánto depende la pérdida INT8 de las imágenes de calibración elegidas |

- **Diseño pareado:** cada alumno destilado se compara con el alumno base **de su misma semilla** (base_s0 con kd_s0,
  etc.). Ambos parten del mismo estado aleatorio, así que la diferencia refleja principalmente la destilación. Las
  diferencias se calculan por semilla y luego se promedian.
- **Determinismo:** con `deterministic=True` y las semillas fijadas en Python, NumPy y PyTorch, repetir una corrida
  con la misma semilla reproduce el mismo modelo. Por eso se usa caché en disco y no en RAM, que en Ultralytics no es
  determinista.
- **Por qué 3 semillas y no 1:** en **I**, la mAP@0.5 de prueba (TensorRT FP32) de los alumnos base fue de 0.327,
  0.316 y 0.288 según la semilla, una diferencia de casi 4 puntos entre la mejor y la peor. Con los destilados fue de
  0.313, 0.322 y 0.314. Con una sola semilla, la conclusión podría depender de qué semilla tocó.
- **I frente a J, en validación** (alumnos base, mAP@0.5 por semilla s0, s1 y s2):
  - **I:** 0.366, 0.334 y 0.325 (rango de 4.1 puntos);
  - **J:** 0.554, 0.552 y 0.553 (rango de 0.2 puntos).
  En **J** la semilla casi no importa; en **I** sí. Por eso se usan 3 semillas en ambos.
- **Cómo se usan en el análisis:**
  - se reporta la media ± d.e. de las 3 semillas;
  - el bootstrap se aplica por semilla y luego se promedia;
  - se reporta el signo del efecto en cada semilla. Por ejemplo, H1a fue favorable a la KD en 2 de 3 semillas (+0.015,
    +0.004 y −0.007).
- **Por qué no más semillas:** cada alumno KD cuesta ~6.3 h de GPU en **I** y ~8.9 h en **J**. Tres semillas por grupo
  son el mínimo para estimar una d.e. dentro del presupuesto de cómputo. Es una limitación reconocida (diapositiva
  5.3, punto 6).

### Diapositiva 3.4 — Cuantización: detalles

- **Calibración:** 500 imágenes de entrenamiento, estratificadas e idénticas para todos los modelos.
- **TensorRT PTQ:**
  - calibrador **MinMax** implícito (la escala sale del máximo absoluto observado);
  - la sigmoide de la cabeza se mantiene en FP32, y las convoluciones de salida y la DFL, en FP16;
  - con el inspector de TensorRT se midió que **entre el 63 y el 68 % de las capas de cómputo quedan en INT8**
    (`results/tables/engine_layers.csv`).
- **OpenVINO:** PTQ de NNCF en CPU.
- **ModelOpt / QAT:**
  - calibrador de **máximo absoluto**, activaciones INT8 por tensor y pesos INT8 **por canal**;
  - calibración con las 500 imágenes sin aumentación;
  - ajuste fino de 30 épocas (10 % del entrenamiento), lr0 0.001, sin calentamiento ni mosaico; best.pt por
    validación;
  - se exporta a ONNX con nodos Q/DQ y de ahí a TensorRT INT8.
- **Trazabilidad:** cada engine se construye **una sola vez** y se registra con su huella **SHA-256**, que se verifica
  antes de evaluarlo o medirlo (`results/tables/engines.csv`). En total hay 45 engines.

### Diapositiva 3.5 — Protocolo de evaluación

- Siempre en **prueba**: 1 166 imágenes, con imgsz 640, batch 1, conf 0.001 e IoU de NMS 0.7. El conjunto de prueba
  no se miró antes de tiempo.
- **Hallazgo metodológico:** la validación por defecto de Ultralytics usa un letterbox *rectangular* (rect=True) y da
  1–2 puntos más de mAP@0.5 por el preprocesado, no por la cuantización. Por eso **todos** los modelos, incluido el
  .pt FP32, se evalúan con **letterbox cuadrado de 640 × 640**, igual que los engines estáticos.
- **Control de consistencia:** el .pt FP32 coincide con TensorRT FP32 (máx. \|Δ mAP@0.5\| = 0.0008 < 0.002).
- **ΔmAP dentro del mismo motor:** TRT FP32 − TRT {FP16, INT8, PTQ-MO, QAT} y OV FP32 − OV INT8.
- **ECE:** detecciones con confianza ≥ 0.25, emparejadas con las cajas reales por IoU ≥ 0.5 (de forma voraz, por
  confianza), correctas si coincide la clase; 15 intervalos.
- **P y R** se toman en el punto de máximo F1.

### Diapositiva 3.6 — Hipótesis preespecificadas

- **Convención de signo:** D > 0 favorece al alumno destilado. dmAP50 es la pérdida de mAP@0.5 al cuantizar.

| Pregunta | Hipótesis | Estadístico D | H0 | Tipo |
|---|---|---|---|---|
| P1 | H1a | dmAP50_base − dmAP50_KD, PTQ INT8 | D = 0 | Bilateral |
| P1 | H1b | dmAP50_base − dmAP50_KD, QAT INT8 | D = 0 | Bilateral |
| P2 | H2a | ECE_base − ECE_KD, PTQ INT8 | D = 0 | Bilateral |
| P2 | H2b | ECE_base − ECE_KD, QAT INT8 | D = 0 | Bilateral |
| P3 | H3 | mAP50(KD + PTQ) − mAP50(base + QAT) | D ≤ −0.010 | No inferioridad (δ = 1 punto) |

### Diapositiva 3.7 — Análisis estadístico

- **Bootstrap pareado:**
  - B = 1 000 remuestreos de las imágenes de prueba (semilla 42), con los **mismos índices para todos los modelos**;
  - cada diferencia se calcula por semilla (base_s frente a KD_s) y se promedia sobre las 3.
- **Valor p** (bootstrap centrado, Efron y Tibshirani):
  - bilateral: `p = (1 + #{|D*_b − D̂| ≥ |D̂|}) / (B + 1)`;
  - H3: `p = (1 + #{D*_b − D̂ ≥ D̂ + δ}) / (B + 1)`.
- **Corrección de Holm** sobre los 5 contrastes (α global = 0.05).
  - IC 95 % para H1a–H2b.
  - IC 90 % para H3: hay no inferioridad si su límite inferior supera −δ.
- **Verificaciones:**
  - sin remuestreo, la implementación reproduce la mAP de Ultralytics (máx. \|Δ\| = 0.000005);
  - si falta una clase en una réplica, se excluye por igual en todos los modelos (en **I**, 0 de 1 000 réplicas).
- **Consistencia:** signo de D en cada semilla (descriptivo).
- **Exploratorios** (IC 95 %, sin valor p): AP de D40, mAP@0.5:0.95, mAP sin D10, maestro frente a alumnos, FP16 y P/R.

### Diapositiva 3.8 — Protocolo de eficiencia

- Batch 1 y 640 × 640, con imágenes de prueba precargadas.
- **3 repeticiones** × (50 de calentamiento + 500 medidas) por modelo.
- Se separan el preprocesado, la inferencia y el NMS. La latencia principal es la **media de la inferencia**, y
  FPS = 1 000 / latencia media.
- **Problema encontrado:** con batch 1 el driver bajaba el reloj de la GPU por ahorro de energía (de 2 730 a 1 080
  MHz) y la latencia se duplicaba de forma intermitente (TRT FP32: de 1.56 a 3.3 ms).
- **Solución:**
  - reloj SM fijado en **2 505 MHz** (`nvidia-smi -lgc`, script `scripts/benchmark.ps1`);
  - cada repetición se valida (mediana del reloj = 2 505 ± 50 MHz; si no, se repite);
  - antes de cada modelo se comprueba que el sistema no tenga otra carga (GPU y CPU).
- **Resultado:** las 99 repeticiones de GPU mantuvieron el reloj, sin reintentos. La variación entre repeticiones fue
  de una mediana de 0.07 % en GPU (máx. 2 %) y de 4 % en CPU (máx. 12 %).

### Diapositiva 3.9 — Cómputo y trazabilidad

- **77 h de GPU** hasta el primer avance (`results/tables/compute_budget.csv`):

| Bloque | I (h) | J (h) |
|---|---|---|
| Maestros | 13.9 (11m 6.2, 11l 6.1, versión obsoleta 1.6) | 5.6 (solo 11m) |
| Alumnos base (3 semillas) | 8.2 (2.7 por corrida) | 11.3 (3.8 por corrida) |
| Piloto de λ (3 × 50 épocas) | 3.1 | 4.5 |
| Alumnos KD (3 semillas) | 18.9 (6.3 por corrida) | pendiente (~27; ~8.9 por corrida) |
| QAT (ajuste fino) | 6.4 | pendiente |
| Exportaciones, evaluación, eficiencia, exploraciones y pruebas | 5.2 | — |
| **Total** | **55.7** | **21.3** |

- La KD cuesta ~2.3 veces más que el entrenamiento base (**I:** 18.9 h frente a 8.2 h para 3 semillas), porque cada
  paso también ejecuta al maestro.
- **J** cuesta ~1.4 veces más por corrida que **I**: tiene 1.36 veces más imágenes.
- **Trazabilidad:**
  - `results/LOG.md` con fecha y resultado de cada fase;
  - `results/tables/protocol_deviations.csv`, con 12 desviaciones registradas, cada una indicando si fue antes o
    después de ver el conjunto de prueba;
  - repositorio público (AGPL-3.0) con código, configuración y listas de partición.

### Preguntas probables (tema 3)

- **¿Por qué la media y no la mediana para la latencia?** Por coherencia con FPS = 1 000 / media. En CPU la
  distribución tiene cola: la mediana de INT8 es ~8.0 ms y la media, 8.98 ms.
- **¿Fijar el reloj no es "hacer trampa"?** No. Mide todos los modelos en las mismas condiciones; sin fijarlo, la
  comparación dependía del ahorro de energía del driver, no del modelo.
- **¿Por qué bootstrap y no una prueba t?** Con 3 semillas no hay grados de libertad suficientes. El bootstrap
  sobre las imágenes de prueba captura la variabilidad del conjunto de evaluación y se combina con las semillas.

---

## Tema 4. Análisis de resultados preliminares (I)

**Objetivo del segmento:** presentar los resultados con las tablas del paper y responder P1, P2 y P3 para **I**.
**J** todavía no tiene alumnos KD ni resultados en prueba (J4c–J7), así que no se incluye; su avance en validación está
en las diapositivas 2.3 y 2.5.

### Diapositiva 4.1 — Desempeño en FP32 (Tabla V)

| Modelo | mAP@0.5 | mAP@0.5:0.95 | AP D40 | Prec. | ECE |
|---|---|---|---|---|---|
| Maestro | 0.329 | 0.141 | 0.413 | 0.382 | 0.097 |
| Base | 0.310 ± 0.020 | 0.129 ± 0.007 | 0.378 ± 0.020 | 0.375 ± 0.046 | 0.088 ± 0.043 |
| Destilado | 0.316 ± 0.005 | 0.129 ± 0.002 | 0.401 ± 0.012 | 0.373 ± 0.022 | 0.098 ± 0.044 |

- La KD apenas cambia la mAP@0.5 (+0.006), pero **reduce cuatro veces la variabilidad entre semillas** (d.e. de 0.005
  frente a 0.020).
- Mejora el AP de los baches: +0.023.
- No mejora la calibración: ECE de 0.098 frente a 0.088.
- El maestro supera a los alumnos por poco: +0.019 frente al base y +0.013 frente al destilado. Tiene poco que
  enseñar.
- **AP por clase (TRT FP32):**

| | D00 | D10 | D20 | D40 |
|---|---|---|---|---|
| Base | 0.274 | 0.014 | 0.574 | 0.378 |
| Destilado | 0.289 | 0.003 | 0.571 | 0.401 |

  D20 (piel de cocodrilo) es la clase más fácil. D10 es casi cero: solo hay 9 instancias.

### Diapositiva 4.2 — Pérdida por cuantización (Tabla VI)

| Modelo | Método | ΔmAP@0.5 | ECE |
|---|---|---|---|
| Base | PTQ FP16 | 0.003 ± 0.003 | 0.086 ± 0.042 |
| Base | PTQ INT8 | 0.035 ± 0.009 | 0.096 ± 0.056 |
| Base | PTQ-MO INT8 | 0.006 ± 0.009 | 0.110 ± 0.047 |
| Base | QAT INT8 | 0.006 ± 0.003 | 0.077 ± 0.020 |
| Destilado | PTQ FP16 | 0.000 ± 0.001 | 0.098 ± 0.042 |
| Destilado | PTQ INT8 | 0.031 ± 0.007 | 0.113 ± 0.065 |
| Destilado | PTQ-MO INT8 | 0.013 ± 0.003 | 0.112 ± 0.062 |
| Destilado | QAT INT8 | 0.011 ± 0.009 | 0.085 ± 0.032 |

- **FP16 no pierde nada.**
- **La PTQ INT8 de TensorRT pierde ~3 puntos** (11 % en el base y 10 % en el destilado). En baches, el AP de D40 cae
  de 0.378 a 0.323 en el base y de 0.401 a 0.354 en el destilado.
- **Hallazgo clave:** el control PTQ-MO, *sin ajuste fino*, ya recupera casi todo. La QAT apenas añade algo. La
  mejora viene del **esquema de cuantización** (Q/DQ explícito de ModelOpt), no del reentrenamiento. Ese esquema
  cuantiza casi las mismas convoluciones (81 frente a 79 de 88) pero deja más cómputo en FP16: ~41 % de las capas de
  cómputo en INT8, frente a ~65 % en la PTQ implícita. Es una observación; el mecanismo no se aisló.
- **OpenVINO INT8 en CPU** casi no pierde: 0.003 en el base y −0.002 en el destilado.
- **Calibración:** la PTQ de TensorRT empeora el ECE, y la QAT obtiene el ECE más bajo de todos (incluso menor que en
  FP32).

### Diapositiva 4.2b — Diagrama de fiabilidad

- Figura: `results/figures/presentacion/fiabilidad.png`, con la confianza media de cada intervalo frente a la fracción de detecciones
  correctas, para FP32, PTQ INT8 y QAT INT8 (3 semillas juntas).
- **Qué señalar:**
  - En FP32 los puntos quedan **por encima** de la diagonal: los alumnos son **subconfiados** (aciertan más de lo que
    su confianza indica). Es lo contrario de la sobreconfianza que Guo *et al.* describen en clasificación, y conviene
    mencionarlo al volver a la diapositiva 1.6.
  - La QAT acerca las curvas a la diagonal; por eso tiene el ECE más bajo.
  - Las curvas de base y destilado casi se superponen: la KD no cambia la calibración (P2).

### Diapositiva 4.3 — Figura: mAP según el método de cuantización

- Figura: `results/figures/map_vs_cuantizacion.png`, con la media ± d.e. y cada semilla como un círculo.
- **Qué señalar:**
  - el "valle" de PTQ INT8 en GPU;
  - la recuperación con PTQ-MO y QAT;
  - las barras de error mucho más cortas del destilado;
  - que OpenVINO INT8 queda al nivel de FP32.

### Diapositiva 4.4 — Contrastes de hipótesis (Tabla VII)

| Hip. | D [IC] | p (Holm) | Semillas a favor de KD | Decisión |
|---|---|---|---|---|
| H1a (ΔmAP, PTQ) | +0.004 [−0.007, +0.015] | 1.000 | 2/3 | No rechaza H0 |
| H1b (ΔmAP, QAT) | −0.005 [−0.020, +0.015] | 1.000 | 0/3 | No rechaza H0 |
| H2a (ECE, PTQ) | −0.016 [−0.031, +0.013] | 1.000 | 1/3 | No rechaza H0 |
| H2b (ECE, QAT) | −0.008 [−0.027, +0.014] | 1.000 | 1/3 | No rechaza H0 |
| H3 (KD + PTQ − base + QAT) | −0.019 [−0.033, −0.009] (IC 90 %) | 1.000 | 1/3 supera −δ | No se demuestra la no inferioridad |

- Los p crudos son 0.471, 0.563, 0.249, 0.425 y 0.847.
- **P1:** la KD no reduce la pérdida, ni con PTQ ni con QAT.
- **P2:** la KD no conserva mejor la calibración.
- **P3:** el alumno destilado con PTQ **no** alcanza al base con QAT. Además, el IC completo de H3 es negativo: base +
  QAT es mejor.
- Los exploratorios dicen lo mismo: sin D10 (H3 sin D10 = −0.021 [−0.036, −0.005]), con mAP@0.5:0.95, en baches y con
  OpenVINO, todos los IC de las diferencias entre alumnos incluyen el cero.

### Diapositiva 4.5 — Eficiencia (Tabla VIII)

Figura de apoyo: `results/figures/presentacion/map_vs_latencia.png` (mAP@0.5 frente a latencia; arriba y a la izquierda es mejor).
Resume en una imagen el compromiso: PTQ INT8 es lo más rápido pero pierde ~3 puntos; PTQ-MO y QAT recuperan la
precisión por ~0.1 ms más.

| Plataforma | Precisión | Tamaño (MB) | Latencia (ms) | FPS |
|---|---|---|---|---|
| RTX 4060 | FP32 | 12.9 | 1.70 ± 0.01 | 589 |
| RTX 4060 | FP16 | 7.5 | 0.87 ± 0.01 | 1 146 |
| RTX 4060 | INT8 (PTQ) | 4.5 | 0.71 ± 0.01 | 1 408 |
| RTX 4060 | INT8 (PTQ-MO) | 4.5 | 0.83 ± 0.02 | 1 202 |
| RTX 4060 | INT8 (QAT) | 4.5 | 0.82 ± 0.01 | 1 214 |
| i7-12700KF | FP32 | 10.2 | 15.55 ± 0.15 | 64 |
| i7-12700KF | INT8 (PTQ) | 3.2 | 8.98 ± 0.15 | 111 |

- El INT8 es **2.4 veces** más rápido en GPU y **1.7 veces** en CPU, y ocupa un tercio.
- Lo que cuesta recuperar la precisión: el Q/DQ explícito (PTQ-MO y QAT) es ~15 % más lento que la PTQ implícita (0.1
  ms en términos absolutos).
- La KD **no cuesta nada** en inferencia: misma arquitectura.
- Preprocesado: ~0.86 ms (GPU) y 1.2 ms (CPU). NMS: 0.3 y 0.5 ms.

### Diapositiva 4.6 — Hallazgo lateral: el maestro colapsa con la PTQ de TensorRT

- El maestro YOLO11m pierde **0.178 de mAP@0.5** con la PTQ INT8 de TensorRT: de 0.329 a 0.151. Es unas 6 veces más
  que los alumnos.
- Por clase, D20 se derrumba (de 0.585 a 0.073). La precisión cae a 0.06 y ninguna detección alcanza la confianza de
  0.25, así que el ECE queda indefinido.
- **Causa:** valores extremos en las activaciones de la cabeza: máximo de 474 frente a un percentil 99.99 de 12.8.
- Figura: `results/figures/presentacion/activaciones_maestro.png`. El cociente máx/p99.99 de cada convolución es ≤ 2.4 en todo el alumno,
  mientras que en el maestro llega a 37 en la cabeza de detección (zona sombreada). MinMax
  fija la escala con el máximo, así que los valores típicos quedan con 3–4 niveles.
- Con **OpenVINO** el mismo maestro no pierde nada (de 0.331 a 0.330).
- **Lección:** la sensibilidad a la cuantización depende de la distribución de activaciones y del calibrador, no
  solo del tamaño del modelo.

### Preguntas probables (tema 4)

- **¿Por qué la QAT del destilado pierde más que la del base (0.011 frente a 0.006)?** La diferencia no es
  significativa (H1b, p crudo 0.563), y la QAT, con lr 0.001 y sin mosaico, no mejora a todas las semillas por igual.
- **¿Que el ECE de la QAT sea menor que el de FP32 es raro?** No necesariamente. El ajuste fino sin mosaico
  reajusta las confianzas a la distribución real de las imágenes. La diferencia (~0.01) está dentro de la variación
  entre semillas.

---

## Tema 5. Discusión, conclusiones preliminares y siguientes pasos

**Objetivo del segmento:** interpretar los resultados, reconocer las limitaciones y mostrar qué falta, sobre todo
la réplica en Japón.

### Diapositiva 5.1 — Interpretación

- **La KD no protege frente a la cuantización** en **I** (P1 y P2).
  - Coincide con Karjol y Hanna (BDD100K) y con SHE-YOLO (degradación comparable a la del modelo base).
  - Aquí se sostiene con 3 semillas, estadística preespecificada y QAT.
- **Lo que sí aporta la KD:** estabilidad. El resultado de un solo entrenamiento es más predecible (d.e. 4 veces menor).
- **P3:** el factor decisivo para conservar la precisión en INT8 es el **esquema de cuantización** (Q/DQ explícito de
  ModelOpt, con o sin QAT), no la destilación.
- **Posible explicación:** el maestro de **I** apenas supera a los alumnos (+0.019 en prueba y +0.003 en validación),
  así que tiene poco que enseñar (Cho y Hariharan: un maestro más preciso no garantiza un mejor alumno).
- **J como contraste:** su maestro supera a los alumnos por +0.018 en validación, con alumnos muy estables (d.e.
  0.001). Si la explicación anterior es correcta, en **J** la KD debería notarse más. Es la pregunta abierta de la
  réplica.
- **Tamaño frente a sensibilidad:** el modelo grande fue el más sensible (por las activaciones extremas), lo que
  contradice la intuición de que los modelos pequeños son más frágiles.

### Diapositiva 5.2 — Implicancia práctica

- Para desplegar un detector ligero en INT8:
  1. priorizar la cuantización explícita Q/DQ (ModelOpt, con o sin QAT), que recupera ~2.5 puntos de mAP@0.5 con
     ~0.1 ms más de latencia;
  2. en CPU, OpenVINO INT8 casi no pierde precisión;
  3. verificar siempre la PTQ de cada modelo, porque puede fallar de forma silenciosa (el caso del maestro);
  4. la KD sirve para estabilizar el entrenamiento, no para proteger de la cuantización (al menos con un maestro poco
     superior).

### Diapositiva 5.3 — Limitaciones

1. **El maestro es poco superior** en **I**: +2.5 puntos de mAP@0.5 sin D10 en validación. El YOLO11l no lo mejoró.
   En **J** la ventaja es clara (+1.9 sin D10, con d.e. de 0.002).
2. **La variación entre semillas** (~2 puntos) supera el margen δ, así que H3 tiene poca potencia. Pero su IC quedó por
   completo bajo cero.
3. **La PTQ usa la configuración por defecto** del exportador (cabeza en mayor precisión), y MinMax es sensible a los
   valores extremos.
4. **La QAT es sin destilación y solo en GPU.**
5. **D10 tiene 9 instancias** de prueba en **I**. En **J** tiene 591, así que esta limitación no se repite.
6. **El bootstrap sobre imágenes** no captura toda la variabilidad entre entrenamientos; por eso se usan 3 semillas y se
   reporta el signo por semilla.
7. **Orden de ejecución:** la PTQ se evaluó en prueba antes de la QAT (con la QAT ya configurada), y la estadística
   se implementó después de ver los resultados descriptivos, sin cambiar nada de lo especificado. Está registrado en
   las desviaciones del protocolo.
8. **Una sola familia de modelos** y un solo método de destilación.

### Diapositiva 5.4 — Avance de I y J

| Paso | I | J |
|---|---|---|
| Datos y auditoría | ✅ | ✅ |
| Maestro YOLO11m | ✅ | ✅ |
| Alumnos base (3 semillas) | ✅ | ✅ |
| Verificación de la KD y piloto de λ | ✅ (λ = 1.0) | ✅ (λ = 1.0) |
| Alumnos KD (3 semillas) | ✅ | Pendiente (~27 h de GPU) |
| PTQ, QAT y evaluación en prueba | ✅ | Pendiente |
| Estadística (H1a–H3) | ✅ | Pendiente |
| Eficiencia | ✅ | Pendiente (solo como verificación) |

- **J** va hasta el piloto de λ. Sus resultados en validación ya se integraron en las diapositivas 1.11, 1.12, 2.3,
  2.5 y 3.3b.
- El conjunto de prueba de **J** no se ha mirado: se usará por primera vez en su evaluación, igual que en **I**.
- **Por qué J es interesante:** permite probar si la KD ayuda **cuando el maestro sí tiene algo que enseñar**.

### Diapositiva 5.5 — Siguientes pasos

1. Completar **J**: alumnos KD, PTQ, QAT, evaluación en prueba, estadística y eficiencia.
2. **Sensibilidad a la calibración (exploratorio):** recalibrar dos alumnos con otros 2 conjuntos de 500 imágenes
   (semillas 43 y 44) para ver cuánto varía la pérdida INT8.
3. Completar las tablas del paper con las columnas de **J** y comparar **I** y **J**: ¿se replican los efectos?

### Diapositiva 5.6 — Conclusión preliminar (cierre)

> En **I**, la destilación por canales no redujo la pérdida por cuantización INT8 ni mejoró la calibración, y un
> alumno destilado con PTQ no alcanzó a un alumno base con QAT. La destilación estabiliza el entrenamiento sin costo
> en inferencia, pero no sustituye a un mejor esquema de cuantización. La réplica en **J** dirá si esto cambia cuando
> el maestro es claramente superior.

### Preguntas probables (tema 5)

- **¿Entonces la destilación no sirve?** Sirve para otra cosa: reduce la variabilidad y mejora un poco los baches.
  Lo que no hace, en **I**, es proteger frente a la cuantización.
- **¿Y si el resultado de J es distinto?** Sería un hallazgo: indicaría que el efecto depende de la ventaja del
  maestro o del dataset. El análisis de **J** está fijado de antemano, igual que en **I**.

---

## Material disponible para las diapositivas

| Recurso | Ruta | Tema |
|---|---|---|
| Diagrama PRISMA | `paper/figures/prisma_flujo.pdf` | 1 |
| Partición I y J (Tabla III) | `results/tables/paper/tabla_III_particion.md` | 1 |
| Auditoría de datos | I: `results/tables/data_audit.csv`; J: `results/japan/tables/data_audit.csv` | 1 |
| Maestros 11m frente a 11l | `results/figures/teachers_m_vs_l.png` | 2 |
| Convergencia del maestro | `results/figures/teacher_convergence.png` | 2 o 3 |
| Piloto λ (I y J) | `results/figures/pilot_lambda.png` y `results/japan/figures/pilot_lambda.png` | 2 |
| Capas por precisión | `results/tables/engine_layers.csv` | 3 |
| Presupuesto de cómputo | `results/tables/compute_budget.csv` | 3 |
| Desviaciones del protocolo | `results/tables/protocol_deviations.csv` | 3 |
| Tablas V–VIII | `results/tables/paper/` | 4 |
| mAP según el método de cuantización | `results/figures/map_vs_cuantizacion.png` | 4 |
| Exploratorios | `results/tables/exploratory.csv` | 4 |
| Diagrama de flujo | `results/figures/presentacion/diagrama_flujo.png` | 2 |
| Distribución de clases | `results/figures/presentacion/distribucion_clases.png` | 1 |
| Diagrama de fiabilidad | `results/figures/presentacion/fiabilidad.png` | 4 |
| mAP frente a latencia | `results/figures/presentacion/map_vs_latencia.png` | 4 |
| Activaciones del maestro y del alumno | `results/figures/presentacion/activaciones_maestro.png` (datos: `results/tables/activaciones_maestro_alumno.csv`) | 4 |
