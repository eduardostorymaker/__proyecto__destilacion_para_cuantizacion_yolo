# Presentación del primer avance: estructura en 5 temas

Destilación de conocimiento vs. pérdida por cuantización en YOLO11n (detección de daños viales, RDD2022).

## Estado en el primer avance (2026-10-31)

Etiqueta git `primer-avance`. Todo lo que se expone corresponde a este punto del proyecto:

| Bloque | Estado |
|---|---|
| India (Fases 0–9) | Completa: entrenamiento, PTQ, QAT, evaluación en test, estadística, eficiencia, tablas y figura |
| Japón (réplica) | Datos, maestro, alumnos base, prueba de humo de la KD y piloto de λ (λ = 1.0); faltan alumnos KD, cuantización, evaluación y estadística |
| Cómputo | 77 h de GPU |
| Material para diapositivas | Tablas en `results/tables/paper/` y figura `results/figures/map_vs_cuantizacion.png` (también en el README, Fase 9) |

## 1. Introducción, estado del arte y dataset

- **Problema:** detectar daños viales en dispositivos de borde exige modelos pequeños y rápidos. La cuantización
  (FP32 → INT8) acelera el modelo, pero pierde precisión.
- **Conceptos base:** cuantización (PTQ y QAT) y destilación de conocimiento (maestro → alumno).
- **Estado del arte:** YOLO en daños viales, cuantización y KD. Antecedente más cercano: Karjol y Hanna (2026,
  arXiv:2604.26857): en BDD100K la KD no redujo la pérdida de mAP; una sola corrida y solo PTQ.
- **Brecha:** en daños viales, nadie compara un detector destilado con uno no destilado tras cuantizar, ni evalúa QAT.
- **Preguntas de investigación:**
  - P1: ¿la KD reduce la pérdida de mAP con INT8?
  - P2: ¿conserva mejor la calibración (ECE)?
  - P3: ¿KD + PTQ alcanza a base + QAT?
- **Dataset RDD2022:**
  - India, 4 clases: D00 grieta longitudinal, D10 grieta transversal, D20 piel de cocodrilo, D40 bache.
  - Japón como réplica (segundo dataset).
  - Desafíos: 58 % de imágenes de fondo en India y solo 9 instancias de D10 en test.

## 2. Flujo de la metodología planteada

Vista conceptual: **qué** se hace y en qué orden (conviene un diagrama de flujo).

1. Preparación de datos: limpieza, auditoría y partición 70/15/15 estratificada.
2. Entrenamiento del maestro (YOLO11m) y de los alumnos base (YOLO11n).
3. Destilación CWD: el alumno imita las características P3–P5 del maestro, con λ elegido en un piloto (λ = 1.0 en India y en Japón).
4. Cuantización de cada alumno con tres rutas: FP16, INT8 PTQ e INT8 QAT, más el control ModelOpt PTQ.
5. Evaluación de cada combinación: precisión (mAP), calibración (ECE) y eficiencia (latencia).
6. Comparación base frente a KD con contrastes estadísticos preespecificados (hipótesis H1–H3).

## 3. Metodología de experimentación computacional

Vista práctica: **cómo** se ejecutó para que sea válido y reproducible.

- **Hardware y software:** RTX 4060, i7-12700KF, WSL2, PyTorch, Ultralytics 8.4.164, TensorRT 10.16, OpenVINO y
  ModelOpt.
- **Entrenamiento:** hiperparámetros fijos (300 épocas, SGD, AMP), entrenamiento determinista y 3 semillas por alumno.
- **Cuantización:**
  - 500 imágenes de calibración tomadas solo de train.
  - TensorRT en GPU y OpenVINO en CPU.
  - Cada engine registrado con su hash SHA-256.
- **Evaluación:** test nunca mirado antes de tiempo; mismo preprocesado para todos los modelos; ΔmAP calculado dentro
  del mismo motor; ECE con 15 intervalos.
- **Estadística preespecificada:**
  - Bootstrap pareado con B = 1000.
  - Corrección de Holm para la familia de 5 contrastes.
  - Contraste de no inferioridad con δ = 1 punto de mAP.
- **Eficiencia:** sistema sin otra carga, reloj de la GPU fijado (2505 MHz), 3 repeticiones × 500 medidas.
- **Cómputo y trazabilidad:** 77 h de GPU hasta el primer avance; registro de desviaciones del protocolo y LOG.

## 4. Análisis de resultados preliminares (India)

| Bloque | Qué mostrar |
|---|---|
| **Entrenamiento** | Base 0.310 ± 0.020 frente a KD 0.316 ± 0.005 de mAP@0.5 en test. La KD reduce la variabilidad entre semillas. El maestro supera poco a los alumnos. |
| **P1: pérdida por INT8** | PTQ: 0.035 base frente a 0.031 KD. QAT: 0.006 frente a 0.011. FP16 y OpenVINO INT8 prácticamente no pierden. |
| **P2: calibración** | ECE sin ventaja de la KD (H2a D = −0.016, H2b D = −0.008). |
| **P3: KD + PTQ frente a base + QAT** | Base + QAT es mejor: D = −0.019, IC90 [−0.033, −0.009]. |
| **Estadística** | Ninguna hipótesis significativa tras Holm. |
| **Eficiencia** | INT8 es 2.4× más rápido en GPU (0.71 frente a 1.70 ms) y 1.7× en CPU (8.98 frente a 15.55 ms), con un tercio del tamaño (4.5 frente a 12.9 MB). La QAT es ~15 % más lenta que la PTQ. La KD no cuesta nada en inferencia. |
| **Hallazgo lateral** | El maestro colapsa con la PTQ de TensorRT (pierde 0.178 de mAP@0.5) por valores atípicos en las activaciones de la cabeza; con OpenVINO no pierde nada. |

## 5. Discusión, conclusiones preliminares y siguientes pasos

- **Interpretación:**
  - En India la KD no protege contra la cuantización, lo que coincide con Karjol y Hanna.
  - La QAT es la vía más eficaz para recuperar mAP, a cambio de algo de latencia.
  - Posible explicación: el maestro de India apenas supera a los alumnos, así que tiene poco que enseñar.
- **Limitaciones:** pocas instancias de D10, un solo dataset completo hasta ahora y una sola GPU.
- **Siguientes pasos:**
  - Réplica en Japón, en curso (siguiente: alumnos KD, ~27 h de GPU). Allí el maestro sí supera claramente a los
    alumnos (+1.8 puntos, con muy poco ruido entre semillas), lo que permite probar si la KD ayuda cuando el maestro
    es mejor.
  - Sensibilidad a la calibración (semillas de calibración 43 y 44).
  - Completar las tablas con Japón y alinear el paper con los resultados del repositorio.

## Nota para los expositores

Los temas 2 y 3 se diferencian en que el 2 explica el diseño y el 3 su ejecución rigurosa. Si un mismo expositor no
tiene ambos, conviene que el 2 cierre con el diagrama completo y que el 3 lo retome punto por punto.
