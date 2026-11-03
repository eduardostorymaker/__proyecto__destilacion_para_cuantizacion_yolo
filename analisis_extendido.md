# Análisis extendido: hallazgos y discusión (India)

Fecha: 2026-11-02. Solo India; lo de Japón queda pendiente (J4c–J7).
Fuentes: `results/tables/hypothesis_tests.csv`, `results/tables/exploratory.csv`, `results/metrics.csv`,
`results/tables/efficiency.csv`, `results/tables/activaciones_maestro_alumno.csv`.
Marcado **(nuevo)**: lo que todavía no está en la Discusión de `paper/main.tex`.
Convención: D > 0 favorece al alumno destilado (KD). mAP = mAP@0.5 salvo que se indique otra cosa.

---

## Hallazgos

Ordenados por fuerza de la evidencia.

### 1. La destilación no redujo la pérdida por cuantización (P3)

- Con la PTQ INT8 de TensorRT, el alumno base perdió 0.035 de mAP@0.5 y el destilado 0.031
  (H3a = +0.004, IC 95 % [−0.007, +0.015], p Holm 1.000; 2/3 semillas a favor).
- Con QAT el signo se invierte: H3b = −0.005 (IC 95 % [−0.020, +0.015]; 0/3 semillas a favor).
- **(nuevo)** No es solo que falte significancia: el IC acota el efecto. Con 95 % de confianza, la KD reduce la
  pérdida de la PTQ en como máximo 0.015, menos de la mitad de los 0.035 que pierde el base. Se descarta que la
  destilación "elimine" la pérdida, pero no que la reduzca un poco.

### 2. La ventaja del destilado tras la PTQ viene de FP32, no de resistir mejor (P1, P2)

- Con PTQ, el destilado supera al base en +0.010 (H2a, IC 95 % [−0.007, +0.019]), en 3/3 semillas, pero con un
  IC que incluye el 0.
- Esa ventaja se descompone en +0.006 que ya tenía en FP32 (H1) más +0.004 de menor pérdida (H3a), según la
  identidad H2 = H1 + H3.
- Con QAT la ventaja desaparece (H2b = +0.001; 1/3 semillas).

### 3. (nuevo) El signo de H3 depende del esquema de cuantización, no de la destilación

| Esquema | D = ΔmAP base − ΔmAP KD | IC 95 % |
|---|---|---|
| PTQ implícita TensorRT (H3a) | +0.004 | [−0.007, +0.015] |
| OpenVINO INT8 (E18) | +0.005 | [−0.002, +0.020] |
| PTQ-MO, ModelOpt (E17) | −0.007 | [−0.024, +0.009] |
| QAT (H3b) | −0.005 | [−0.020, +0.015] |

- Todos los IC incluyen el 0.
- Con los esquemas explícitos (PTQ-MO y QAT), ambos alumnos convergen a ~0.304 de mAP@0.5 (base 0.304 / KD 0.303
  con PTQ-MO; 0.304 / 0.305 con QAT). La ventaja del destilado en FP32 se pierde y por eso su ΔmAP es mayor.
- Lectura prudente: con un buen esquema de cuantización, el nivel final parece fijarlo la arquitectura y no el
  entrenamiento previo. Es una hipótesis para Japón, no una conclusión.

### 4. La destilación no mejoró la calibración tras cuantizar (P4)

- El ECE del destilado fue peor que el del base: 0.113 frente a 0.096 con PTQ (H4a = −0.016, p crudo 0.249), y
  0.085 frente a 0.077 con QAT (H4b = −0.008, p crudo 0.425). Ninguna diferencia es significativa.
- Descontando la diferencia que ya había en FP32, el deterioro del ECE al cuantizar fue igual en ambos
  (H4c = −0.007, H4d = +0.001; IC con el 0).
- **(nuevo)** El diagrama de fiabilidad muestra que en FP32 los alumnos están subconfiados. La QAT dio el ECE más
  bajo de todos, incluso menor que en FP32 (0.077 y 0.085 frente a 0.088 y 0.098). Probablemente se debe al ajuste
  fino de 30 épocas sin mosaico, que recalibra las confianzas. No es un efecto de la cuantización en sí.

### 5. Un alumno KD con PTQ no alcanza a un alumno base con QAT (P5)

- H5 = −0.019, IC 90 % [−0.033, −0.009], p crudo 0.847 (p Holm 1.000).
- El intervalo queda entero bajo cero, así que el base con QAT es superior. El límite superior (−0.009) apenas pasa
  −δ (−0.010), por lo que tampoco se puede afirmar que el KD con PTQ sea peor por más de 1 punto.
- Sin D10 se mantiene: −0.021 [−0.036, −0.005] (E07).

### 6. Lo que decide la pérdida es el esquema de cuantización

ΔmAP@0.5 sobre los mismos pesos (base / KD):

| Esquema | ΔmAP@0.5 |
|---|---|
| FP16 (TensorRT) | 0.003 / 0.000 |
| OpenVINO INT8 (NNCF) | 0.003 / −0.001 |
| PTQ-MO (ModelOpt, sin ajuste fino) | 0.006 / 0.013 |
| QAT | 0.006 / 0.011 |
| PTQ implícita de TensorRT | 0.035 / 0.031 |

- La mayor parte de la recuperación la da PTQ-MO, sin ajuste fino. La QAT añade poco sobre ese control.
- **(nuevo)** Que OpenVINO INT8 no pierda nada con los mismos modelos confirma que la pérdida de ~3 puntos viene de
  la cadena de herramientas (calibrador MinMax implícito + selección de capas de TensorRT), no de un límite de INT8
  para YOLO11n.

### 7. (nuevo) El efecto más robusto de la destilación es la estabilidad entre semillas, y se mantiene al cuantizar

| Formato | d.e. de mAP@0.5, base | d.e. de mAP@0.5, KD |
|---|---|---|
| FP32 | 0.020 | 0.005 |
| PTQ INT8 | 0.012 | 0.004 |
| PTQ-MO | 0.026 | 0.005 |
| QAT | 0.020 | 0.004 |

La reducción de la variabilidad es de 3 a 5 veces en todos los formatos. Es descriptivo, con 3 semillas, pero
consistente en todos los formatos.

### 8. (nuevo, parcial) Baches (D40): el destilado conserva su ventaja en AP en todos los formatos, sin significancia

| Formato | AP@0.5 D40, base | AP@0.5 D40, KD |
|---|---|---|
| FP32 | 0.378 | 0.401 |
| PTQ INT8 | 0.323 | 0.354 |
| OpenVINO INT8 | 0.379 | 0.403 |
| QAT | 0.372 | 0.381 |

La diferencia en la pérdida de D40 tiene un IC que incluye el 0 (E01 = +0.009 [−0.009, +0.025]).

### 9. El maestro colapsa con la PTQ de TensorRT por valores extremos en sus activaciones, no por su tamaño

- Pierde 0.178 de mAP@0.5 (0.14 más que los alumnos, IC 95 % [0.10, 0.17], E08–E09), y ninguna detección llega a
  conf 0.25 (ECE no definido).
- La causa está en la cabeza del maestro: máximo 474 frente a un p99.99 de 12.8 (cociente hasta 37). En el alumno,
  ese cociente es ≤ 2.4.
- Con FP16 (0.001) y con OpenVINO (0.331 frente a 0.330) no pierde nada.

### 10. Eficiencia: costo y beneficio por esquema

| Plataforma y esquema | Aceleración vs. FP32 | Latencia | Tamaño | Pérdida |
|---|---|---|---|---|
| GPU, PTQ implícita | 2.4× | 0.71 ms | 4.5 MB | ~3 pts |
| GPU, ModelOpt / QAT | 2.0–2.1× | 0.82–0.83 ms | 4.5 MB | ≤ 1.3 pts |
| CPU, OpenVINO INT8 | 1.7× | 8.98 ms | 3.2 MB | ~0 |

- Los esquemas explícitos son ~15 % más lentos que la PTQ implícita (+0.12 ms).
- **(nuevo)** El mejor compromiso en GPU es ModelOpt: cambia 0.12 ms por ~2.5 puntos de mAP. En CPU, OpenVINO INT8
  da la aceleración sin costo de exactitud.
- **(nuevo)** El maestro acelera 4.5× con INT8 frente a 2.4× de los alumnos. En un modelo más grande el cómputo pesa
  más que los costos fijos de cada inferencia.

---

## Discusión

**Respuesta a la pregunta general.** En India, la destilación CWD no hace al YOLO11n más robusto a la cuantización
INT8. Lo deja algo mejor después de la PTQ solo porque ya era algo mejor antes. Es decir, la KD "mejora la
cuantización" por nivel, pero no por robustez, y aun esa mejora de nivel no es concluyente. La distinción entre
nivel (P2) y robustez (P3), con H2 = H1 + H3, es la contribución metodológica que permite decirlo con precisión. La
literatura suele mezclar ambas cosas.

**Comparación con antecedentes.**
- Karjol y Hanna (2026), en BDD100K, encontraron pérdidas casi iguales con y sin KD (0.030 frente a 0.031). Aquí la
  magnitud es similar (0.031 frente a 0.035), en otro dominio.
- SHE-YOLO también reportó una degradación comparable a la de su modelo base.
- Lo que este estudio agrega: 3 semillas, bootstrap pareado con hipótesis preespecificadas y corrección de Holm, y
  la QAT. La QAT tampoco revela ventaja del destilado y, de hecho, invierte el signo.
- **(nuevo)** Karjol y Hanna interpretan que el destilado "conserva mayor precisión en INT8". Nuestros datos
  sugieren que esa lectura confunde nivel con robustez: el destilado conserva su ventaja de FP32, no resiste mejor.

**Por qué la destilación no ayudaría a cuantizar** (posibles explicaciones, no probadas).
- La hipótesis de partida era que imitar los mapas de activación del maestro suavizaría las distribuciones del
  alumno. En la práctica, el alumno base ya tiene activaciones sin valores extremos (cociente ≤ 2.4), así que hay
  poco que corregir.
- El maestro es justamente el modelo con activaciones extremas. Imitar sus distribuciones por canal (CWD normaliza
  con softmax espacial) no transfiere su escala, así que no las hereda el alumno, pero tampoco aporta suavidad.
- La ventaja del maestro es pequeña (0.019 de mAP@0.5 en prueba), lo que limita lo que el alumno puede aprender
  (Cho y Hariharan).

**El esquema de cuantización importa más que el entrenamiento.**
- El mismo alumno pierde 3 puntos con la PTQ implícita de TensorRT y casi nada con OpenVINO, con ModelOpt sin ajuste
  fino o con QAT.
- Implicación práctica: antes de invertir en destilación o en QAT para un modelo ligero, conviene probar otro
  calibrador o esquema de PTQ, que es más barato. La QAT (~1 h por modelo) añadió poco sobre su control PTQ-MO
  (minutos).
- ModelOpt deja más cómputo en FP16 (~41 % en INT8 frente a ~65 %). El resultado es coherente con que la pérdida
  venga de cuantizar más activaciones intermedias, pero no se aisló el mecanismo.

**Calibración.** Los alumnos están subconfiados, y la cuantización implícita empeora el ECE en ambos por igual. La
QAT lo mejora, probablemente por su ajuste fino y no por la cuantización. Para un sistema de inspección vial que use
un umbral de confianza, esto significa que el umbral elegido en FP32 no se transfiere tal cual a INT8: hay que
recalibrarlo con el modelo desplegado.

**Valor práctico de la destilación.** Aunque no ayude a cuantizar, la destilación:
- reduce la variabilidad entre entrenamientos de 3 a 5 veces, en todos los formatos;
- mejora algo el AP de los baches;
- no tiene costo en inferencia (los adaptadores se descartan).

En la práctica, un único entrenamiento con KD es más predecible. Es un beneficio de reproducibilidad, no de
robustez a la cuantización.

**Tamaño del modelo y sensibilidad.** El colapso del maestro contradice la idea de que los modelos pequeños son más
frágiles al cuantizar. La sensibilidad depende de la distribución de las activaciones y del calibrador (MinMax), no
del número de parámetros. Hay que verificar la PTQ de cada modelo antes de desplegarlo, y un fallo así se detecta
con un simple análisis de máximo frente a percentil.

**Qué falta: Japón (pendiente J4c–J7).** Japón pone a prueba dos explicaciones de India:
1. El maestro tiene más ventaja (+1.8 puntos en validación) y el dataset es más fácil (mAP ~0.55). Si la falta de
   efecto se debía a un maestro débil, Japón debería mostrar H1 y H2 más claras.
2. Los alumnos base de Japón ya son muy estables (d.e. 0.001 en validación), así que el beneficio de estabilidad de
   la KD probablemente no aparezca.

Hay que ver también:
- si el patrón del hallazgo 3 (signo de H3 según el esquema) se repite;
- si la PTQ implícita vuelve a perder ~3 puntos;
- si el maestro de Japón también colapsa (activaciones extremas).

**Limitaciones (sin cambios respecto al paper):**
- maestro con poca ventaja;
- d.e. del base mayor que δ en H5;
- PTQ con la configuración por defecto del exportador;
- QAT sin KD;
- D10 con 9 instancias;
- el bootstrap no captura la variabilidad entre entrenamientos;
- análisis implementado después de los resultados descriptivos;
- una sola familia de modelos y un solo método de destilación.

**Limitación nueva:** las hipótesis exploratorias se formularon después de ver test en India, así que India no puede
confirmarlas. Japón es su primera prueba real.

---

## Puntos abiertos para revisión

- **Hallazgo 3 (el signo de H3 según el esquema):** es el más interpretativo; decidir si entra al paper o solo a la
  presentación.
- **Crítica a Karjol y Hanna (confunden nivel con robustez):** es un poco fuerte; puede suavizarse.
- **Mecanismo de la CWD (por qué no transfiere la escala):** especulativo; quitarlo o dejarlo como "posible
  explicación".
