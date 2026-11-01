# Proyecto: Destilación de conocimiento vs. pérdida por cuantización en YOLO (detección de baches)

## Contexto de investigación (leer antes de cualquier tarea)

Paper IEEE (máx. 10 págs. de contenido, sin contar los apéndices), en LaTeX: `paper/main.tex`. La fuente de
verdad del diseño y de los resultados es ESTE repositorio (configs, LOG, CSV); el paper se escribió por separado y
debe ajustarse al repo, no al revés. No editar el paper: el usuario lo corrige (lista de correcciones pendientes
en la memoria del proyecto, hasta que el usuario avise).
Tres preguntas:

- **P1.** ¿La destilación de conocimiento (KD) reduce la pérdida de mAP por cuantización INT8 en
  un YOLO11n para daños viales?
- **P2.** ¿El alumno KD conserva mejor calibración (ECE) tras cuantizar?
- **P3.** ¿Un alumno KD con PTQ alcanza el desempeño de un alumno base con QAT?

**Brecha:** en detección de daños viales basada en imágenes, nadie compara un detector destilado
vs. no destilado tras cuantizar, ni evalúa QAT. Antecedente más cercano fuera del dominio:
Karjol y Hanna (2026, arXiv:2604.26857, preprint no revisado): en BDD100K, KD NO redujo la
pérdida de mAP (0.030 vs 0.031) pero sí conservó mayor precisión en INT8; una sola corrida y
solo PTQ.

Nota: la búsqueda de antecedentes en Scopus se reporta con PRISMA 2020 (apéndices del paper). PRISMA se queda
(decisión del usuario, 2026-10-31).

## Hardware y entorno

- GPU: NVIDIA RTX 4060 8 GB · CPU: Intel i7-12700KF · RAM: 32 GB
- SO: WSL2 (Ubuntu 26.04) sobre Windows 11, 24 GB de RAM asignados a WSL
- Python 3.13 (venv `~/.venvs/yolokd`, creado con `scripts/setup_wsl.sh`), PyTorch con CUDA 13.0,
  `ultralytics` (versiones exactas en requirements.txt), TensorRT 10.16, OpenVINO, onnx, pandas,
  numpy, scipy, matplotlib, pycocotools.
- TensorRT se mantiene en la rama 10 (no 11) para conservar la PTQ implícita con el calibrador
  del exportador de Ultralytics (ver D4). TensorRT 11 elimina la calibración implícita y obliga a
  Q/DQ explícito con ModelOpt. Por esto Python ≤ 3.13 (TensorRT 10 no tiene versión para 3.14).
- ultralytics fijado en 8.4.164: si cambia la versión, puede cambiar el calibrador INT8.

## Diseño experimental (NO cambiar sin preguntarme)

### D1. Datos
- Fuente: `data/raw/RDD2022/India/train` (el test oficial no tiene etiquetas). No descargar nada.
- Clases: D00=0, D10=1, D20=2, D40=3. (Se revirtió la exclusión de D10: si se borran sus
  etiquetas pero se conservan las imágenes, las grietas transversales se vuelven "fondo"
  (etiquetas faltantes), lo que puede dañar el aprendizaje de D00. Sus pocas instancias se
  tratan en el análisis (sensibilidad sin D10), no en los datos.)
  Con la semilla 42, D10 queda con 9 instancias en test (aceptado 2026-09-28; no se cambia la
  semilla). La mAP de 4 clases es la métrica principal; la mAP sin D10 es sensibilidad.
- Reglas de limpieza (aplicar en este orden y registrar cada conteo):
  1. Etiquetas de otras clases: se eliminan SOLO esas cajas; la imagen se conserva.
  2. Cajas fuera de la imagen: recortar a los bordes.
  3. Cajas con ancho o alto < 1 px tras recortar: eliminar.
  4. Cajas duplicadas idénticas (misma clase y coordenadas): dejar una.
  5. Imágenes corruptas o ilegibles: eliminar la imagen.
  6. Imágenes sin cajas tras la limpieza: se CONSERVAN como fondo (.txt vacío). No submuestrear el fondo.
  NUNCA eliminar una imagen por otro motivo que no sea el punto 5.
- Auditoría obligatoria antes de seguir → `results/tables/data_audit.csv` y resumen en LOG.md:
  a) imágenes totales en raw y en data/yolo (iguales salvo corruptas);
  b) cajas por etiqueta ORIGINAL (todas las del XML, incluidas las no estándar) y cuántas se
     eliminaron por cada regla;
  c) número y porcentaje de imágenes de fondo;
  d) explicación de qué fueron las "filas" eliminadas en la versión anterior de la Fase 1 y por qué.
- Partición: 70/15/15 con estratificación iterativa multietiqueta (Sechidis et al., 2011; paquete
  `iterative-stratification`) sobre la presencia por imagen de las 4 clases + etiqueta "fondo".
  Semilla 42. Listas en `data/splits/`.
- data/yolo en el sistema de archivos de Linux (WSL), no en /mnt, por velocidad de lectura
  (actualmente `~/datasets/rdd2022_india_yolo`, configurable en `configs/data_prep.yaml`).
- Estadísticas: `results/tables/dataset_stats.csv` (imágenes e instancias por clase y split).
  Si D10 queda con < 10 instancias en test, avisar.

### D2. Entrenamiento (común a maestro y alumnos)
- Pesos iniciales COCO: yolo11m.pt y yolo11l.pt (maestros candidatos), yolo11n.pt (alumnos).
- imgsz=640, batch=8, nbs=64, epochs=300.
- optimizer='SGD' (explícito, NO 'auto'), lr0=0.01, lrf=0.01, momentum=0.937,
  weight_decay=0.0005, warmup_epochs=3, cos_lr=False.
- Aumentación por defecto de Ultralytics: mosaic=1.0, close_mosaic=10, hsv_h=0.015, hsv_s=0.7,
  hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, mixup=0.0. Sin aumentaciones extra.
- Precisión de entrenamiento: amp=True (precisión mixta) para TODOS los modelos (maestro, alumnos
  y QAT); confirmado tras la Fase 1b (2026-09-28): FP32 satura la VRAM y es ~7× más lento. AMP no es cuantización: los pesos maestros quedan en FP32 y
  best.pt es el punto de partida de toda la cuantización.
- deterministic=True; semillas fijadas en Python, NumPy y PyTorch.
- Maestros: patience=50, seed=0, 1 corrida cada uno. Dos candidatos (decisión 2026-10-01, porque el
  YOLO11m apenas supera a los alumnos base): YOLO11m (`teacher_s0`) y YOLO11l (`teacher_l_s0`), misma
  config D2. Regla de selección fijada antes del resultado del 11l (`configs/kd.yaml`,
  seleccion_maestro): revalidar best.pt de ambos en VALIDACIÓN (imgsz 640, batch 8, conf 0.001,
  iou 0.7); se usa YOLO11l solo si su mAP@0.5:0.95 supera a la del 11m en ≥ 0.010. Ambos se conservan.
- Alumnos (base y KD): patience=300 (sin parada temprana), seeds 0, 1, 2; se usa best.pt
  (fitness de Ultralytics sobre validación).
- workers=8. cache='disk' para todas las corridas ('ram' es no determinista en Ultralytics con
  deterministic=True).
- Todo en `configs/train.yaml`; guardar args.yaml de cada corrida.

### D3. Destilación (solo alumno KD)
- CWD (Shu et al., ICCV 2021) sobre las 3 entradas de la cabeza Detect (P3, P4, P5).
- Maestro: best.pt del maestro elegido (D2), congelado, eval(), sin gradientes, con autocast; recibe el
  MISMO batch aumentado que el alumno.
- Adaptadores: conv 1×1 en el alumno para igualar canales con el maestro; se entrenan con el
  alumno y NO se exportan (verificar que no están en best.pt).
- Pérdida: L = L_det + λ·L_CWD, con τ=1.0.
- λ: piloto antes de las corridas finales: λ ∈ {0.5, 1.0, 2.0}, seed 0, 50 épocas, mismos
  hiperparámetros; elegir por mAP@0.5:0.95 de VALIDACIÓN (nunca test). Durante el piloto,
  registrar por época L_det y L_CWD para ver que L_CWD baja y que su escala es razonable frente
  a L_det. Documentar en LOG.md.
- `configs/kd.yaml`. Si se adapta código externo, registrar repositorio y commit.

### D4. Cuantización post-entrenamiento (PTQ)
- Punto de partida: best.pt en FP32 de cada modelo. FP16 e INT8 se generan POR SEPARADO desde ese
  modelo (no hay paso FP32 → FP16 → INT8).
- GPU (RTX 4060, TensorRT 10): FP32 (half=False), FP16 (half=True), INT8 (int8=True) con el
  exportador de Ultralytics 8.4.164 tal cual (decisión 2026-09-28): calibración implícita MinMax
  (`IInt8Calibrator`, MINMAX_CALIBRATION), sigmoide de la cabeza forzada a FP32 y convoluciones de
  salida de la cabeza y DFL forzadas a FP16; el flag FP16 está activo, así que TensorRT puede
  dejar otras capas en FP16. Motivo: representa el uso práctico y es coherente con la calibración
  por máximo de ModelOpt en la QAT.
- CPU (i7-12700KF, OpenVINO): FP32 e INT8 (NNCF).
- Exportación: imgsz=640, batch=1, dynamic=False, simplify=True.
- Calibración INT8: `data/splits/calib.txt` = 500 imágenes de TRAIN, semilla 42, estratificadas
  por clase. `configs/calib.yaml` con 'train' y 'val' apuntando a calib.txt, para que nunca se
  calibre con val o test. Mismas imágenes para todos los modelos y para la QAT.
- Engine inspector: por cada engine INT8, reportar número y porcentaje de capas en INT8, FP16 y
  FP32 en `results/tables/engine_layers.csv`.
- Cada engine TensorRT/OpenVINO se construye UNA sola vez y se registra con su hash SHA-256 en
  `results/tables/engines.csv` (con versión del driver NVIDIA, CUDA, TensorRT y OpenVINO). La evaluación y la
  eficiencia usan siempre esos archivos y verifican el hash antes (`src/quant/engines_registry.py`).
- Exploratorio (después de la Fase 6): sensibilidad a la calibración. Recalibrar base_s0 y kd_s0 con 2
  conjuntos de calibración adicionales (500 imágenes de train, estratificadas, mismas reglas, semillas 43 y 44)
  y reportar la variación del ΔmAP INT8 entre los 3 conjuntos (42, 43, 44).
- Se cuantizan: 6 alumnos (todas las precisiones) y el maestro (TensorRT FP16 e INT8, como
  referencia).

### D5. QAT (solo GPU, TensorRT INT8)
- Herramienta: NVIDIA TensorRT Model Optimizer (modelopt), cuantización explícita Q/DQ INT8.
- Para cada alumno (base y KD, seeds 0-2 = 6 ajustes), desde best.pt:
  1. Insertar cuantizadores y calibrar con calib.yaml.
  2. Evaluar ese modelo SIN ajuste fino ("ModelOpt PTQ") como control: así la ganancia de la QAT
     se mide contra el mismo esquema de cuantización.
  3. Ajuste fino: SOLO pérdida de detección (sin CWD) en ambos alumnos. Propuesta inicial:
     30 épocas (10 % del entrenamiento), lr0=0.001, lrf=0.01, warmup_epochs=0, mosaic=0 (como
     las últimas épocas del entrenamiento), resto igual a D2. Proponer cambios si la prueba de
     humo lo sugiere.
  4. Exportar ONNX con Q/DQ → TensorRT INT8.
- Confirmar qué calibrador usa ModelOpt por defecto para las activaciones y registrarlo
  (LOG.md y Métodos).
- `configs/qat.yaml`. Prueba de humo con 1 alumno y 1 época antes de las 6 corridas.
- Configuración CONGELADA el 2026-10-18 en `configs/qat.yaml`. Punto de control: best.pt por el fitness de
  Ultralytics en VALIDACIÓN (igual que los alumnos, D2). La prueba de humo se evalúa solo en validación; el
  test de la QAT no se mira hasta evaluarla con `src/eval/evaluate.py`.
- Calibración de la QAT con las 500 imágenes de calib.yaml, sin aumentación (no los 8 lotes aumentados que
  usa Ultralytics por defecto). Calibrador de ModelOpt: algoritmo "max", activaciones por tensor, pesos por canal.
- Si la integración se bloquea seriamente, avisar: la QAT es opcional y el paper puede seguir
  solo con PTQ.

### D6. Evaluación
- Siempre split test, imgsz=640, batch=1, conf=0.001, iou=0.7, save_json=True.
- Preprocesado (decisión 2026-10-17): todos los modelos, incluido el .pt FP32, se evalúan con letterbox
  cuadrado 640×640 (rect=False) y batch 1, igual que los engines estáticos. (La validación por defecto de
  Ultralytics usa rect=True y da 1–2 puntos más de mAP@0.5 por el preprocesado, no por la cuantización;
  hallazgo de la exploración PTQ, hecha en validación.)
- ΔmAP se calcula dentro del mismo motor: GPU = TRT FP32 − TRT {FP16, INT8, QAT INT8};
  CPU = OV FP32 − OV INT8.
- El .pt FP32 (rect=False) es control de consistencia: debe coincidir con TRT FP32
  (|Δ mAP@0.5| < 0.002); si no, avisar antes de seguir.
- ECE y bootstrap (D7) usan exactamente esas mismas predicciones.
- Métricas: mAP@0.5, mAP@0.5:0.95, AP por clase, precisión y recall (punto de máximo F1 de
  Ultralytics).
- ΔmAP = mAP_FP32 − mAP_cuantizado, para PTQ FP16, PTQ INT8 y QAT INT8, global y en D40, con
  mAP@0.5 y mAP@0.5:0.95.
- ECE (métrica principal): detecciones con conf ≥ 0.25, emparejadas con cajas reales por
  IoU ≥ 0.5 (emparejamiento greedy por confianza); correcta si coincide la clase; 15 intervalos
  de igual ancho.
- Sensibilidad: mAP recalculada sin D10 (3 clases).
- `results/metrics.csv`: una fila por modelo/semilla/formato/precisión.

### D7. Estadística (preespecificada)
Este análisis se fija ANTES de tener resultados. Los valores de alfa, delta, B y las hipótesis NO
se cambian después de ver los datos. Si durante la implementación algo obliga a cambiarlo,
preguntar primero.

Descriptivo:
- Media ± d.e. de las 3 semillas para todas las métricas y modelos.

Bootstrap pareado:
- B = 1000 remuestreos con reemplazo de las 1 166 imágenes de test, semilla 42.
- En cada réplica se usan los MISMOS índices de imágenes para TODOS los modelos (base, KD,
  maestro, todas las precisiones).
- Cada diferencia se calcula por semilla, emparejando base_s con kd_s (s = 0, 1, 2), y luego se
  promedia sobre las 3 semillas.
- mAP en cada réplica: evaluación tipo COCO sobre las imágenes remuestreadas (una imagen repetida
  cuenta tantas veces como aparezca). Usar una implementación rápida (p. ej., faster-coco-eval) y
  verificar que, sin remuestreo, reproduce la mAP de Ultralytics (diferencia < 0.001); si no
  coincide, avisar antes de seguir.
- Si una réplica no contiene instancias de una clase, esa clase se excluye de la mAP de esa
  réplica en TODOS los modelos por igual; registrar cuántas réplicas se vieron afectadas.
- Guardar la distribución bootstrap completa de cada estadístico en `results/bootstrap/*.npy`.

Definiciones:
- dmAP50 = mAP@0.5(FP32) − mAP@0.5(cuantizado), 4 clases, dentro del mismo motor (D6).
- ECE según D6 (conf ≥ 0.25, IoU ≥ 0.5, 15 intervalos).
- Convención de signo: D > 0 favorece al alumno KD en H1 y H2.

Familia confirmatoria (5 contrastes, alfa global = 0.05, corrección Holm):
- H1a: D = dmAP50_base − dmAP50_kd, PTQ INT8.   H0: D = 0 (bilateral)
- H1b: D = dmAP50_base − dmAP50_kd, QAT INT8.   H0: D = 0 (bilateral)
- H2a: D = ECE_base − ECE_kd, PTQ INT8.         H0: D = 0 (bilateral)
- H2b: D = ECE_base − ECE_kd, QAT INT8.         H0: D = 0 (bilateral)
- H3 : D = mAP50(kd + PTQ INT8) − mAP50(base + QAT INT8). No inferioridad con margen
  delta = 0.010 (1 punto de mAP@0.5). H0: D ≤ −delta ; H1: D > −delta (unilateral).
Si la QAT se descarta por problemas técnicos, la familia se reduce a H1a y H2a (Holm con 2
contrastes) y se reporta así en el paper.

Valor p (bootstrap centrado, Efron y Tibshirani, 1993):
- Bilateral:      p = (1 + #{ |D*_b − D_obs| ≥ |D_obs| }) / (B + 1)
- H3 unilateral:  p = (1 + #{ D*_b − D_obs ≥ D_obs + delta }) / (B + 1)
  donde D_obs es el estadístico en la muestra original y D*_b en la réplica b.
- Corrección de Holm sobre los valores p de la familia; reportar p crudo y p ajustado.

Intervalos:
- IC 95 % por percentiles (2.5 y 97.5) para H1a–H2b.
- Para H3: IC 90 % bilateral; se concluye no inferioridad si su límite inferior es mayor que
  −delta (equivale al contraste unilateral al 5 %).

Consistencia entre semillas:
- Para cada contraste, reportar el signo de D en cada semilla por separado (p. ej., "3/3 semillas
  positivas"). Es descriptivo, no un contraste.

Análisis exploratorios (IC 95 %, sin valor p ni afirmación de significancia): AP@0.5 de D40,
dmAP con mAP@0.5:0.95, mAP sin D10, dmAP del maestro frente a los alumnos, FP16 frente a FP32,
precisión y recall en INT8.

Salidas:
- `results/tables/hypothesis_tests.csv` con columnas: hipotesis, estadistico, D_obs, IC_inf,
  IC_sup, p_crudo, p_holm, decision (rechaza H0 / no rechaza H0), signos_por_semilla.
- `results/tables/exploratory.csv` con estadístico, D_obs, IC_inf, IC_sup.

La Fase 7 se implementa y se prueba con predicciones de modelos de prueba (p. ej., los de la
prueba de humo) antes de tener los modelos finales, para que el análisis esté listo y congelado
antes de ver los resultados reales.

### D8. Eficiencia
- Batch 1, imgsz 640, 50 calentamientos + 500 medidas; mediana y media.
- Separar preproceso, inferencia y postproceso (NMS); latencia principal = inferencia.
  FPS = 1000 / latencia media de inferencia.
- Tamaño en MB del archivo exportado. Registrar driver NVIDIA y CUDA.
- Sistema sin otra carga durante la medición (sin entrenamientos ni otras aplicaciones pesadas).
- 3 repeticiones completas de la medición por modelo (cada una con sus 50 calentamientos + 500 medidas);
  reportar la variación entre repeticiones.
- Registrar el estado de los relojes de la GPU (frecuencias de SM y memoria, estado de energía) antes y
  durante la medición.
- Usar exactamente los engines registrados en `results/tables/engines.csv` (verificar el hash).
- `results/tables/efficiency.csv`.
- Reloj SM de la GPU fijado en 2505 MHz durante la medición (`scripts/benchmark.ps1`, decisión 2026-10-22) y cada
  repetición de GPU validada (mediana del reloj SM en 2505 ± 50 MHz; si no, se repite). Ver protocol_deviations.csv.

### D9. Réplica en RDD2022-Japón (segundo dataset; decisión 2026-10-18)
- Objetivo: repetir el estudio completo (D1–D8) con el subconjunto Japón de RDD2022 para tener dos datasets de
  entrada y comparar si los efectos de la KD frente a la cuantización se repiten. Antes figuraba como
  "Japón diferido" (ver `results/tables/protocol_deviations.csv`).
- Mismo protocolo que India: mismas clases (D00, D10, D20, D40), reglas de limpieza, partición 70/15/15 con
  estratificación iterativa (semilla 42), 500 imágenes de calibración de train, mismos hiperparámetros (D2),
  misma CWD (D3), misma PTQ (D4), misma QAT congelada (D5), misma evaluación (D6), misma estadística (D7) y
  misma eficiencia (D8).
- Fuente: `data/raw/RDD2022/Japan/train` (el test oficial no tiene etiquetas). YOLO en
  `~/datasets/rdd2022_japan_yolo`. Todo separado de India: `configs/japan/`, `data/splits/japan/`,
  `runs/japan/`, `results/japan/` (mismos nombres de archivo que en India).
- El test de Japón no se mira hasta su Fase J6, igual que en India.
- Decisiones CONFIRMADAS por el usuario el 2026-10-18, antes de cualquier resultado de Japón:
  - Maestro: solo YOLO11m (en India el 11l no superó al 11m).
  - λ: se repite el piloto en Japón (λ ∈ {0.5, 1.0, 2.0}, seed 0, 50 épocas, elegir por mAP@0.5:0.95 de val).
  - Estadística: análisis D7 completo por separado en cada dataset (cada uno con su familia de 5 contrastes y su
    corrección de Holm); la coincidencia entre datasets se reporta de forma descriptiva.
- Eficiencia en Japón (decisión 2026-10-31, opción 2): el paper reporta la eficiencia una sola vez, con India
  (la latencia depende de la arquitectura, no de los datos). La J8 se ejecuta igual, al final del pipeline y sin
  prioridad, solo como VERIFICACIÓN: diferencia relativa de latencia de inferencia Japón vs. India por motor y
  precisión, para respaldar esa afirmación con un dato ("en Japón la latencia difirió menos de X %").
  Las tablas del paper no cambian.
- Ejecución: todos los scripts aceptan `--dataset japan` (por defecto `india`), que toma la configuración de
  `configs/japan/` en lugar de `configs/`.
- Diferencias de los datos (auditoría J1): Japón tiene 10 506 imágenes (1.36× India), 24.8 % de fondo (India
  58.2 %), D10 abundante (591 instancias en test frente a 9 en India) e imágenes de 3 tamaños (600×600,
  540×540, 1024×1024; India todas 720×720).
- Cómputo estimado (× 1.36 sobre India): maestro 8–14 h, alumnos base ~11 h, piloto λ ~4 h, alumnos KD ~26 h,
  QAT ~12–15 h, exportación ~2 h → **~65–75 h de GPU**.

## Pipeline (orden de ejecución)

Una fase a la vez; al terminar, resumen y esperar mi confirmación.
Los entrenamientos largos los lanzo yo: preparar scripts .sh.

- **Fase 1** (REHACER) Datos con 4 clases + reglas D1 + auditoría completa.
- **Fase 1b** Pruebas de tiempo, antes de cualquier entrenamiento largo:
  - Maestro YOLO11m, 3 épocas con amp=True y 3 con amp=False.
  - Alumno YOLO11n, 3 épocas con amp=True.
  - Reportar: tiempo por época (sin contar la primera), VRAM máxima, pérdidas (detectar NaN) y
    estimación total del proyecto: maestro 300 ép., 3 alumnos base × 300, 3 alumnos KD × 300
    (estimar el costo extra del maestro en KD), piloto λ 3 × 50, QAT 6 × 30.
  - Recomendar AMP o FP32 y esperar decisión.
- **Fase 2** Maestro (lo lanzo yo). Mientras entrena, avanzar Fase 4a y Fase 6.
- **Fase 4a** Implementar CWD + prueba de humo (2 épocas, fraction pequeño; puede usarse
  yolo11m.pt de COCO como maestro temporal solo para la prueba).
- **Fase 3** Alumnos base, seeds 0-2.
- **Fase 4b** Piloto de λ (con el maestro final) y luego alumnos KD, seeds 0-2.
- **Fase 5** PTQ (D4).
- **Fase 5b** QAT (D5).
- **Fase 6** Evaluación (D6).
- **Fase 7** Estadística (D7).
- **Fase 8** Eficiencia (D8).
- **Fase 9** Tablas III, IV y V del paper (media ± d.e.) en CSV y Markdown, y figura de mAP vs.
  método de cuantización para ambos alumnos.
- **Réplica Japón (D9)**, después de terminar India y con el mismo código y configuración:
  J1 datos + auditoría · J2 maestro YOLO11m · J3 alumnos base · J4 piloto λ + alumnos KD · J5 PTQ ·
  J5b QAT · J6 evaluación · J7 estadística · J8 eficiencia · J9 tablas comparando India y Japón.

## Estructura del proyecto

```
proyecto/
├── CLAUDE.md
├── requirements.txt
├── configs/        data_prep.yaml, data.yaml, calib.yaml, train.yaml, kd.yaml, qat.yaml; japan/ (réplica)
├── data/           raw/, splits/   (data/yolo vive en el disco de Linux, ver D1)
├── src/            data/, train/, quant/, eval/, bench/, report/, utils/
├── scripts/        check_env.py y comandos .sh para lanzar entrenamientos
├── runs/           teacher/, students/, exports/   (no subir a git)
├── results/        metrics.csv, LOG.md, tables/ (incl. data_audit.csv), figures/
└── paper/          borrador del paper y referencias
```

## Reglas de trabajo

- Responde y comenta el código en **español**.
- **Entrenamientos largos:** no los ejecutes tú dentro de la sesión. Prepara el comando o script
  `.sh` y dime cómo lanzarlo; yo lo corro en mi terminal y te comparto el resultado.
- Toda configuración va en `configs/*.yaml`; nada de valores fijos escondidos en el código.
- Fija semillas (Python, NumPy, PyTorch) y registra versiones de librerías en cada corrida.
- Nombres de corridas consistentes: `{rol}_{kd|base}_s{seed}` (ej. `student_kd_s1`).
- Antes de cualquier cambio al diseño experimental de arriba, pregúntame: afecta la validez del paper.
- Si un resultado parece raro (p. ej., INT8 mejor que FP32 por mucho), avísame en vez de seguir.
- Mantén un `results/LOG.md` con fecha, qué se corrió y resultado principal de cada fase.
- Registra cada desviación del protocolo en `results/tables/protocol_deviations.csv` (fecha, cambio, motivo,
  ¿antes o después de ver test?).
- Actualiza `results/tables/compute_budget.csv` (horas de GPU por fase) con `python -m src.report.compute_budget`.

## Estado actual

- [x] Revisión sistemática y borrador del paper (v3)
- [x] Fase 0 — Entorno (2026-09-27, `results/env_check.json`)
- [x] Fase 1 — Datos (2026-09-28, 4 clases, auditoría en `results/tables/data_audit.csv`)
- [x] Fase 1b — Pruebas de tiempo (2026-09-28, AMP; ~41 h de GPU estimadas)
- [x] Fase 2 — Maestro YOLO11m (2026-09-30; best.pt época 148, val mAP@0.5 0.344 / mAP@0.5:0.95 0.148)
- [x] Fase 2b — Maestro YOLO11l (2026-10-02; val mAP@0.5:0.95 0.1444 < 0.1571) → regla: se usa YOLO11m
- [x] Fase 4a — Implementación CWD (2026-09-30, `src/train/cwd.py`; humo y verificación OK)
- [x] Fase 3 — Alumnos base (2026-10-01; val mAP@0.5 0.3415 ± 0.0215; alerta: maestro apenas mejor)
- [x] Fase 4b — Piloto λ (λ = 1.0) + alumnos KD (2026-10-17; val mAP@0.5 0.3467 ± 0.0079; check_kd OK)
- [x] Fase 5 — PTQ (2026-10-18; 33 exportaciones TRT/OV; ~65 % de capas de cómputo en INT8)
- [x] Fase 5b — QAT (2026-10-19; 6 controles ModelOpt PTQ + 6 QAT, evaluados en test con `src/eval/evaluate.py`)
- [x] Fase 6 — Evaluación (2026-10-18, PTQ: `results/metrics.csv`; los modelos QAT se evaluarán con el mismo `src/eval/evaluate.py`)
- [x] Fase 7 — Estadística (2026-10-20, India: ninguna hipótesis rechaza H0 tras Holm; `results/tables/hypothesis_tests.csv`)
- [x] Fase 8 — Eficiencia (2026-10-22, India: `results/tables/efficiency.csv`; reloj GPU fijado en 2505 MHz con `scripts/benchmark.ps1`)
- [x] Fase 9 — Tablas y figuras (2026-10-31, India: `python -m src.report.paper_tables` → `results/tables/paper/`; columnas JP en J9)
- [x] J1 — Datos Japón (2026-10-18; auditoría en `results/japan/tables/data_audit.csv`)
- Réplica Japón (D9; decisiones confirmadas 2026-10-18). Todos los comandos con `--dataset japan`:
  - [x] J2 — Maestro YOLO11m (2026-10-20; best.pt época 81, val mAP@0.5 0.571 / mAP@0.5:0.95 0.277)
  - [x] J3 — Alumnos base s0–s2 (2026-10-21; val mAP@0.5 0.5529 ± 0.0012; maestro +1.8 puntos)
  - [x] J4a — Humo y verificación KD con el maestro de Japón (2026-10-31, TODO OK): `--kd on --seed 0 --humo --dataset japan` + `check_kd`
  - [x] J4b — Piloto λ (2026-10-31; λ = 1.0, val mAP@0.5:0.95 0.2500): `bash scripts/pilot_lambda.sh japan` (~4 h) → fijar λ en `configs/japan/kd.yaml`
  - [ ] J4c — Alumnos KD s0–s2: `for s in 0 1 2; do bash scripts/train_student.sh --kd on --seed $s --dataset japan; done` (~26 h)
  - [ ] J5 — PTQ: `bash scripts/export_all.sh --dataset japan` (~2 h)
  - [ ] J5b — QAT: `bash scripts/qat.sh --dataset japan` (~7–9 h)
  - [ ] J6 — Evaluación en test: `bash scripts/evaluate.sh --dataset japan`
  - [ ] J7 — Estadística D7 (familia y Holm propios de Japón)
  - [ ] J8 — Eficiencia D8, solo como verificación frente a India (al final, sin prioridad; ver D9): `powershell -ExecutionPolicy Bypass -File scripts\benchmark.ps1 --dataset japan` (PowerShell como administrador)
  - [ ] J9 — Tablas y figuras comparando India y Japón: volver a ejecutar `python -m src.report.paper_tables`
