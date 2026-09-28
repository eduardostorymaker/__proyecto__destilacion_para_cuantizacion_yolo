# Proyecto: Destilación de conocimiento vs. pérdida por cuantización en YOLO (detección de baches)

## Contexto de investigación (leer antes de cualquier tarea)

Estoy escribiendo un paper (formato IEEE, máx. 8 páginas) con esta pregunta:

> ¿La destilación de conocimiento (KD) reduce la pérdida de precisión que produce la
> cuantización INT8 en detectores YOLO ligeros para la detección de baches y daños viales?

**Hipótesis:** un YOLO11n entrenado con KD pierde menos mAP al cuantizarse a INT8 que el
mismo YOLO11n entrenado sin KD.

**Brecha (confirmada con revisión PRISMA):** en daños viales, la KD y la cuantización se han
estudiado por separado. SBP-YOLO (Liang et al., 2026) usa ambas pero evalúa la KD solo en FP32.
SHE-YOLO (Li et al., 2026) reporta que su modelo mejorado pierde al cuantizarse lo mismo que el
modelo base. Karjol y Hanna (2026, arXiv:2604.26857) muestran en peatones que los alumnos
destilados degradan menos bajo INT8 (−5.6 % vs −23 % mAP) por mejor calibración.

**Objetivo general:** evaluar el efecto de la KD en la pérdida de precisión por cuantización de
detectores YOLO ligeros, en el subconjunto India de RDD2022.

**Objetivos específicos (el código cubre del 2 al 4):**
1. Revisión sistemática PRISMA 2020 (ya hecha).
2. Entrenar YOLO11n con y sin KD (maestro YOLO11m) y comparar en FP32.
3. Cuantizar ambos a FP16 e INT8 y medir ΔmAP global y por clase (énfasis en D40, bache).
4. Medir eficiencia (tamaño, latencia, FPS) en RTX 4060 (TensorRT) y CPU i7-12700KF (OpenVINO).

## Hardware y entorno

- GPU: NVIDIA RTX 4060 8 GB · CPU: Intel i7-12700KF · RAM: 32 GB
- SO: WSL2 (Ubuntu 26.04) sobre Windows 11, 24 GB de RAM asignados a WSL
- Python 3.13 (venv `~/.venvs/yolokd`, creado con `scripts/setup_wsl.sh`), PyTorch con CUDA 13.0,
  `ultralytics` (versiones exactas en requirements.txt), TensorRT 10.16, OpenVINO, onnx, pandas,
  numpy, scipy, matplotlib, pycocotools.
- TensorRT se mantiene en la rama 10 (no 11): conserva la PTQ INT8 clásica (IInt8EntropyCalibrator2,
  todas las capas en INT8). TensorRT 11 obliga a Q/DQ con ModelOpt y deja capas en float.
  Por esto Python ≤ 3.13 (TensorRT 10 no tiene versión para 3.14).

## Diseño experimental (NO cambiar sin preguntarme)

| Elemento | Decisión |
|---|---|
| Dataset | `data/raw/RDD2022/India` (RDD2022, github.com/sekilab/RoadDamageDetector), solo carpeta `train` (el `test` oficial no tiene etiquetas) |
| Clases | D00, D10, D20, D40. Eliminar cualquier otra etiqueta y registrar cuántas se eliminaron |
| Partición | 70 % train / 15 % val / 15 % test, semilla 42, listas de archivos guardadas en `data/splits/` |
| Maestro | YOLO11m, preentrenado en COCO, 1 entrenamiento |
| Alumnos | YOLO11n base y YOLO11n + KD, preentrenados en COCO, 3 semillas cada uno (0, 1, 2) |
| KD | CWD (Shu et al., ICCV 2021) en las salidas del neck P3, P4, P5; pérdida = L_det + λ·L_CWD |
| Hiperparámetros | Idénticos para ambos alumnos (imgsz 640, mismas épocas, batch, optimizador, aumentación). La ÚNICA diferencia es la pérdida CWD |
| Cuantización | PTQ. GPU: TensorRT FP32/FP16/INT8. CPU: OpenVINO FP32/INT8 |
| Calibración INT8 | 500 imágenes del split **train** (mismas para ambos alumnos). **Nunca** usar val ni test |
| Evaluación | Siempre sobre el mismo split test |
| Métricas | mAP@0.5, mAP@0.5:0.95, AP por clase (D40), recall; ΔmAP = mAP_FP32 − mAP_INT8 |
| Estadística | media ± d.e. de 3 semillas; IC 95 % por bootstrap (1 000 remuestreos de imágenes de test) para la diferencia de ΔmAP entre alumnos |
| Eficiencia | tamaño (MB), latencia batch 1 (50 iteraciones de calentamiento + 500 medidas), FPS |

## Pipeline técnico (fases)

Trabajar **una fase a la vez**. Al terminar cada fase, resumir qué se hizo y esperar mi confirmación.

### Fase 0 — Entorno y estructura
- Crear estructura de carpetas (ver abajo), `requirements.txt` con versiones fijas y un script
  `scripts/check_env.py` que verifique CUDA, versión de TensorRT, OpenVINO y ultralytics.

### Fase 1 — Preparación de datos (`src/data/`)
- De `data/raw/RDD2022/India/train`, convertir XML (Pascal VOC) a formato YOLO txt.
- Filtrar clases, mantener imágenes sin daños como fondo (etiqueta vacía).
- Partición 70/15/15 con semilla 42; guardar listas en `data/splits/`.
- Generar `configs/data.yaml` y `configs/calib.yaml` (500 imágenes de train).
- Script de estadísticas: imágenes e instancias por clase y por split → `results/tables/dataset_stats.csv`.

### Fase 2 — Maestro (`src/train/train_teacher.py`)
- Entrenar YOLO11m. Guardar pesos en `runs/teacher/`.

### Fase 3 — Alumno base (`src/train/train_student.py --kd off --seed N`)
- YOLO11n, 3 semillas.

### Fase 4 — Alumno con KD (`src/train/train_student.py --kd on --seed N`)
- Implementar CWD como un `DetectionTrainer` personalizado de ultralytics (o adaptar una
  implementación abierta, documentando la fuente y el commit).
- Registrar hooks en el neck del maestro (congelado, `eval()`) y del alumno (P3–P5).
- Si los canales no coinciden (n vs m), usar convoluciones 1×1 de adaptación en el alumno;
  se entrenan junto con él y se descartan al exportar.
- τ y λ en `configs/kd.yaml`.
- **Antes de entrenar en serio:** prueba de humo de 2 épocas con `fraction` pequeño que confirme
  que L_CWD baja y que el modelo exportado no contiene los adaptadores.

### Fase 5 — Exportación y cuantización (`src/quant/export_all.py`)
- Para cada alumno y semilla: TensorRT (FP32, FP16, INT8 con `calib.yaml`) y OpenVINO (FP32, INT8).
- Verificar que la calibración INT8 usa `calib.yaml` y no el split val.
- Guardar en `runs/exports/{modelo}_{seed}_{formato}_{precision}/`.

### Fase 6 — Evaluación (`src/eval/evaluate.py`)
- Evaluar cada modelo exportado en el split test: métricas globales, por clase y recall.
- Guardar las predicciones por imagen (JSON) para el bootstrap.
- Resultados en `results/metrics.csv` (una fila por modelo/semilla/formato/precisión).

### Fase 7 — Estadística (`src/eval/bootstrap.py`)
- Bootstrap de imágenes de test (1 000 remuestreos, semilla fija) para ΔmAP de cada alumno y
  su diferencia. Salida: `results/tables/bootstrap.csv`.

### Fase 8 — Eficiencia (`src/bench/benchmark.py`)
- Latencia y FPS con batch 1 en GPU y CPU; tamaño en MB. Salida: `results/tables/efficiency.csv`.

### Fase 9 — Tablas y figuras para el paper (`src/report/`)
- Generar las Tablas III, IV y V del paper (media ± d.e.) en CSV y Markdown.
- Figura: mAP vs. precisión numérica (FP32/FP16/INT8) para ambos alumnos.

## Estructura del proyecto

```
proyecto/
├── CLAUDE.md
├── requirements.txt
├── configs/        data.yaml, calib.yaml, train.yaml, kd.yaml
├── data/           raw/, yolo/, splits/
├── src/            data/, train/, quant/, eval/, bench/, report/
├── scripts/        check_env.py y comandos .sh para lanzar entrenamientos
├── runs/           teacher/, students/, exports/   (no subir a git)
├── results/        metrics.csv, tables/, figures/
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

## Estado actual

- [x] Revisión sistemática y borrador del paper (v1)
- [x] Fase 0 — Entorno (2026-09-27, `results/env_check.json`)
- [ ] Fase 1 — Datos
- [ ] Fase 2 — Maestro
- [ ] Fase 3 — Alumno base
- [ ] Fase 4 — Alumno KD
- [ ] Fase 5 — Exportación
- [ ] Fase 6 — Evaluación
- [ ] Fase 7 — Estadística
- [ ] Fase 8 — Eficiencia
- [ ] Fase 9 — Tablas y figuras
