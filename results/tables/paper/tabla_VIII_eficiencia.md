| Plataforma | Precisión | Tamaño (MB) | Latencia (ms) | FPS |
|---|---|---|---|---|
| RTX 4060 | FP32 | 12.9 | 1.70 ± 0.01 | 589 |
| RTX 4060 | FP16 | 7.5 | 0.87 ± 0.01 | 1146 |
| RTX 4060 | PTQ INT8 | 4.5 | 0.71 ± 0.01 | 1408 |
| RTX 4060 | PTQ-MO INT8 | 4.5 | 0.83 ± 0.02 | 1202 |
| RTX 4060 | QAT INT8 | 4.5 | 0.82 ± 0.01 | 1214 |
| i7-12700KF | FP32 | 10.2 | 15.55 ± 0.15 | 64 |
| i7-12700KF | PTQ INT8 | 3.2 | 8.98 ± 0.15 | 111 |

Batch 1, 640 × 640, latencia de inferencia (media ± d.e. entre los 6 alumnos, cada uno con 3 repeticiones de 500 medidas); FPS = 1000 / latencia media; reloj SM de la GPU fijado en 2505 MHz.
