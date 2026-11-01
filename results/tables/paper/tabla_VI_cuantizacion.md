| Modelo | Método | ΔmAP (IN) | ECE (IN) | ΔmAP (JP) | ECE (JP) |
|---|---|---|---|---|---|
| Maestro | PTQ FP16 | 0.001 | 0.097 | — | — |
| Maestro | PTQ INT8 | 0.178 | — | — | — |
| Base | PTQ FP16 | 0.003 ± 0.003 | 0.086 ± 0.042 | — | — |
| Base | PTQ INT8 | 0.035 ± 0.009 | 0.096 ± 0.056 | — | — |
| Base | PTQ-MO INT8 | 0.006 ± 0.009 | 0.110 ± 0.047 | — | — |
| Base | QAT INT8 | 0.006 ± 0.003 | 0.077 ± 0.020 | — | — |
| Destilado | PTQ FP16 | 0.000 ± 0.001 | 0.098 ± 0.042 | — | — |
| Destilado | PTQ INT8 | 0.031 ± 0.007 | 0.113 ± 0.065 | — | — |
| Destilado | PTQ-MO INT8 | 0.013 ± 0.003 | 0.112 ± 0.062 | — | — |
| Destilado | QAT INT8 | 0.011 ± 0.009 | 0.085 ± 0.032 | — | — |

ΔmAP en mAP@0.5 respecto al TensorRT FP32 del mismo modelo; ECE del modelo cuantizado. Media ± d.e. de 3 semillas. PTQ-MO: PTQ con ModelOpt (control sin ajuste fino de la QAT). ECE en blanco: ninguna detección con confianza ≥ 0.25 (ECE no definido).
