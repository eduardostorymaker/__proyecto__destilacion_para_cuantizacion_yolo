# Análisis breve: hallazgos y discusión (India)

Versión resumida de `analisis_extendido.md` (2026-11-02), para el paper y la presentación. Solo India; Japón
pendiente. D > 0 favorece al alumno destilado (KD); mAP = mAP@0.5.

---

## 1. Hallazgos clave

1. **La KD no reduce la pérdida por cuantización (P3).** PTQ: 0.035 base frente a 0.031 KD (H3a = +0.004,
   IC 95 % [−0.007, +0.015]); QAT: −0.005. Con 95 % de confianza, la KD evita como máximo 0.015 de los 0.035 que
   se pierden.
2. **La ventaja del KD tras la PTQ viene de FP32 (P1, P2).** +0.010 con PTQ (3/3 semillas, IC con el 0)
   = +0.006 en FP32 + 0.004 de menor pérdida (H2 = H1 + H3). Con QAT desaparece (+0.001).
3. **La KD no mejora la calibración (P4).** ECE con PTQ: 0.113 KD frente a 0.096 base; el deterioro al cuantizar es
   igual en ambos. La QAT da el mejor ECE (0.077 / 0.085), por su ajuste fino.
4. **KD + PTQ no alcanza a base + QAT (P5).** H5 = −0.019, IC 90 % [−0.033, −0.009]: base + QAT es superior.
5. **El esquema de cuantización decide la pérdida.** Mismos pesos: PTQ implícita de TensorRT ~3 puntos; ModelOpt
   sin ajuste fino ≤ 1.3; QAT ≤ 1.1; OpenVINO INT8 y FP16 ≈ 0.
6. **La KD estabiliza el entrenamiento.** d.e. entre semillas 3–5 veces menor en todos los formatos
   (FP32: 0.005 frente a 0.020).
7. **El maestro colapsa con PTQ implícita (−0.178)** por activaciones extremas en su cabeza (máx. 474 frente a
   p99.99 de 12.8); con FP16 y OpenVINO no pierde nada. La sensibilidad depende de las activaciones y del
   calibrador, no del tamaño.
8. **Eficiencia.** INT8 acelera 2.4× en GPU y 1.7× en CPU; ModelOpt/QAT cuestan +15 % de latencia (+0.12 ms) a
   cambio de ~2.5 puntos de mAP.

Ninguna hipótesis confirmatoria rechaza H0 tras Holm (todos los p ajustados = 1.000).

---

## 2. Texto para el paper (Discusión)

> En India, la destilación no hizo al alumno más robusto a la cuantización INT8: su pérdida fue similar a la del
> alumno base con PTQ (0.031 frente a 0.035) y algo mayor con QAT, y su intervalo descarta que la destilación evite
> más de 0.015 de los 0.035 que pierde el base. La ventaja del alumno destilado tras la PTQ (0.010) se explica casi
> por completo por la que ya tenía en FP32 (0.006) más una pérdida apenas menor (0.004); es decir, la destilación
> mejora el nivel del modelo cuantizado, no su robustez, y ni siquiera esa mejora es concluyente. Este resultado
> coincide con Karjol y Hanna en BDD100K, que observaron pérdidas casi iguales con y sin destilación (0.030 frente a
> 0.031), y lo extiende con tres semillas, contrastes preespecificados y QAT.
>
> El factor decisivo fue el esquema de cuantización: con los mismos pesos, la PTQ implícita de TensorRT perdió unos
> 3 puntos de mAP@0.5, mientras que ModelOpt sin ajuste fino, la QAT y OpenVINO casi no perdieron. Por ello, para un
> detector ligero conviene probar otro calibrador o esquema de PTQ antes que invertir en destilación o en QAT. La
> sensibilidad tampoco dependió del tamaño: el maestro colapsó con la PTQ implícita por valores extremos en sus
> activaciones, que la calibración MinMax no tolera, y no perdió precisión con OpenVINO.
>
> El beneficio más claro de la destilación fue la estabilidad: redujo de 3 a 5 veces la variación entre semillas en
> todos los formatos y mejoró algo el AP de los baches, sin costo en inferencia. No mejoró la calibración; la QAT sí,
> probablemente por su ajuste fino, lo que indica que el umbral de confianza debe ajustarse sobre el modelo
> desplegado.

---

## 3. Contenido para la presentación (tema 5)

**Diapositiva 1. Hallazgos**
- La KD no reduce la pérdida INT8 (0.031 frente a 0.035)
- Su ventaja tras la PTQ viene de FP32: H2 = H1 + H3 (+0.010 = +0.006 + 0.004)
- No calibra mejor; base + QAT supera a KD + PTQ
- 0 de 5 confirmatorias rechazadas (Holm)

**Diapositiva 2. Discusión**
- El esquema de cuantización pesa más que la KD: PTQ implícita ~3 pts; ModelOpt, QAT y OpenVINO ≈ 0
- La sensibilidad no depende del tamaño: el maestro colapsa por activaciones extremas
- Coincide con Karjol y Hanna (0.030 frente a 0.031), ahora con 3 semillas, estadística preespecificada y QAT

**Diapositiva 3. Conclusiones y recomendación**
- KD: más estable (d.e. ÷ 3–5) y algo mejor en baches, sin costo en inferencia; no sustituye a una buena cuantización
- Desplegar en GPU con ModelOpt (+0.12 ms, −2.5 pts de pérdida); en CPU, OpenVINO INT8 (1.7×, sin pérdida)
- Verificar la PTQ de cada modelo y reajustar el umbral de confianza en INT8

**Diapositiva 4. Siguientes pasos (Japón)**
- J4c alumnos KD → J5 PTQ → J5b QAT → J6 evaluación → J7 estadística
- Pregunta: con un maestro más fuerte (+1.8 pts) y un dataset más fácil, ¿aparece el efecto de la KD?
- Replicar: pérdida de la PTQ implícita y colapso del maestro
