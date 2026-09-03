| Exp ID | Arquitectura | Pooling | Tamaño Kernel | Dropout | Data Aug. | Train Acc | Val Acc | Test Acc | Param. Totales |
|:---:|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|
| 1 | LeNet-5 Clásica | Average | 5x5 | No | No | 0.9187 | 0.8952 | 0.8926 | 44,426 |
| 2 | LeNet-5 Optimizada | Max | 3x3 | No | No | 0.9234 | 0.8977 | 0.8943 | 106,154 |
| 3 | AlexNet Adaptada | Max | 3x3 | No | No | 0.9255 | 0.9025 | 0.9049 | 3,310,858 |
| 4 | AlexNet + DataAug | Max | 3x3 | No | Sí (Rot ±0.05) | 0.9221 | 0.9030 | 0.9021 | 3,310,858 |
| 5 | WRN-28-10 | GlobalAvg | 3x3 | Sí (0.3 en bloque) | No | 0.9345 | 0.9157 | 0.9169 | 36,478,906 |
| 6 | WRN-28-10 + RandomErasing ⭐ | GlobalAvg | 3x3 | Sí (0.3 en bloque) | Sí (Random Erasing) | 0.9623 | 0.9267 | 0.9305 | 36,478,906 |
| 7 | AlexNet angosta + Dropout | Max | 3x3 | Sí (0.5) | No | 0.9388 | 0.9110 | 0.9124 | 877,258 |
| 8 | AlexNet angosta + Dropout + DataAug | Max | 3x3 | Sí (0.5) | Sí (Rot ±0.05) | 0.9314 | 0.9098 | 0.9099 | 877,258 |
