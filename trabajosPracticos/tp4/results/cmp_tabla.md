| # | Exp | Arquitectura | Regularización | Train | Val | Test | Params | Épocas | Mejor ép. | Tiempo |
|:---:|:---:|---|---|---:|---:|---:|---:|:---:|:---:|---:|
| 1 | 1 | LeNet-5 Clásica | — | 0.9187 | 0.8952 | 0.8926 | 44,426 | 20 | 17 | 56 s |
| 2 | 2 | LeNet-5 Optimizada | — | 0.9234 | 0.8977 | 0.8943 | 106,154 | 12 | 7 | 30 s |
| 3 | 7 | AlexNet angosta + Dropout | Dropout 0.5 | 0.9388 | 0.9110 | 0.9124 | 877,258 | 11 | 6 | 48 s |
| 4 | 8 | AlexNet angosta + Dropout + DataAug | Dropout 0.5 + Rot ±0.05 | 0.9314 | 0.9098 | 0.9099 | 877,258 | 20 | 17 | 78 s |
| 5 | 3 | AlexNet Adaptada | — | 0.9255 | 0.9025 | 0.9049 | 3,310,858 | 9 | 4 | 46 s |
| 6 | 4 | AlexNet + DataAug | Rot ±0.05 | 0.9221 | 0.9030 | 0.9021 | 3,310,858 | 12 | 7 | 66 s |
| 7 | 5 | WRN-28-10 | Dropout 0.3 en bloque | 0.9345 | 0.9157 | 0.9169 | 36,478,906 | 10 | 5 | 1326 s |
| 8 | 6 | WRN-28-10 + RandomErasing | Dropout 0.3 en bloque + Random Erasing | 0.9623 | 0.9267 | 0.9305 | 36,478,906 | 19 | 14 | 2519 s |
