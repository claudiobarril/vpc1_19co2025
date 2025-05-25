# TP2 - CLASE 3

## Objetivo

Implementar un detector de máximo enfoque sobre un video aplicando técnicas de análisis espectral similar al que utilizan las cámaras digitales modernas. El video a procesar será: `focus_video.mov`.

### 1. Algoritmo de métrica de enfoque

Se debe implementar un algoritmo que, dada una imagen o región, calcule la métrica propuesta en el paper _"Image Sharpness Measure for Blurred Images in Frequency Domain"_ y realizar tres experimentos:

1. **Medición sobre todo el frame.**  
2. **Medición sobre una ROI** ubicada en el centro del frame.  
   - Área de la ROI: 5% o 10% del área total del frame.  
3. **Opcional**: Medición sobre una matriz de enfoque compuesta por un arreglo de NxM elementos rectangulares equiespaciados.  
   - N y M son valores arbitrarios; probar con varios valores (3×3, 7×5, etc.) — al menos 3 configuraciones.

#### Presentación de resultados

- Una curva (o varias curvas) que muestre la evolución de la métrica frame a frame, donde se vea claramente cuándo el algoritmo detectó el punto de máximo enfoque.

### 2. Cambio de métrica de enfoque

Cambiar la métrica de enfoque eligiendo uno de los algoritmos explicados en el apéndice de _Analysis of focus measure operators in shape-from-focus_.

---

El algoritmo de detección a implementar debe detectar y devolver los puntos de máximo enfoque de manera automática.
