# Visión por Computadora I

## Integrantes

- **Iñaki Larrumbide** (a1703)  
  ✉️ [ilarrumbide10@gmail.com](mailto:ilarrumbide10@gmail.com)

- **Claudio Barril** (a1708)  
  ✉️ [claudiobarril@gmail.com](mailto:claudiobarril@gmail.com)

- **Christian Pisani Testa** (a1715)  
  ✉️ [christian.tpg@gmail.com](mailto:christian.tpg@gmail.com)

## Contenido

El presente repositorio contiene la resolución de los trabajos prácticos de la materia, correspondientes a la cohorte 19 del año 2025.

## TP1

- Parte 1 (imágenes en /white_patch):
1. Implementar el algoritmo White Patch para librarnos de las diferencias de color de iluminación.
2. Mostrar los resultados obtenidos y analizar las posibles fallas (si es que las hay) en el caso de
White patch.
- Parte 2:
1. Para las imágenes img1_tp.png y img2_tp.png leerlas con OpenCV en escala de grises y
visualizarlas.
2. Elija el número de bins que crea conveniente y grafique su histograma, compare los histogramas
entre sí. Explicar lo que se observa, si tuviera que entrenar un modelo de clasificación/detección
de imágenes, ¿considera que puede ser de utilidad tomar como ‘features’ a los histogramas?

## TP2

Implementar un detector de máximo enfoque sobre un video aplicando técnicas de análisis espectral similar al que utilizan las
cámaras digitales modernas. El video a procesar será: “focus_video.mov”.
1. Se debe implementar un algoritmo que dada una imagen, o región, calcule la métrica propuesta en el paper "Image
Sharpness Measure for Blurred Images in Frequency Domain" y realizar tres experimentos:
   1. Medición sobre todo el frame.
   2. Medición sobre una ROI ubicada en el centro del frame. Área de la ROI = 5% o 10% del área total del frame.
   Opcional:
   3. Medición sobre una matriz de enfoque compuesta por un arreglo de NxM elementos rectangulares equiespaciados. N y M son valores
   arbitrarios, probar con varios valores 3x3, 7x5, etc. (al menos 3)

   Para cada experimento se debe presentar:
   - Una curva o varias curvas que muestren la evolución de la métrica frame a frame donde se vea claramente cuando el algoritmo detecto el punto
   de máximo enfoque.
2. Cambiar la métrica de enfoque eligiendo uno de los algoritmos explicados en el apéndice de: Analysis of focus measure
operators in shape from focus.

El algoritmo de detección a implementar debe detectar y devolver los puntos de máximo enfoque de manera
automática.

## TP2

- Encontrar el logotipo de la gaseosa dentro de las imágenes provistas en
tp3/material/images a partir del template tp3/material/template
  1. (4 puntos) Obtener una detección del logo en cada imagen sin falsos positivos
  2. (4 puntos) Plantear y validar un algoritmo para múltiples detecciones en la imagen
  coca_multi.png con el mismo témplate del ítem 1
  3. (2 puntos) Generalizar el algoritmo del item 2 para todas las imágenes.

- Visualizar los resultados con bounding boxes en cada imagen mostrando el nivel de confianza
de la detección.