
# CT-maskgenerator-pdi - Segmentation Tool for CT images.

*Trabajo final de la materia "Procesamiento Digital de Imágenes" (FICH - UNL).*

Dependencias: SimpleITK, SimpleElastix, numpy, medpy, matplotlib.

En el Main ejecutar sólo una función a la vez:

**Registrar:**
   - Normalizar y registrar imágenes `registerImgs(inputImgPath='~/pdi/tpf/data', savePath='reg', normalize=True)`
   - Registrar (sin normalizar), útil para GraphCuts: `registerImgs(inputImgPath='~/pdi/tpf/data', savePath='reg', normalize=False)`

**GraphCuts:**
   - Crear el .csv para el algoritmo de GraphCuts, con la función `f.createCSV("~/pdi/tpf/data/reg/craniectomy") `
   - Compilar y ejecutar "ITKGraphCutSegmentation/Examples/ImageGraphCut3DSegmentationExample.cpp" cambiando antes la direccion del csv por la del generado anteriormente
   - Este algoritmo guardará las segmentaciones de graphcuts en un directorio con ese nombre en la misma carpeta del (o los) archivo(s) procesado(s). Esta segmentación se cargará en el algoritmo siguiente para comparar.

**Level Sets y Region growing - Comparacion de metodos:**
   - Ejecutar `mainRegistrado()` esableciendo antes las variables inputImgsPath (carpeta con imagenes de entrada),     imageAtlasPath (Atlas de TAC), maskAtlasPath (Mascara que vamos a usar para inicializar), paramPath (Mapa de parametros a usar en la registracion) y savePath (Carpeta donde se guardan las segmentaciones).
   - Al terminar de ejecutarse este algoritmo, se crearán las imágenes de los boxplot en CT-maskgenerator-pdi (dice.png, hausdorff.png y asd.png).
   - En consola se mostrará el detalle de cada segmentación (la cual no se guardará) y la media y varianza de las medidas de la calidad de la segmentación.
   - En la carpeta de salida se guardarán todas las imágenes:
	   - La TAC registrada y normalizada (*regCT+name*).
	   - El Ground Truth registrado (*regGT+name*).
	   - Inicialización (semilla) utilizada (*atlasMask+name*).
	   - Segmentaciones generadas:
	      - Level-Sets (*lvlSet_+name*).
	      - Region Growing (*ConfConnected_+name*)
	      - GraphCuts (*GraphCuts_+name*)
