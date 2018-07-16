
# CT-maskgenerator-pdi - Segmentation Tool for CT images.

Dependencias: SimpleITK, SimpleElastix, numpy, medpy, matplotlib.

**Importante: MEDPY**

Si al ejecutar el código falla alguna de las funciones de medpy, es porque se restan dos arreglos de numpy para compararlos y esto ya no se permite. Para solucionarlo, hay que editar el archivo */local/lib/python2.7/site-packages/medpy/metric/binary.py*, cambiando en `__surface_distances(result, reference, voxelspacing=None, connectivity=1)` las siguientes líneas:

    result_border = result - binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference - binary_erosion(reference, structure=footprint, iterations=1)

por:

    result_border = numpy.logical_xor(result, binary_erosion(result, structure=footprint, iterations=1))
    reference_border = numpy.logical_xor(reference, binary_erosion(reference, structure=footprint, iterations=1))


**Registrar:**
   - Normalizar y registrar imágenes `registerImgs(inputImgPath='~/pdi/tpf/data', savePath='reg', normalize=True)`

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
