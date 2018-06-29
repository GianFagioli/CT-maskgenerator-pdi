# -*- coding: utf-8 -*-
import os
import time
from matplotlib import pyplot as plt
import func as f
import SimpleITK as sitk
import math
import numpy as np
import csv
import nibabel as nib
from medpy import metric


class CTandMasks:
    """
        Clase hecha para cargar las tomografías e implementar dentro todas las funciones necesarias para hallar los dis-
        tintos tipos de segmentaciones.
    """

    def __init__(self):
        self.inputct = None  # TAC sin preprocesar (imagen stik)
        self.atlasct = None  # Atlas de las TAC (imagen stik). Imagen fija de la registracion
        self.maskGT = None # Ground truth (mascara que queremos alcanzar)
        self.MaskArray = []  # Lista que contiene varias máscaras (imagenes stik)
        self.fName = ""  # Nombre del archivo de entrada (se usara para adivinar el nombre del archivo del Ground Truth)

        # Transformation details of a registration (useful for later transformation of the masks)
        self.paramMap = None

        # output folder
        self.savePath = ""


    def open(self, inputImgPath, imageAtlasPath, paramPath, maskAtlasPath=None):
        """
            Loads an image.
        """
        print "Abriendo archivos: "
        print "   TAC de entrada: ", inputImgPath
        print "   Atlas TAC: ", imageAtlasPath
        print "   Parametros de la registracion: ", paramPath

        self.inputct = sitk.ReadImage(inputImgPath)
        self.fName = inputImgPath # Store the img name for saving later
        self.atlasct = sitk.ReadImage(imageAtlasPath)

        # MaskArray: Lista que contiene listas, compuestas por una máscara como primer atributo, y el nombre archivo
        # como segundo atributo.
        if maskAtlasPath is not None:
            self.MaskArray.append([sitk.ReadImage(maskAtlasPath), 'atlasMask'])  # La primera máscara será la del atlas

        self.paramMap = sitk.ReadParameterFile(paramPath)

    def saveFiles(self, savePath):
        """
        Guardar la imagen de entrada registrada y todas las mascaras generadas (incluso el atlas solamente registrado)
        en la carpeta de salida.
        """
        print "Guardando archivos..."
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        pathImg, filenameImg = os.path.split(self.fName)

        savedImg = os.path.join(savePath, 'regCT' + filenameImg)
        # Write registered ct image
        sitk.WriteImage(self.inputct, savedImg)
        print "   Imagen de entrada registrada: ", savedImg

        # Write registered ground truth
        if self.maskGT is not None:
            maskGTpath = os.path.join(savePath, 'regGT' + filenameImg)
            sitk.WriteImage(self.maskGT, maskGTpath)
            print "   Ground truth registrado: ", maskGTpath

        # Write generated masks
        for i, mask in enumerate(self.MaskArray):
            maskName = self.MaskArray[i][1]
            if not maskName:
                maskName = 'mask'

            maskPath = maskName + '_' + filenameImg
            savedMask = os.path.join(savePath, maskPath)
            sitk.WriteImage(self.MaskArray[i][0], savedMask)
            print "   Mascara generada (",maskName,"): ", maskPath

        return savedImg


    def register(self, registerGT = False, savePath=None, saveGT=False):
        # Register input CT scan to the atlas, getting the atlas (isotropic) dimensions in the input image

        print "Registrando..."

        elastixImageFilter = sitk.ElastixImageFilter()

        elastixImageFilter.LogToConsoleOn()
        elastixImageFilter.LogToFileOff()

        elastixImageFilter.SetFixedImage(self.atlasct)
        elastixImageFilter.SetMovingImage(self.inputct)
        self.paramMap["ResampleInterpolator"] = ["FinalBSplineInterpolator"]

        elastixImageFilter.SetParameterMap(self.paramMap)
        elastixImageFilter.Execute()

        registeredImage = elastixImageFilter.GetResultImage()
        self.inputct = registeredImage

        # Save parameter map
        outTPM = elastixImageFilter.GetTransformParameterMap(0)

        print "   Imagen de entrada registrada."

        if savePath is not None:
            folder = os.path.join(self.fName.replace(os.path.split(self.fName)[-1],''),savePath)
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, os.path.split(self.fName)[-1])
            sitk.WriteImage(registeredImage, filename)
            print "   Imagen registrada guardada en ", filename

            outParamFile = os.path.join(filename + "_reg.txt")
            elastixImageFilter.WriteParameterFile(outTPM, outParamFile)

            print "   Parametermap guardado."

            # Depende si es una mascara o una imagen normal registro diferente
            if registerGT:  # Si es una mascara
                # Get the Transform parameters file from mask filename
                pathImg, filenameImg = os.path.split(self.fName)

                # Make the route to the parameter file PREVIOUSLY created (if I'm running this on a mask)
                folder = os.path.join(self.fName.replace(os.path.split(self.fName)[-1], ''), savePath)
                filename = os.path.join(folder, os.path.split(self.fName)[-1])

                paramFilename = os.path.join(filename + "_reg.txt")

                paramFile = os.path.join(self.savePath, paramFilename)

                # Make sure that the parameter map exists
                try:
                    currentTransform = sitk.ReadParameterFile(paramFile)
                except:
                    print paramFile + " not found. Did you run register on " + filenameImg + "?"
                    return False

                # Switch to Nearest Neighbor
                currentTransform["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

                # Apply transformation to mask
                registeredMask = sitk.Transformix(self.maskGT, currentTransform)
                self.maskGT = registeredMask

                print "   Mascara registrada exitosamente."

            if saveGT:
                maskSavePath = filename.replace("image","head_mask")
                sitk.WriteImage(registeredMask, maskSavePath)
                print "   Mascara guardada en ", maskSavePath


    def erodeInitialMask(self):
        """
        Erosionar la mascara inicial (atlas) para usarlo como semilla en la segmentacion
        """

        nErode = 8
        kernelRadius = 5
        foregroundVal = 1

        print "Erosion de la mascara inicial"
        print "  Cantidad de iteraciones: ",nErode,", radio del kernel: ", kernelRadius,", valor del frente: ", foregroundVal,"."

        filter = sitk.BinaryErodeImageFilter()
        filter.SetKernelRadius(kernelRadius)
        filter.SetForegroundValue(foregroundVal)

        for i in xrange(nErode):
            self.MaskArray[0][0] = filter.Execute(self.MaskArray[0][0])
        print "   Imagen erosionada."


    def erodeBackgForeg(self, fgPath, bgPath):
        """
        Useful for GraphCuts (c++) method, it generates foreground and background masks.
        :return:
        """
        nErodeF = 3  # Times I'll erode the foreground seed
        nErodeB = 3  # Times I'll erodde the background (0 will make inverse of the foreground)
        kernelRadius = 5
        foregroundVal = 1

        print "Erosion de la mascara inicial (foreground)"
        print "  Cantidad de iteraciones: ",nErodeF,", radio del kernel: ", kernelRadius,", valor del frente: ", foregroundVal,"."

        filter = sitk.BinaryErodeImageFilter()
        filter.SetKernelRadius(kernelRadius)
        filter.SetForegroundValue(foregroundVal)

        foregroundSeed = self.MaskArray[0][0]

        for i in xrange(nErodeF):
            foregroundSeed = filter.Execute(foregroundSeed)
        print "   Foreground erosionado."

        print "Erosion de la mascara inicial invertida (background)"
        print "  Cantidad de iteraciones: ",nErodeB,", radio del kernel: ", kernelRadius,", valor del frente: ", foregroundVal,"."

        backgroundSeed = foregroundSeed < 1

        for i in xrange(nErodeB):
            backgroundSeed = filter.Execute(backgroundSeed)
        print "   Background erosionado."

        sitk.WriteImage(foregroundSeed, fgPath)
        sitk.WriteImage(backgroundSeed, bgPath)

        print "   Mascaras para GraphCuts guardadas."
        print "     Foreground: ", fgPath
        print "     Background: ", bgPath


    def segLevelSets(self):
        # Level sets method segmentation

        # GeodesicActiveContourLevelSetImageFilter:
        # - The first input is a initial level set. The initial level set is a real image which contains the initial
        # contour/surface as the zero level set. For example, a signed distance function from the initial contour/surface
        # is typically used. The initial contour does not have to lie wholly within the shape to be segmented.
        # The initial contour is allow to overlap the shape boundary. The extra advection term in the update equation
        # behaves like a doublet and attracts the contour to the boundary.

        # - The second input is the feature image. For this filter, this is the edge potential map. General characteristics
        # of an edge potential map is that it has values close to zero in regions near the edges and values close to one
        # inside the shape itself. Typically, the edge potential map is compute from the image gradient.

        # La primer entrada, osea la semilla  debe estar invertida, es decir, el fondo debe ser blanco y la semilla negra

        print "Segmentacion por Level Sets"

        propSc, curvSc, advSc, maxRMS, nIter = 1, 1, 1, 0.02, 1200

        gac = sitk.GeodesicActiveContourLevelSetImageFilter()
        gac.SetPropagationScaling(propSc)
        gac.SetCurvatureScaling(curvSc)
        gac.SetAdvectionScaling(advSc)
        gac.SetMaximumRMSError(maxRMS)
        gac.SetNumberOfIterations(nIter)

        print "   Propagation Scaling: ",propSc,", curvature scaling: ",curvSc,", advection scaling: ",advSc,", max RMS error: ",maxRMS,", cantidad iteraciones: ",nIter,"."

        seed = self.MaskArray[0][0]
        seed = sitk.Cast(seed, sitk.sitkFloat32) * -1 + 0.5

        # Get the border map
        gm = sitk.GradientMagnitude(self.inputct)
        gm = gm < 100

        # El primer parametro es el volumen que crece, el segundo el mapa de bordes
        imgg = gac.Execute(seed, sitk.Cast(gm, sitk.sitkFloat32))

        imgg = imgg < 0

        self.MaskArray.append([imgg, 'lvlSet'])
        print "   Segmentacion por Level Sets generada."


    def segConfConnected(self):
        # Use as seeds the positions of the white pixels of the eroded mask
        coordsMask = np.where(sitk.GetArrayFromImage(self.MaskArray[0][0]))
        coordsMask = zip(coordsMask[0], coordsMask[1], coordsMask[2])

        nIter, multip, initRadius, replaceVal = 5, 2.5, 0, 1

        print "Segmentacion por crecimiento de regiones"
        print "   Cantidad de iteraciones: ",nIter,", multiplicador: ",multip,", valor de reemplazo: ",replaceVal,"."
        # Segmentation using region growing
        seg_implicit_thresholds = sitk.ConfidenceConnected(self.inputct, seedList=coordsMask,
                                                           numberOfIterations=nIter,
                                                           multiplier=multip,
                                                           initialNeighborhoodRadius=initRadius,
                                                           replaceValue=replaceVal)

        print "   Operaciones morfologicas (Apertura - Componente Conectada - Cierre)"
        # Apply opening for separating big regions
        vectorRadius = (1, 1, 1)
        kernel = sitk.sitkBall
        seg_implicit_thresholds = sitk.BinaryMorphologicalOpening(seg_implicit_thresholds, vectorRadius, kernel)

        # Keep the biggest Connected Component (the brain)
        seg_implicit_thresholds = f.retainLargestConnectedComponent(seg_implicit_thresholds)

        # Closening for filling holes
        for i in range(10):
            seg_implicit_thresholds = sitk.BinaryMorphologicalClosing(seg_implicit_thresholds, 3)

        self.MaskArray.append([seg_implicit_thresholds, 'ConfConnected'])
        print "   Segmentacion por Crecimiento de regiones generada."


    def segGraphCuts(self):
        """
        Locate the GraphCuts segmentation (it should be in the reg/graphcuts folder and have the same name as input filename).
        but adding "GC_" at the beginning.
        """
        path, name = os.path.split(self.fName)
        pathGC = os.path.join(path, "graphcuts","GC_" + name)
        print "Cargando segmentacion por Graph Cuts"
        print "   archivo: ", pathGC

        segGC = sitk.ReadImage(pathGC)
        segGC = segGC / 255

        print "Segmentación cargada."

        self.MaskArray.append([segGC, 'GraphCuts'])



    def locateGroundTruth(self):
        # From the filename of the input CT scan, get the mask (that should be located in the same folder)
        nameGT = self.fName.replace("_image.nii.gz", "_head_mask.nii.gz")
        self.maskGT = sitk.ReadImage(nameGT)
        print "Ground Truth cargado (",nameGT,")."


    def compareWithGT(self):
        # Get statistics of the segmentations, comparing with the Ground Truth
        # returns a list with the required metrics (the first value will be the name of the segmentation used)
        img1 = sitk.GetArrayFromImage(self.maskGT)

        for i in xrange(1,4):
            img2 = sitk.GetArrayFromImage(self.MaskArray[i][0])
            print "Segmentacion usada: ", self.MaskArray[i][1]
            print "Calidad de la segmentación: "
            dice = metric.binary.dc(img1, img2)
            print "   Dice coefficient: ", dice
            # print "   Jaccard coefficient: ", metric.binary.jc(img1, img2)
            hausdorff = metric.binary.hd(img1, img2, self.maskGT.GetSpacing())
            print "   Hausdorff distance: ", hausdorff
            asd = metric.binary.asd(img1, img2, self.maskGT.GetSpacing())
            print "   Average surface distance metric: ", asd
            # print "   Average symmetric surface distance distance: ", metric.binary.assd(img1, img2, self.maskGT.GetSpacing())
            # print "   Precision : ", metric.binary.precision(img1, img2)
            # print "   Recall : ", metric.binary.recall(img1, img2)
            # print "   Sensivity : ", metric.binary.sensitivity(img1, img2)
            # print "   Specifity: ", metric.binary.specificity(img1, img2)
            # print "   True positive rate: ", metric.binary.true_positive_rate(img1, img2)
            # print "   Positive predictive value: ", metric.binary.positive_predictive_value(img1, img2)
            # print "   Relative absolute volume difference: ", metric.binary.ravd(img1, img2)
            if i == 1:
                levelsets = [dice, hausdorff, asd]
            elif i == 2:
                reggrow = [dice, hausdorff, asd]
            elif i == 3:
                graphcuts = [dice, hausdorff, asd]
        return levelsets, reggrow, graphcuts


def mainRegistrado():
    """
    Calcular las segmentaciones por level sets y region growing de una carpeta determinada, comparando la calidad con el
    Ground Truth que se encuentra en la misma carpeta de las imagenes de entrada.
    :return:
    """
    start_time = time.time()

    inputImgsPath = '../data/reg/normal'  # Imagenes que vamos a procesar
    imageAtlasPath = '../Atlas3/atlas3_nonrigid_masked_1mm.nii.gz'  # Atlas de TAC (promedio de muchas tomografias)
    maskAtlasPath = '../Atlas3/atlas3_nonrigid_brain_mask_1mm.nii.gz'  # Mascara que vamos a usar para inicializar
    paramPath = 'Par0000affine.txt'  # Mapa de parametros a usar en la registracion

    savePath = '../data/reg/normal/output'  # Carpeta donde se guardan las segmentaciones

    stats_levelsets = []
    stats_reggrow = []
    stats_graphcuts = []

    for name in os.listdir(inputImgsPath):
        if os.path.isfile(os.path.join(inputImgsPath, name)):
            if "image" in name and not name.endswith(".txt"):  # Only process CT files, not masks and other stuff.
                inputImgPath = os.path.join(inputImgsPath, name)
                print "File name: ", inputImgPath, '\n'

                # Inicializar y cargar
                imgs = CTandMasks()
                imgs.open(inputImgPath, imageAtlasPath, paramPath, maskAtlasPath)

                # Erosionar mascara
                imgs.erodeInitialMask()

                # Obtener la segmentacion por level sets
                imgs.segLevelSets()

                # Obtener la segmentacion por crecimiento de regiones
                imgs.segConfConnected()

                # Cargar las segmentaciones hechas con graphcuts (para luego compararlas)
                imgs.segGraphCuts()

                # Cargar el Ground Truth a partir del nombre de la imagen de entrada
                imgs.locateGroundTruth()

                # Realizar estadisticas comparando las diferentes mascaras con el Ground Truth
                # Para cada uno de los metodos de segmentacion, obtener las metricas pedidas (3)
                levelsets, reggrow, graphcuts = imgs.compareWithGT()
                if not np.any(stats_levelsets): # If empty, add first row
                    stats_levelsets.append(levelsets)
                    stats_graphcuts.append(graphcuts)
                    stats_reggrow.append(reggrow)
                else:  # Append row
                    stats_levelsets = np.vstack((stats_levelsets, levelsets))
                    stats_reggrow = np.vstack((stats_reggrow, reggrow))
                    stats_graphcuts = np.vstack((stats_graphcuts, graphcuts))

                # Guardar los archivos en la carpeta establecida
                imgs.saveFiles(savePath)

    # Level sets
    plt.figure()
    plt.boxplot(stats_levelsets[:, 0])
    plt.title('Levelset Dice Coefficient')
    plt.savefig('ls_dc.png')

    plt.figure()
    plt.boxplot(stats_levelsets[:, 1])
    plt.title('Levelset Hausdorff Distance')
    plt.savefig('ls_hc.png')

    plt.figure()
    plt.boxplot(stats_levelsets[:, 2])
    plt.title('Levelset Average Surface Distance Metric')
    plt.savefig('ls_asd.png')

    # Region Growing
    plt.figure()
    plt.boxplot(stats_reggrow[:, 0])
    plt.title('Region Growing Dice Coefficient')
    plt.savefig('rg_dc.png')

    plt.figure()
    plt.boxplot(stats_reggrow[:, 1])
    plt.title('Region Growing Hausdorff Distance')
    plt.savefig('rg_hd.png')

    plt.figure()
    plt.boxplot(stats_reggrow[:, 2])
    plt.title('Region Growing Average Surface Distance Metric')
    plt.savefig('rg_asd.png')

    # Graph Cuts
    plt.figure()
    plt.boxplot(stats_graphcuts[:, 0])
    plt.title('Graph Cuts Dice Coefficient')
    plt.savefig('gc_dc.png')

    plt.figure()
    plt.boxplot(stats_graphcuts[:, 1])
    plt.title('Graph Cuts Hausdorff Distance')
    plt.savefig('gc_hd.png')

    plt.figure()
    plt.boxplot(stats_graphcuts[:, 2])
    plt.title('Graph Cuts Average Surface Distance Metric')
    plt.savefig('gc_asd.png')

    print "El tiempo transcurrido fue de " + str(int(time.time() - start_time)) + " segundos."


def registerImgs(inputImgPath=None, savePath=''):
    if inputImgPath is None:
        print "Input folder not specified!"
        return
    print "Output folder: ", savePath

    imageAtlasPath = '../Atlas3/atlas3_nonrigid_masked_1mm.nii.gz'  # Atlas de TAC (promedio de muchas tomografias)
    paramPath = 'Par0000affine.txt'  # Mapa de parametros a usar en la registracion
    maskAtlasPath = '../Atlas3/atlas3_nonrigid_brain_mask_1mm.nii.gz'  # Mascara que vamos a usar para inicializar


    for root, dirs, files in os.walk(inputImgPath):
        for name in files:
            if "image" in name:  # Only process CT files, not masks.
                filepath = os.path.join(root, name)
                print "File name: ", filepath, '\n'

                imgs = CTandMasks()
                imgs.open(filepath, imageAtlasPath, paramPath, maskAtlasPath)

                imgs.locateGroundTruth()
                imgs.register(registerGT=True, savePath=savePath, saveGT=True)


if __name__ == "__main__":
    # Registrar todas las imagenes de la carpeta 1111 y guardarlas dentro de reg
    # registerImgs(inputImgPath='/home/franco/facultad/pdi/tpf/data', savePath='reg')

    # Crear CSV para luego tomarlo con el algoritmo de graphcuts
    # f.createCSV("/home/franco/facultad/pdi/tpf/data/reg/craniectomy")
    # f.createCSV("/home/franco/facultad/pdi/tpf/data/reg/normal")

    # Ejecutar level sets y region growing
    mainRegistrado()