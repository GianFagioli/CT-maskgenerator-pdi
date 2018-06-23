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
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        pathImg, filenameImg = os.path.split(self.fName)

        savedImg = os.path.join(savePath, 'regCT' + filenameImg)
        # Write registered ct image
        sitk.WriteImage(self.inputct, savedImg)

        # Write generated masks
        for i, mask in enumerate(self.MaskArray):
            maskName = self.MaskArray[i][1]
            if not maskName:
                maskName = 'mask'

            savedMask = os.path.join(savePath, maskName + '_' + filenameImg)
            sitk.WriteImage(self.MaskArray[i][0], savedMask)

        return savedImg


    def register(self, registerGT = False, savePath=None, saveGT=False):
        # Register input CT scan to the atlas, getting the atlas (isotropic) dimensions in the input image
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

        if savePath is not None:
            folder = os.path.join(self.fName.replace(os.path.split(self.fName)[-1],''),savePath)
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, os.path.split(self.fName)[-1])
            sitk.WriteImage(registeredImage, filename)

            outParamFile = os.path.join(filename + "_reg.txt")
            elastixImageFilter.WriteParameterFile(outTPM, outParamFile)

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

            if saveGT:
                sitk.WriteImage(registeredMask, filename.replace("image","head_mask"))


    def erodeInitialMask(self):
        for i in xrange(4):
            self.MaskArray[0][0] = sitk.ErodeObjectMorphology(self.MaskArray[0][0])
        #sitk.Show(movingImage)


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

        gac = sitk.GeodesicActiveContourLevelSetImageFilter()
        gac.SetPropagationScaling(1.0)
        gac.SetCurvatureScaling(0.2)
        # gac.SetCurvatureScaling(20)
        # gac.SetCurvatureScaling(4)
        gac.SetAdvectionScaling(3.0)
        gac.SetMaximumRMSError(0.01)
        gac.SetNumberOfIterations(2)

        movingImagef = sitk.Cast(self.inputct, sitk.sitkFloat32) * -1 + 0.5

        # En vez de movingImagef uso gm
        gm = sitk.GradientMagnitude(self.inputct)

        # El primer parametro es el volumen que crece, el segundo
        imgg = gac.Execute(sitk.Cast(self.MaskArray[0][0], sitk.sitkFloat32), gm)
        # sitk.Show(imgg)

        # Umbralizo con el valor mas bajo
        imgg > sitk.GetArrayFromImage(imgg).min()

        imgg = imgg - sitk.GetArrayFromImage(imgg).min()
        imgg = imgg > 0

        self.MaskArray.append([imgg, 'lvlSet'])


    def segConfConnected(self):
        coordsMask = np.where(sitk.GetArrayFromImage(self.MaskArray[0][0]))
        coordsMask = zip(coordsMask[0], coordsMask[1], coordsMask[2])

        #seg_explicit_thresholds = sitk.ConnectedThreshold(self.inputct, seedList=coordsMask, lower=100,
        #                                                  upper=170)
        #self.MaskArray.append([seg_explicit_thresholds, 'ConfConnected'])

        seg_implicit_thresholds = sitk.ConfidenceConnected(self.inputct, seedList=[coordsMask[500]],
                                                           numberOfIterations=0,
                                                           multiplier=2,
                                                           initialNeighborhoodRadius=1,
                                                           replaceValue=1)
        self.MaskArray.append([seg_implicit_thresholds, 'ConfConnected'])

        #seg_implicit_threshold_vector = sitk.VectorConfidenceConnected(self.inputct,
        #                                                               coordsMask[1],
        #                                                               numberOfIterations=2,
        #                                                               multiplier=4)
        #self.MaskArray.append([seg_implicit_threshold_vector, 'ConfConnected'])




    def locateGroundTruth(self):
        nameGT = self.fName.replace("_image.nii.gz", "_head_mask.nii.gz")
        self.maskGT = sitk.ReadImage(nameGT)

    def compareWithGT(self):
        img1 = sitk.GetArrayFromImage(self.maskGT)

        i = 1  # Segmentacion que vamos a comparar
        img2 = sitk.GetArrayFromImage(self.MaskArray[i][0])
        print "Segmentacion usada: ", self.MaskArray[i][1]
        print "Dice coefficient: ", metric.binary.dc(img1, img2)
        print "Jaccard coefficient: ", metric.binary.jc(img1, img2)
        # todo Ver como pasarle el spacing!
        # print "Hausdorff distance: ", metric.binary.hd(img1, img2)
        # print "Average surface distance metric: ", metric.binary.asd(img1, img2)
        # print "Average symmetric surface distance distance: ", metric.binary.assd(img1, img2)
        print "Precision : ", metric.binary.precision(img1, img2)
        print "Recall : ", metric.binary.recall(img1, img2)
        print "Sensivity : ", metric.binary.sensitivity(img1, img2)
        print "Specifity: ", metric.binary.specificity(img1, img2)
        print "True positive rate: ", metric.binary.true_positive_rate(img1, img2)
        print "Positive predictive value: ", metric.binary.positive_predictive_value(img1, img2)
        print "Relative absolute volume difference: ", metric.binary.ravd(img1, img2)


def mainRegistrado():
    start_time = time.time()

    # todo Esto deberia ir dentro de un loop que recorra todas las TAC, en vez de una sola (inputImgPath)
    inputImgPath = '../1111/reg/1111_16216_image.nii.gz'  # Imagen que vamos a procesar
    imageAtlasPath = '../Atlas3/atlas3_nonrigid_masked_1mm.nii.gz'  # Atlas de TAC (promedio de muchas tomografias)
    maskAtlasPath = '../Atlas3/atlas3_nonrigid_brain_mask_1mm.nii.gz'  # Mascara que vamos a usar para inicializar
    paramPath = 'Par0000affine.txt'  # Mapa de parametros a usar en la registracion

    savePath = '../1111/seg'  # Carpeta donde se guarda la salida

    # Inicializar y cargar
    imgs = CTandMasks()
    imgs.open(inputImgPath, imageAtlasPath, paramPath, maskAtlasPath)

    # Erosionar mascara
    imgs.erodeInitialMask()

    # Obtener la segmentacion por level sets
    imgs.segLevelSets()

    # Obtener la segmentacion por crecimiento de regiones
    imgs.segConfConnected()

    # Cargar el Ground Truth a partir del nombre de la imagen de entrada
    imgs.locateGroundTruth()

    # Realizar estadisticas comparando las diferentes mascaras con el Ground Truth
    imgs.compareWithGT()

    # todo agregar boxplot, estudiar graficos para comparar

    # Guardar los archivos en la carpeta establecida
    imgs.saveFiles(savePath)

    print "Elapsed time was " + str(int(time.time() - start_time)) + " seconds."


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
    # registerImgs(inputImgPath='../1111', savePath='reg')

    mainRegistrado()