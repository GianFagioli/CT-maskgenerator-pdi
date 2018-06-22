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


    def open(self, inputImgPath, imageAtlasPath, maskAtlasPath, paramPath):
        """
            Loads an image.
        """
        self.inputct = sitk.ReadImage(inputImgPath)
        self.fName = inputImgPath # Store the img name for saving later
        self.atlasct = sitk.ReadImage(imageAtlasPath)

        # MaskArray: Lista que contiene listas, compuestas por una máscara como primer atributo, y el nombre archivo
        # como segundo atributo.
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


    def register(self, registerGT = False):
        # Register input CT scan to the atlas, getting the atlas (isotropic) dimensions in the input image
        elastixImageFilter = sitk.ElastixImageFilter()

        elastixImageFilter.LogToConsoleOn()
        elastixImageFilter.LogToFileOff()

        # Depende si es una mascara o una imagen normal registro diferente
        if registerGT:  # Si es una mascara
            elastixImageFilter.SetFixedImage(self.MaskArray[0][0])
            elastixImageFilter.SetMovingImage(self.maskGT)
            self.paramMap["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
        else:
            elastixImageFilter.SetFixedImage(self.atlasct)
            elastixImageFilter.SetMovingImage(self.inputct)
            self.paramMap["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
        elastixImageFilter.SetParameterMap(self.paramMap)
        elastixImageFilter.Execute()

        if registerGT:
            self.maskGT = elastixImageFilter.GetResultImage()
        else:
            # Sobreescribo la imagen de entrada con la registrada
            self.inputct = elastixImageFilter.GetResultImage()


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
        # gac.SetCurvatureScaling(4)
        gac.SetAdvectionScaling(3.0)
        gac.SetMaximumRMSError(0.01)
        gac.SetNumberOfIterations(200)

        movingImagef = sitk.Cast(self.inputct, sitk.sitkFloat32) * -1 + 0.5

        # El primer parametro es el volumen que crece, el segundo
        imgg = gac.Execute(sitk.Cast(self.MaskArray[0][0], sitk.sitkFloat32), movingImagef)
        # sitk.Show(imgg)

        # Umbralizo con el valor mas bajo
        imgg > sitk.GetArrayFromImage(imgg).min()

        imgg = imgg - sitk.GetArrayFromImage(imgg).min()
        imgg = imgg > 0

        self.MaskArray.append([imgg, 'lvlSet'])


    def locateGroundTruth(self):
        nameGT = self.fName.replace("_image.nii.gz", "_head_mask.nii.gz")
        self.maskGT = sitk.ReadImage(nameGT)
        self.register(registerGT=True)


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


if __name__ == "__main__":
    start_time = time.time()

    # todo Esto deberia ir dentro de un loop que recorra todas las TAC, en vez de una sola (inputImgPath)
    inputImgPath = '1111/1111_16216_image.nii.gz'  # Imagen que vamos a procesar
    imageAtlasPath = 'Atlas3/atlas3_nonrigid_masked_1mm.nii.gz'  # Atlas de TAC (promedio de muchas tomografias)
    maskAtlasPath = 'Atlas3/atlas3_nonrigid_brain_mask_1mm.nii.gz'  # Mascara que vamos a usar para inicializar
    paramPath = 'Par0000affine.txt' # Mapa de parametros a usar en la registracion

    savePath = '1111' # Carpeta donde se guarda la salida

    # Inicializar y cargar
    imgs = CTandMasks()
    imgs.open(inputImgPath, imageAtlasPath, maskAtlasPath, paramPath)

    # Registrar la imagen de entrada
    imgs.register()

    # Obtener la segmentacion por level sets
    imgs.segLevelSets()

    # Cargar el Ground Truth a partir del nombre de la imagen de entrada
    imgs.locateGroundTruth()

    # Realizar estadisticas comparando las diferentes mascaras con el Ground Truth
    imgs.compareWithGT()

    # todo agregar boxplot, estudiar graficos para comparar

    # Guardar los archivos en la carpeta establecida
    imgs.saveFiles(savePath)

    print "Elapsed time was " + str(int(time.time() - start_time)) + " seconds."
