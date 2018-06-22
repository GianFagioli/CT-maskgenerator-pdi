# -*- coding: utf-8 -*-
import os
import time
from matplotlib import pyplot as plt
import SimpleITK as sitk
import math
import numpy as np
import funciones as f

"""
Segmentacion del cerebro en TAC mediante registración, operaciones morfológicas y caracterización de texturas.
La idea es generar una segmentación para una imagen TAC a partir de otra segmentación base 

- Trabajar en las dimensiones de la imagen de entrada o del atlas?
- 

"""


inputImgPath = '1111/1111_16216_image.nii.gz'  # Imagen que vamos a procesar
imageAtlasPath = 'Atlas3/atlas3_nonrigid_masked_1mm.nii.gz'  # Atlas de TAC (promedio de muchas tomografias)
maskAtlasPath = 'Atlas3/atlas3_nonrigid_brain_mask_1mm.nii.gz'  # Mascara que vamos a usar para inicializar
paramPath = 'Par0000affine.txt'

movingImage = sitk.ReadImage(inputImgPath)
mask = sitk.ReadImage(maskAtlasPath)
fixedImage = sitk.ReadImage(imageAtlasPath)
parameterMap = sitk.ReadParameterFile(paramPath)

# Registro la imaegen de entrada para que tenga las mismas caracteristicas que la máscara
movingImage = f.register(movingImage, fixedImage, parameterMap, isMask=False)

feature_img = sitk.GradientMagnitude(movingImage)
feature_img = feature_img>100
# for i in xrange(2):
#     feature_img = sitk.BinaryMorphologicalClosing(feature_img)
#
# feature_img = f.retainLargestConnectedComponent(feature_img)


for i in xrange(2):
    mask = sitk.ErodeObjectMorphology(mask)
sitk.Show(movingImage)

# Snakes method
timeStep_, conduct, numIter = (0.04, 9.0, 5)
imgRecast = sitk.Cast(movingImage, sitk.sitkFloat32)
curvDiff = sitk.CurvatureAnisotropicDiffusionImageFilter()
curvDiff.SetTimeStep(timeStep_)
curvDiff.SetConductanceParameter(conduct)
curvDiff.SetNumberOfIterations(numIter)
imgFilter = curvDiff.Execute(imgRecast)

sigma_ = 2.0
K1, K2 = 18.0, 8.0
alpha_ = (K2 - K1) / 6
beta_ = (K1 + K2) / 2
imgGauss = sitk.GradientMagnitudeRecursiveGaussian(image1=imgFilter, sigma=sigma_)
sigFilt = sitk.SigmoidImageFilter()
sigFilt.SetAlpha(alpha_)
sigFilt.SetBeta(beta_)
sigFilt.SetOutputMaximum(1.0)
sigFilt.SetOutputMinimum(0.0)
imgSigmoid = sigFilt.Execute(imgGauss)

gac = sitk.GeodesicActiveContourLevelSetImageFilter()
gac.SetPropagationScaling(1.0)
gac.SetCurvatureScaling(0.2)
# gac.SetCurvatureScaling(4)
gac.SetAdvectionScaling(3.0)
gac.SetMaximumRMSError(0.01)
gac.SetNumberOfIterations(200)

movingImagef = sitk.Cast(movingImage, sitk.sitkFloat32) * -1 + 0.5

imgg = gac.Execute(sitk.Cast(mask, sitk.sitkFloat32), movingImagef)
sitk.Show(imgg)
imgg[imgg!=0] = 255

imgg = imgg - sitk.GetArrayFromImage(imgg).min()
imgg = imgg > 0

sitk.WriteImage(imgg, 'mascarita.nii.gz')

# sitk.WriteImage(movingImage, 'transformada2.nii.gz')
sitk.WriteImage(imgg, 'mascarita.nii.gz')

#
# # If it's a mask, I need to use the previously saved parametermaps and Nearest Neighbor interp.
# if self.isMask:
#     # Get the Transform parameters file from mask filename
#     pathImg, filenameImg = os.path.split(self.fName)
#
#     # Make the route to the parameter file PREVIOUSLY created (if I'm running this on a mask)
#     if self.isMask:
#         paramFilename = filenameImg.replace("_origSize_brainmaskLV_FIXED.nii.gz", ".nii.gz_reg.txt")
#     else:
#         paramFilename = filenameImg.replace("_seg.nii.gz", ".nii.gz_reg.txt")
#     paramFile = os.path.join(self.savePath, paramFilename)
#
#     # Make sure that the parameter map exists
#     try:
#         currentTransform = sitk.ReadParameterFile(paramFile)
#     except:
#         print paramFile + " not found. Did you run register on " + filenameImg + "?"
#         return False
#
#     # Switch to Nearest Neighbor
#     currentTransform["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
#
#     # Apply transformation to mask
#     resultImage = sitk.Transformix(self.image, currentTransform)
#     self.image = resultImage
# else:
#     # Normal image, use the atlas and the default parameter map.
#     elastixImageFilter = sitk.ElastixImageFilter()
#
#     elastixImageFilter.LogToConsoleOn()
#
#     elastixImageFilter.LogToFileOff()
#     elastixImageFilter.SetFixedImage(fixedImage)
#     elastixImageFilter.SetMovingImage(self.image)
#     elastixImageFilter.SetParameterMap(parameterMap)
#     elastixImageFilter.Execute()
#     resultImage = elastixImageFilter.GetResultImage()
#
#     # Save the registered image and its parameters map
#     transformParameterMap = elastixImageFilter.GetTransformParameterMap(0)
#     self.image = resultImage
#     self.paramMap = transformParameterMap
#
#     # save the transformparametermap
#     outParamFile = os.path.join(self.savePath, os.path.split(self.fName)[1] + "_reg.txt")
#     elastixImageFilter.WriteParameterFile(transformParameterMap, outParamFile)
#
#
# if __name__ == "__main__":
#     start_time = time.time()
#
#
#     print "Elapsed time was " + str(int(time.time() - start_time)) + " seconds."
