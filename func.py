import os, sys
import SimpleITK as sitk
from matplotlib import pyplot as plt
import math
import numpy as np


def maxSize(dataPath):
    i = 0
    for root, dirs, files in os.walk(dataPath):
        for name in sorted(files, key=len):
            if name.endswith(".nii.gz"):
                filepath = os.path.join(root, name)
                img = sitk.ReadImage(filepath)
                if img is not None:
                    timg = img.GetSize()
                    if i == 0:
                        tam = list(timg)
                    else:
                        tam[0] = max(tam[0], timg[0])
                        tam[1] = max(tam[1], timg[1])
                        tam[2] = max(tam[2], timg[2])
                    i += 1
    print "Maximun possible size: ", tam
    return tam


def saveImage(inputPath, outputFolder, outImg):
    # Extract path and filename
    pathImg, filenameImg = os.path.split(inputPath)

    # Create out directory if not exists
    outputDir = os.path.join(pathImg, outputFolder);
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    savedImg = os.path.join(outputDir, filenameImg)
    # Write image
    sitk.WriteImage(outImg, savedImg)
    return savedImg


# Resample image using 1mm3 spacing (targetSpacing)
def resample(imgPath):
    img = sitk.ReadImage(imgPath)
    isMask = imgPath.endswith("origSize_brainmaskLV_FIXED.nii.gz") or imgPath.endswith("_seg.nii.gz")
    print "File: " + imgPath

    originalSize = img.GetSize()
    originalSpacing = img.GetSpacing()

    targetSpacing = [1, 1, 1]

    targetSize = [int(math.ceil(originalSize[0] * (originalSpacing[0] / targetSpacing[0]))),
                  int(math.ceil(originalSize[1] * (originalSpacing[1] / targetSpacing[1]))),
                  int(math.ceil(originalSize[2] * (originalSpacing[2] / targetSpacing[2])))]

    if isMask:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear

    resampledImg = sitk.Resample(img, targetSize, sitk.Transform(),
                                 interpolator, img.GetOrigin(),
                                 targetSpacing, img.GetDirection(), 0.0,
                                 img.GetPixelIDValue())

    return saveImage(imgPath, "resampled", resampledImg)


def register(fixedImPath, movingImPath, paramPath):
    fixedImage = sitk.ReadImage(fixedImPath)
    movingImage = sitk.ReadImage(movingImPath)
    isMask = movingImPath.endswith("origSize_brainmaskLV_FIXED.nii.gz") or movingImPath.endswith("_seg.nii.gz")
    parameterMap = sitk.ReadParameterFile(paramPath)

    # If it's a mask, I need to use the previously saved parametermaps and Nearest Neighbor interp.
    if isMask:
        # Get the Transform parameters file from mask filename
        pathImg, filenameImg = os.path.split(movingImPath)
        if movingImPath.endswith("origSize_brainmaskLV_FIXED.nii.gz"):
            paramFilename = filenameImg.replace("_origSize_brainmaskLV_FIXED.nii.gz", ".nii.gz_reg.txt")
        else:
            paramFilename = filenameImg.replace("_seg.nii.gz", ".nii.gz_reg.txt")
        paramFile = os.path.join(pathImg, "registered", paramFilename)

        # Make sure that the file exists
        try:
            currentTransform = sitk.ReadParameterFile(paramFile)
        except:
            print paramFile + " not found. Did you run register on " + filenameImg + "?"
            return False

        # Switch to Nearest Neighbor
        currentTransform["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

        # Apply transformation to mask
        resultImage = sitk.Transformix(movingImage, currentTransform)
        savedPath = saveImage(movingImPath, "registered", resultImage)
    else:
        # Normal image, use the atlas and the default parameter map.
        elastixImageFilter = sitk.ElastixImageFilter()

        elastixImageFilter.LogToConsoleOn()

        elastixImageFilter.LogToFileOff()
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(movingImage)
        elastixImageFilter.SetParameterMap(parameterMap)
        elastixImageFilter.Execute()
        resultImage = elastixImageFilter.GetResultImage()

        # Save the registered image and its parameters map
        transformParameterMap = elastixImageFilter.GetTransformParameterMap(0)
        savedPath = saveImage(movingImPath, "registered", resultImage)
        elastixImageFilter.WriteParameterFile(transformParameterMap, savedPath + "_reg.txt")
    return savedPath


# Normalize using Z-scores
def normalizeZS(imgPath):
    img = sitk.ReadImage(imgPath)

    data = sitk.GetArrayFromImage(img)

    # data has integer values (data.dtype), so it needs to be casted to float
    dataAux = data.astype(float)

    # Find mean, std then normalize
    mean = dataAux.mean()
    std = dataAux.std()
    dataAux = dataAux - float(mean)
    dataAux = dataAux / float(std)

    # Save edited data
    outImg = sitk.GetImageFromArray(dataAux)
    outImg.SetSpacing(img.GetSpacing())
    outImg.SetOrigin(img.GetOrigin())
    outImg.SetDirection(img.GetDirection())

    return saveImage(imgPath, "normalized", outImg)


# Resize folder imgs mantaing the same spacing
def resizeimgfixed(imgPath, maxpx):
    img = sitk.ReadImage(imgPath)
    isMask = imgPath.endswith("origSize_brainmaskLV_FIXED.nii.gz") or imgPath.endswith("_seg.nii.gz")
    print "File: " + imgPath

    originalSize = img.GetSize()
    maxval = max(originalSize)

    targetSpacing = list(img.GetSpacing())

    targetSize = list(originalSize)
    targetSize = [int(1.0 * x * maxpx / maxval) for x in targetSize]
    targetSpacing = [1.0 * x * maxval / maxpx for x in targetSpacing]

    if isMask:
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear

    resampledImg = sitk.Resample(img, targetSize, sitk.Transform(),
                                 interpolator, img.GetOrigin(),
                                 targetSpacing, img.GetDirection(), 0.0,
                                 img.GetPixelIDValue())

    return saveImage(imgPath, "resized", resampledImg)


def histImgs(imgPath):
    img = sitk.ReadImage(imgPath)
    arr = sitk.GetArrayFromImage(img)
    plt.hist(arr.flatten(), 128, histtype="step")


def thresImg(imgPath):
    img = sitk.ReadImage(imgPath)
    img = sitk.BinaryThreshold(img,
                               lowerThreshold=900, upperThreshold=1200,
                               insideValue=1, outsideValue=0)
    return sitk.GetArrayFromImage(img)


# bbox3: get the coordinates of the bounding box in 3D
# based on this 2D code: https://stackoverflow.com/a/31402351
def bbox3(imgPath):
    img = sitk.ReadImage(imgPath)
    old = img

    # Apply a Threshold in the image, I choose 900 and 1200 as lower and upper values
    # watching the superimposed histograms (histimgs function)
    img = sitk.BinaryThreshold(img,
                               lowerThreshold=1500, upperThreshold=2966,
                               insideValue=1, outsideValue=0)
    # For using numpy functions
    img = sitk.GetArrayFromImage(img)

    xmin, xmax, ymin, ymax, zmin, zmax = np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf,
    for i in xrange(img.shape[2]):
        x = np.any(img[:, :, i], axis=1)
        y = np.any(img[:, :, i], axis=0)
        pos_zero_x = np.where(x)
        pos_zero_y = np.where(y)
        if len(pos_zero_x[0]):
            xmin_a, xmax_a = pos_zero_x[0][[0, -1]]
            xmin, xmax = min(xmin, xmin_a), max(xmax, xmax_a)
        if len(pos_zero_y[0]):
            ymin_a, ymax_a = pos_zero_y[0][[0, -1]]
            ymin, ymax = min(ymin, ymin_a), max(ymax, ymax_a)

    for i in xrange(img.shape[1]):
        if i == img.shape[1]/2:
            print 3
        z = np.any(img[:, i, :], axis=0)
        pos_zero_z = np.where(z)
        if len(pos_zero_z[0]):
            zmin_a, zmax_a = np.where(z)[0][[0, -1]]
            zmin, zmax = min(zmin, zmin_a), max(zmax, zmax_a)

    sitk.Show(sitk.GetImageFromArray(img))
    sitk.Show(sitk.GetImageFromArray(img[xmin:xmax, ymin:ymax, zmin:zmax]))
    # sitk.Show(old)
    # sitk.Show(old[xmin:xmax, ymin:ymax, zmin:zmax])
    return img[ymin:ymax + 1, xmin:xmax + 1, zmin:zmax + 1]