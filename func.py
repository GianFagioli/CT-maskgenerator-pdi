# -*- coding: utf-8 -*-

import os
import SimpleITK as sitk
from matplotlib import pyplot as plt
import math
import numpy as np
import csv


def retainLargestConnectedComponent(image):
    """
       Retains only the largest connected component of a binary image, and returns it.
   """
    connectedComponentFilter = sitk.ConnectedComponentImageFilter()
    objects = connectedComponentFilter.Execute(image)

    # If there is more than one connected component
    if connectedComponentFilter.GetObjectCount() > 1:
        objectsData = sitk.GetArrayFromImage(objects)

        # Detect the largest connected component
        maxLabel = 1
        maxLabelCount = 0
        for i in range(1, connectedComponentFilter.GetObjectCount() + 1):
            componentData = objectsData[objectsData == i]

            if len(componentData.flatten()) > maxLabelCount:
                maxLabel = i
                maxLabelCount = len(componentData.flatten())

        # Remove all the values, exept the ones for the largest connected component

        dataAux = np.zeros(objectsData.shape, dtype=np.int8)

        # Fuse the labels

        dataAux[objectsData == maxLabel] = 1

        # Save edited data
        output = sitk.GetImageFromArray(dataAux)
        output.SetSpacing(image.GetSpacing())
        output.SetOrigin(image.GetOrigin())
        output.SetDirection(image.GetDirection())
    else:
        output = image

    return output

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


# Create csv file of the files in dataPath and subfolders
def createCSV(dataPath):
    print "Creating CSV file..."
    filelist = []
    for root, dirs, files in os.walk(dataPath):
        for name in sorted(files, key=str.lower):
            if name.endswith("_head_mask.nii.gz"):
                name = os.path.join(root, name)
                names = (name.replace("_head_mask.nii.gz",'_image.nii.gz'), name)
                filelist.append(names)
            else:
                continue

    nameCSV = 'files.csv'
    print "   writing file", nameCSV
    with open(os.path.join(dataPath, nameCSV), 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerows(filelist)
    print "Done."


def boxplots(stats_levelsets, stats_reggrow, stats_graphcuts):
    # Dices de los métodos
    dice, ax = plt.subplots()
    dice.canvas.draw()
    plt.boxplot([stats_levelsets[:, 0], stats_reggrow[:, 0], stats_graphcuts[:, 0]])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Level Sets'
    labels[1] = 'Region Growing'
    labels[2] = 'Graph Cuts'
    ax.set_xticklabels(labels)
    plt.title('Dice Coefficient')
    plt.savefig('dice.png')

    # Hausdorff de los métodos
    hausdorff, ax = plt.subplots()
    hausdorff.canvas.draw()
    plt.boxplot([stats_levelsets[:, 1], stats_reggrow[:, 1], stats_graphcuts[:, 1]])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Level Sets'
    labels[1] = 'Region Growing'
    labels[2] = 'Graph Cuts'
    ax.set_xticklabels(labels)
    plt.title('Hausdorff Distance')
    plt.savefig('hausdorff.png')

    # Average Surface Distance de los métodos
    hausdorff, ax = plt.subplots()
    hausdorff.canvas.draw()
    plt.boxplot([stats_levelsets[:, 2], stats_reggrow[:, 2], stats_graphcuts[:, 2]])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = 'Level Sets'
    labels[1] = 'Region Growing'
    labels[2] = 'Graph Cuts'
    ax.set_xticklabels(labels)
    plt.title('Average Surface Distance')
    plt.savefig('asd.png')

    # Means and std
    print 'Dice mean and std'
    print '   Levelset: ', np.mean(stats_levelsets[:, 0]), np.std(stats_levelsets[:, 0])
    print '   Region Growing: ', np.mean(stats_reggrow[:, 0]), np.std(stats_reggrow[:, 0])
    print '   Graph Cuts: ', np.mean(stats_graphcuts[:, 0]), np.std(stats_graphcuts[:, 0])

    print 'Hausdorff Distance mean and std'
    print '   Levelset: ', np.mean(stats_levelsets[:, 1]), np.std(stats_levelsets[:, 1])
    print '   Region Growing: ', np.mean(stats_reggrow[:, 1]), np.std(stats_reggrow[:, 1])
    print '   Graph Cuts: ', np.mean(stats_graphcuts[:, 1]), np.std(stats_graphcuts[:, 1])

    print 'Average Surface Distance Metric mean and std'
    print '   Levelset: ', np.mean(stats_levelsets[:, 2]), np.std(stats_levelsets[:, 2])
    print '   Region Growing: ', np.mean(stats_reggrow[:, 2]), np.std(stats_reggrow[:, 2])
    print '   Graph Cuts: ', np.mean(stats_graphcuts[:, 2]), np.std(stats_graphcuts[:, 2])