import SimpleITK as sitk
import math
import numpy as np


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


def resample(img, targetSpacing, isMask = True):
    """
    Resample image to match targetSpacing
    :param img: input image
    :param targetSpacing: desired spacing
    :param isMask: true if it's a segmentation
    :return:
    """
    originalSize = img.GetSize()
    originalSpacing = img.GetSpacing()

    targetSize = [int(math.ceil(originalSize[0] * (originalSpacing[0] / targetSpacing[0]))),
                  int(math.ceil(originalSize[1] * (originalSpacing[1] / targetSpacing[1]))),
                  int(math.ceil(originalSize[2] * (originalSpacing[2] / targetSpacing[2])))]

    if isMask:
        # For binary images, it's better to use Nearest Neighbor
        interpolator = sitk.sitkNearestNeighbor
    else:
        interpolator = sitk.sitkLinear

    resampledImg = sitk.Resample(img, targetSize, sitk.Transform(),
                                 interpolator, img.GetOrigin(),
                                 targetSpacing, img.GetDirection(), 0.0,
                                 img.GetPixelIDValue())

    return resampledImg


def register(movingImg, fixedImage, parameterMap, isMask = True):
    if isMask:
        # Switch to Nearest Neighbor (because it's a mask)
        parameterMap["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]

    elastixImageFilter = sitk.ElastixImageFilter()

    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.LogToFileOff()

    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImg)
    elastixImageFilter.SetParameterMap(parameterMap)
    elastixImageFilter.Execute()
    resultImage = elastixImageFilter.GetResultImage()

    return resultImage


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