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

class Preprocessor:
    """ Preprocessor class. It implements several methods to pre process medical images.

    Attributes:
        image The original imagen opened with method 'open'
        fName Complete filename corresponding to the original opened image
    """

    def __init__(self):
        self.image = None
        self.fName = ""
        self.isMask = False

        # Transformation details of a registration (useful for later transformation of the masks)
        self.paramMap = None

        # output folder
        self.savePath = ""

    def open(self, fileName):
        """
            Loads an image for pre processing.
        """
        self.image = sitk.ReadImage(fileName)
        self.fName = fileName

        # Depending on the filename, the file may be a mask
        self.isMask = self.fName.endswith("origSize_brainmaskLV_FIXED.nii.gz") or self.fName.endswith("_seg.nii.gz")

    def setSavePath(self, savePath):
        self.savePath = savePath
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

    def saveFile(self):
        # Extract path and filename
        pathImg, filenameImg = os.path.split(self.fName)

        savedImg = os.path.join(self.savePath, filenameImg)
        # Write image
        sitk.WriteImage(self.image, savedImg)
        return savedImg

    def calcHist(self):
        f.histImgs(self.fName)

    # Resample image getting 1mm3 voxels
    def resample(self):
        print "Resampling file: " + self.fName

        originalSize = self.image.GetSize()
        originalSpacing = self.image.GetSpacing()

        targetSpacing = [1, 1, 1]

        targetSize = [int(math.ceil(originalSize[0] * (originalSpacing[0] / targetSpacing[0]))),
                      int(math.ceil(originalSize[1] * (originalSpacing[1] / targetSpacing[1]))),
                      int(math.ceil(originalSize[2] * (originalSpacing[2] / targetSpacing[2])))]

        if self.isMask:
            # For binary images, it's better to use Nearest Neighbor
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkLinear

        resampledImg = sitk.Resample(self.image, targetSize, sitk.Transform(),
                                     interpolator, self.image.GetOrigin(),
                                     targetSpacing, self.image.GetDirection(), 0.0,
                                     self.image.GetPixelIDValue())

        self.image = resampledImg

    def register(self, fixedImPath, paramPath):
        fixedImage = sitk.ReadImage(fixedImPath)
        parameterMap = sitk.ReadParameterFile(paramPath)

        # If it's a mask, I need to use the previously saved parametermaps and Nearest Neighbor interp.
        if self.isMask:
            # Get the Transform parameters file from mask filename
            pathImg, filenameImg = os.path.split(self.fName)

            # Make the route to the parameter file PREVIOUSLY created (if I'm running this on a mask)
            if self.isMask:
                paramFilename = filenameImg.replace("_origSize_brainmaskLV_FIXED.nii.gz", ".nii.gz_reg.txt")
            else:
                paramFilename = filenameImg.replace("_seg.nii.gz", ".nii.gz_reg.txt")
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
            resultImage = sitk.Transformix(self.image, currentTransform)
            self.image = resultImage
        else:
            # Normal image, use the atlas and the default parameter map.
            elastixImageFilter = sitk.ElastixImageFilter()

            elastixImageFilter.LogToConsoleOn()

            elastixImageFilter.LogToFileOff()
            elastixImageFilter.SetFixedImage(fixedImage)
            elastixImageFilter.SetMovingImage(self.image)
            elastixImageFilter.SetParameterMap(parameterMap)
            elastixImageFilter.Execute()
            resultImage = elastixImageFilter.GetResultImage()

            # Save the registered image and its parameters map
            transformParameterMap = elastixImageFilter.GetTransformParameterMap(0)
            self.image = resultImage
            self.paramMap = transformParameterMap

            # save the transformparametermap
            outParamFile = os.path.join(self.savePath, os.path.split(self.fName)[1] + "_reg.txt")
            elastixImageFilter.WriteParameterFile(transformParameterMap, outParamFile)

    # Normalize using Z-scores
    def normalizeZS(self):
        data = sitk.GetArrayFromImage(self.image)

        # data has integer values (data.dtype), so it needs to be casted to float
        dataAux = data.astype(float)

        # Find mean, std then normalize
        mean = dataAux.mean()
        std = dataAux.std()
        dataAux = dataAux - float(mean)
        dataAux = dataAux / float(std)

        # Save edited data
        outImg = sitk.GetImageFromArray(dataAux)
        outImg.SetSpacing(self.image.GetSpacing())
        outImg.SetOrigin(self.image.GetOrigin())
        outImg.SetDirection(self.image.GetDirection())

        self.image = outImg

    # Resize folder imgs mantaing the same spacing
    def resizeimgfixed(self, maxpx):
        originalSize = self.image.GetSize()
        maxval = max(originalSize)

        targetSpacing = list(self.image.GetSpacing())

        targetSize = list(originalSize)
        targetSize = [int(1.0 * x * maxpx / maxval) for x in targetSize]

        # el tamaño de cada dimension se definia en base a los tamaños originales de la imagen,
        # pero como esto achica los tamaños de una forma que no queda bien para la red, entonces
        # se llevan todas las dimensiones de la imagen a maxpx. Para volver a como estaba antes,
        # descomentar la linea que sigue y eliminar la linea que establece targetSize como maxpx x3
        # targetSize = [int(1.0 * x * maxpx / maxval) for x in targetSize]
        targetSpacing = [1.0 * x * maxval / maxpx for x in targetSpacing]

        if self.isMask:
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkLinear

        targetSize = [maxpx, maxpx, maxpx]
        resampledImg = sitk.Resample(self.image, targetSize, sitk.Transform(),
                                     interpolator, self.image.GetOrigin(),
                                     targetSpacing, self.image.GetDirection(), 0.0,
                                     self.image.GetPixelIDValue())

        self.image = resampledImg

    def clipIntensity(self, imin, imax):
        arr = sitk.GetArrayFromImage(self.image)
        outimg = sitk.GetImageFromArray(np.clip(arr, imin, imax))
        self.image = outimg

    def normalize(self):
        arr = sitk.GetArrayFromImage(self.image)
        outimg = sitk.GetImageFromArray((arr-arr.min())/(arr.max()-arr.min()))
        self.image = outimg

    def crop(self, pos):
        arr = sitk.GetArrayFromImage(self.image)
        arr = arr[pos[0][0]:pos[0][1], pos[1][0]:pos[1][1], pos[2][0]:pos[2][1]]
        self.image = sitk.GetImageFromArray(arr)


# Method for preprocessing datapath folder files and the files in the subfolders
def prepFSF(dataPath, outFolder, pos = None, resamp=False, regist=False, normal=False,
            hist=False, resize=False):
    # Folder that contains the files to be preprocessed
    print "\n-------------\nRoot folder name: ", dataPath
    # Get the files in dataPath folder and files in subfolders (except output folder)
    for root, dirs, files in os.walk(dataPath):
        if os.path.split(root)[-1] == outFolder:
            print "Skipping output directory for preprocessing..."
            continue

        # In this particular case, I need the files to be sorted because this way I guarantee
        # that I'm taking first the image and then the mask (this is only important
        # when I register an image)
        for i, name in enumerate(sorted(files, key=len)):
            # Im skipping _seg files
            if name.endswith(".nii.gz") and "_seg" not in name:
                filepath = os.path.join(root, name)
                print "File name: ", filepath, '\n'

                if os.path.exists(os.path.join(root, outFolder, name)):
                    print "File already preprocessed, skipping.. " \
                          "(if you want to preprocess again, delete te output folder)"
                    continue

                pp = Preprocessor()
                pp.open(filepath)

                # Depending the folder of the input file, get the output folder path (output its just the
                # out folder name)
                if len(os.path.split(root)) is 2:
                    pp.setSavePath(os.path.join(os.path.split(root)[0], outFolder, os.path.split(root)[1]))
                else:
                    pp.setSavePath(os.path.join(root, outFolder))

                # Calculate its histogram (I need to call plt.show after)
                if hist is True:
                    if not pp.isMask:
                        i += 1
                        pp.calcHist()
                        if i % 16 == 15:
                            plt.figure()
                            plt.draw()

                # Resample image getting 1mm3 voxels
                if resamp is True:
                    pp.resample()

                # Registration parameters
                if regist is True:
                    regAtlasPath = 'Atlas3/atlas3_nonrigid_masked_1mm.nii.gz'
                    regParamPath = 'Par0000affine.txt'
                    pp.register(regAtlasPath, regParamPath)

                # Two normalization methods: normalization with z-scores (1) and clippling
                # intensities in a range and then map to 0-1 (2)
                # 1. Normalize using z-scores (subtract mean and divide by std)
                if normal is True:
                    pp.normalizeZS()

                # #2. Clip intensity values in a given range
                # pp.clipIntensity(-200, 1500)
                # # Normalize in [0-1] range
                # pp.normalize()

                # # Crops the images with the sizes given
                if pos is not None:
                    pp.crop(pos)

                # Resize the image (again) making the largest size 80 pixels
                if resize is True:
                    pp.resizeimgfixed(80)

                # Saves the file in preprocessed folder (set with setSavePath())
                pp.saveFile()
    if hist is True:
        # Show the accumulated histograms
        plt.show()


# Method for getting the bounding box of all overlapped images
def sumMasks(dataPath):
    # Folder that contains the files to be preprocessed
    print "\n-------------\nRoot folder name: ", dataPath
    auximg = None
    i = 0
    # Get the files in dataPath folder and files in subfolders (except output folder)
    for root, dirs, files in os.walk(dataPath):
        # if os.path.split(root)[-1] == outFolder:
        #     print "Skipping output directory for preprocessing..."
        #     continue

        # In this particular case, I need the files to be sorted because this way I guarantee
        # that I'm taking first the image and then the mask (this is only important
        # when I register an image)
        for name in sorted(files, key=len):
            # Im skipping _seg files
            if name.endswith("origSize_brainmaskLV_FIXED.nii.gz"):
                filepath = os.path.join(root, name)
                print "File name: ", filepath, '\n'
                if i is 0:
                    vimg = sitk.ReadImage(filepath)
                    auximg = sitk.GetArrayFromImage(vimg)
                else:
                    auximg += sitk.GetArrayFromImage(vimg)
                i += 1

    return sitk.GetImageFromArray(auximg)


def bbox3(img):
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

    return [[max(0, xmin-10), min(img.shape[0], xmax+10)],
            [max(0, ymin - 10), min(img.shape[1], ymax + 10)],
            [max(0, zmin - 10), min(img.shape[2], zmax + 10)]]
    # return sitk.GetImageFromArray(img[xmin-10:xmax+10, ymin-10:ymax+10, zmin-10:zmax+10])


# Create csv file of the files in dataPath and subfolders
def createCSV(dataPath):
    filelist = []
    for root, dirs, files in os.walk(dataPath):
        for name in sorted(files, key=str.lower):
            if name.endswith("_origSize_brainmaskLV_FIXED.nii.gz"):
                name = os.path.join(root, name)
                names = (name.replace("_origSize_brainmaskLV_FIXED.nii.gz",'.nii.gz'), name)
                filelist.append(names)
            else:
                continue

    with open(os.path.join(dataPath, 'files.csv'), 'w') as fp:
        writer = csv.writer(fp, delimiter=',')
        # writer.writerow(["your", "header", "foo"])  # write header
        writer.writerows(filelist)


def calculateBoundingBox(inpath, outpath):
    # Sum all masks (located in output folder), calculate the bounding box and crop the images.
    img = sumMasks(inpath)

    # position of the bounding box
    pos = bbox3(img)

    # crop all images in those positions
    prepFSF(inpath, outpath, pos, resize=True)


def preprocessPFC(inpath, outpath):
    # preprocess folder and subfolders and save them in "preprocessed"
    prepFSF(inpath, outpath, resamp=True, regist=True, normal=False)


def flipData(m, axis):
    """
    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Input array.
    axis : integer
        Axis in array, which entries are reversed.
    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).
    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.
    Examples
    --------

    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    True


    franco todo borrar ejemplo de como usar (flipear en el eje y)
    if not os.path.exists(os.path.split(outmask)[0]):
        os.makedirs(os.path.split(outmask)[0])

    img = nib.load(outmask)
    flipped_image = nib.Nifti1Image(flipData(img.get_data(), axis=1), affine=np.eye(4))
    nib.save(flipped_image, outmask)
    """
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]



def convertMasksStep1(masksPath, scansPath, outFolder):
    '''
       Given path of masks and ct scans, save in outFolder just both and not crappy masks
       Procedimiento: Esto solo agarra las mascaras que terminen en origSize_brainmaskLV_FIXED.nii.gz
       y se encuentren en masksPath y a partir de esto flipea en x e y (manteniendo spacing)
       y guarda en outFolder manteniendo estructura de carpetas.

       IMPORTANTE: Si en scansPath no está la CT que corresponde con la mascara, entonces no guarda
       la mascara en outFolder.
   '''

    # Folder that contains the files to be preprocessed
    print "\n-------------\nMasks folder name: ", masksPath
    print "CT scans folder name: ", scansPath

    # Get the files in dataPath folder and files in subfolders (except output folder)
    for root, dirs, files in os.walk(masksPath):
        for name in files:
            if name.endswith("origSize_brainmaskLV_FIXED.nii.gz"):
                filepath = os.path.join(root, name)
                print "File name: ", filepath, '\n'

                ctfile = os.path.join(scansPath, os.path.split(root)[1], os.path.split(root)[1] +
                                      name.replace('ct', '').replace('_origSize_brainmaskLV_FIXED.nii.gz',
                                                                     '_image.nii.gz'))
                # If the CT scan exists for that mask
                if not os.path.isfile(ctfile):
                    print "File "+ctfile+" not found."
                else:
                    outimg = os.path.join(outFolder, os.path.split(root)[1],
                                          os.path.split(root)[1] +
                                          name.replace('ct', '').replace('_origSize_brainmaskLV_FIXED.nii.gz',
                                                                         '_image.nii.gz'))

                    outmask = os.path.join(outFolder, os.path.split(root)[1],
                                           os.path.split(root)[1] +
                                           name.replace('ct', '').replace('_origSize_brainmaskLV_FIXED.nii.gz',
                                                                          '_head_mask.nii.gz'))


                    if not os.path.exists(os.path.split(outmask)[0]):
                        os.makedirs(os.path.split(outmask)[0])

                    img = nib.load(filepath)
                    flipped_image = nib.Nifti1Image(flipData(img.get_data(), axis=0), affine=img.affine)
                    nib.save(flipped_image, outmask)

                    img = nib.load(outmask)
                    flipped_image = nib.Nifti1Image(flipData(img.get_data(), axis=1), affine=img.affine)
                    nib.save(flipped_image, outmask)

                    # # I'll need to do this later, after flipping all the masks
                    # # Save both files in the output folder
                    # ctscan = sitk.ReadImage(ctfile)
                    # mask = sitk.ReadImage(filepath)
                    #
                    # if not os.path.exists(os.path.split(outimg)[0]):

                    # Save both files in the output folder
                    # ctscan = sitk.ReadImage(ctfile)
                    # mask = sitk.ReadImage(filepath)
                    #
                    # if not os.path.exists(os.path.split(outimg)[0]):
                    #     os.makedirs(os.path.split(outimg)[0])
                    #
                    # if not os.path.exists(os.path.split(outmask)[0]):
                    #     os.makedirs(os.path.split(outmask)[0])
                    #
                    # sitk.WriteImage(ctscan, outimg)
                    # sitk.WriteImage(mask, outmask)


def convertMasksStep2(masksPath, scansPath, outFolder):
    '''
       Given path of masks and ct scans, save in outFolder just both and not crappy masks
       Procedimiento: Recorrer la carpeta de máscaras masksPath y guardarlas en la correspondiente carpeta
                      de la salida. También buscar la
                      CT y guardarla en la misma carpeta.
   '''

    # Folder that contains the files to be preprocessed
    print "\n-------------\nMasks folder name: ", masksPath
    print "CT scans folder name: ", scansPath

    # Get the files in dataPath folder and files in subfolders (except output folder)
    for root, dirs, files in os.walk(masksPath):
        for name in files:
            filepath = os.path.join(root, name)
            print "File name: ", filepath, '\n'

            ctfile = os.path.join(scansPath, os.path.split(root)[1], name.replace('head_mask', 'image'))
            # If the CT scan exists for that mask
            if not os.path.isfile(ctfile):
                print "File "+ctfile+" not found."
            else:
                outimg = os.path.join(outFolder, os.path.split(root)[1], name.replace('head_mask', 'image'))

                outmask = os.path.join(outFolder, os.path.split(root)[1], name)

                # Save both files in the output folder
                ctscan = sitk.ReadImage(ctfile)
                mask = sitk.ReadImage(filepath)

                if not os.path.exists(os.path.split(outimg)[0]):
                    os.makedirs(os.path.split(outimg)[0])

                if not os.path.exists(os.path.split(outmask)[0]):
                    os.makedirs(os.path.split(outmask)[0])

                sitk.WriteImage(ctscan, outimg)
                sitk.WriteImage(mask, outmask)


if __name__ == "__main__":
    start_time = time.time()

    # createCSV("/home/fmatzkin/Code/Torch3DUnet/data/segundopaso")

    # Primero realizar el preprocesamiento de los datos, así el bounding box se calcula en
    # imagenes con caracteristicas similares
    # preprocessPFC("data", "primerpaso")

    # Calcular el bounding box y recortar las imagenes al tamaño hallado
    # calculateBoundingBox("data/primerpaso", "segundopaso")

    # convertMasksStep1("masksonly", "data", "cleandata")
    # convertMasksStep2("cleandata", "data", "ct_and_masks")

    print "Elapsed time was " + str(int(time.time() - start_time)) + " seconds."
