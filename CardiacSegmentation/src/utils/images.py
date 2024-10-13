"""
File: images_utilities.py
Author: Ariel Hernán Curiale and Lucca Dellazoppa
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

import numpy as np
import vtk
import vtk.util.numpy_support as numpy_support


def normalize_image(I):
    # Scale intensity to [0, 255]
    img_p = I.astype('float32')
    img_p -= img_p.min(axis=(0, 1))[np.newaxis, np.newaxis]
    img_p /= img_p.max(axis=(0, 1))[np.newaxis, np.newaxis]
    img_p *= 255
    img_p -= img_p.mean(axis=(0, 1))[np.newaxis, np.newaxis]
    img_p /= img_p.std(axis=(0, 1))[np.newaxis, np.newaxis] + 1e-7
    return img_p


def vtkImageDataFromArray(data, size, origin, spacing, direction):
    # data: 1D  numpy array
    vtk_data = numpy_support.numpy_to_vtk(data)
    # Creamos una vtkImageData
    vtkimg = vtk.vtkImageData()
    vtkimg.SetDimensions(size)
    vtkimg.SetSpacing(spacing)
    vtkimg.SetOrigin(origin)
    vtkimg.SetDirectionMatrix(direction)
    vtkimg.GetPointData().SetScalars(vtk_data)
    vtkimg.GetPointData().GetArray(0).SetName('ImageData')
    return vtkimg


def slicerNodeFromArray(array, name, refVol):
    import slicer
    # Get IJK To Ras Directions
    IJKToRasdirections = np.zeros((3, 3))
    refVol.GetIJKToRASDirections(IJKToRasdirections)
    # Convert to vtk
    # Numpy shape is (z,y,x)
    # VTK/ITK shape is (x,y,z)
    size = array.shape[::-1]
    origin = refVol.GetImageData().GetOrigin()
    spacing = refVol.GetImageData().GetSpacing()
    direction = refVol.GetImageData().GetDirectionMatrix()
    data = array.ravel()
    vtk_img = vtkImageDataFromArray(data, size, origin, spacing, direction)
    segVol = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    segVol.AddDefaultStorageNode()
    segVol.SetName(name)
    segVol.SetAndObserveImageData(vtk_img)

    # Slicer Node properties
    segVol.SetSpacing(refVol.GetSpacing())
    segVol.SetOrigin(refVol.GetOrigin())
    segVol.SetIJKToRASDirections(IJKToRasdirections)
    return segVol
