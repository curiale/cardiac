import os

import numpy as np
import SimpleITK as sitk
import sitkUtils
import slicer
from scipy.ndimage import (
    binary_fill_holes,
    center_of_mass,
    convolve1d,
    gaussian_filter,
)
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from slicer.util import VTKObservationMixin

try:
    import tensorflow as tf
    from tensorflow import keras

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

except ImportError:
    pass  # See Installation UI


class CardiacStrainLogic(ScriptedLoadableModuleLogic, VTKObservationMixin):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        VTKObservationMixin.__init__(self)
        self._cache = {}

    def setDefaultParameters(self, parameterNode):
        if not parameterNode.GetParameter("Status"):
            parameterNode.SetParameter("Status", "Load images to start.")
        if not parameterNode.GetParameter("MyoLabel"):
            parameterNode.SetParameter("MyoLabel", "2")
        if not parameterNode.GetParameter("RVLabel"):
            parameterNode.SetParameter("RVLabel", "1")
        if not parameterNode.GetParameter("hasStrain"):
            parameterNode.SetParameter("hasStrain", "False")

    def loadImages(self, filenames, seqName):
        seqNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", seqName)
        seqNodeId = seqNode.GetID()

        for time, filename in enumerate(filenames):
            node = slicer.util.loadVolume(
                filename, properties={"name": f"{seqName}_{time}"}
            )
            seqNode.SetDataNodeAtValue(node, str(time))

        seqBrowserNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLSequenceBrowserNode", seqName + "_browser"
        )
        seqBrowserNode.SetAndObserveMasterSequenceNodeID(seqNode.GetID())

        slicer.modules.sequences.logic().UpdateAllProxyNodes()
        slicer.app.processEvents()
        slicer.modules.sequences.setToolBarActiveBrowserNode(seqBrowserNode)
        sequenceProxyNode = seqBrowserNode.GetProxyNode(seqNode)
        slicer.util.setSliceViewerLayers(background=sequenceProxyNode.GetID())

        self.getParameterNode().SetNodeReferenceID("ImgSequence", seqNodeId)

    def loadSegmentation(self, filenames, sequenceName):
        seg = sitk.ReadImage(filenames[0])
        segNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", sequenceName + "_segED"
        )
        segNodeId = segNode.GetID()
        sitkUtils.PushVolumeToSlicer(seg, segNode)

        slicer.util.setSliceViewerLayers(label=segNodeId)
        self.getParameterNode().SetNodeReferenceID("SegSequence", segNodeId)

    def runStrainPipeline(self):
        sequence = self.getParameterNode().GetNodeReference("ImgSequence")
        segmentation = self.getParameterNode().GetNodeReference("SegSequence")
        myoLabel = int(self.getParameterNode().GetParameter("MyoLabel"))
        rvLabel = int(self.getParameterNode().GetParameter("RVLabel"))

        seqId = sequence.GetID()
        if ("strain", seqId) not in self._cache:
            self._set_status("Measuring Motion")
            motion = self.getMotion(sequence, segmentation, rvLabel, myoLabel)
            self._cache[("motion", seqId)] = motion

            self._set_status("Analyzing strain")
            localCoords = self.getLocalCoords(segmentation, myoLabel, rvLabel)
            ahaModel = self.getAhaModel(segmentation, myoLabel, rvLabel)
            self._cache[("aha", seqId)] = ahaModel, localCoords

            strain = self.getStrain(motion, ahaModel, localCoords)
            self._cache[("strain", seqId)] = strain

            self._set_status("Ready for plotting")
            self.getParameterNode().SetParameter("hasStrain", "True")
        return self._cache[("strain", seqId)]

    def getMotion(self, sequence, segmentation, rvLabel, myoLabel):
        if "model" not in self._cache:
            self._cache["model"] = keras.models.load_model(self.modelPath)
        motionModel = self._cache["model"]

        segImg = self._resample(sitkUtils.PullVolumeFromSlicer(segmentation))
        seqImgs = []
        for t in range(sequence.GetNumberOfDataNodes()):
            seqImg = sitkUtils.PullVolumeFromSlicer(sequence.GetNthDataNode(t))
            seqImgs.append(self._resample(seqImg))
        seqImgs = sitk.JoinSeries(seqImgs)
        M = sitk.GetArrayFromImage(segImg).transpose((2, 1, 0))
        V = sitk.GetArrayFromImage(seqImgs).transpose((3, 2, 1, 0))

        center = center_of_mass(M == myoLabel)

        V = self._cropROI(V, center, [128, 128])
        V = self._getOrientation(V, M, rvLabel)
        V = self._normalize(V)

        motion = []
        for t in range(sequence.GetNumberOfDataNodes()):
            V_0 = V[..., 0][None, ..., None]
            V_t = V[..., t][None, ..., None]
            df = motionModel([V_0, V_t]).numpy()
            df = gaussian_filter(df, sigma=(0, 2, 2, 0, 0)).squeeze()
            df = df.transpose((2, 1, 0, 3))
            motion.append(sitk.GetImageFromArray(df))
        return motion

    def getLocalCoords(self, segmentation, myoLabel, rvLabel):
        segImg = sitkUtils.PullVolumeFromSlicer(segmentation)
        segImg = self._resample(segImg)
        lvCenter = self._lvCenter(segmentation, myoLabel, rvLabel)
        segArray = sitk.GetArrayFromImage(segImg)
        myo = segImg == myoLabel
        myoDilated = sitk.GetArrayFromImage(sitk.BinaryDilate(myo, [5, 5, 0]))
        myoTransp = myoDilated.transpose((2, 1, 0))
        center = center_of_mass(myoTransp)
        myo = self._cropROI(myoTransp, center, [128, 128]).transpose((2, 1, 0))
        maxRv = np.argmax((segArray == rvLabel).sum(axis=(1, 2)))
        isInverted = maxRv < segArray.shape[0] // 2
        localCoords = np.zeros(myo.shape + (3, 3), dtype=float)
        z, y, x = np.where(myo)
        localCoords[z, y, x, :, 0] = [0, 0, -1 if isInverted else 1]  # c_l
        for ii, z_i in enumerate(lvCenter[:, 0]):
            idz = z == z_i
            yx = np.vstack([y[idz], x[idz]]).T - lvCenter[ii, 1:]
            yx[np.abs(yx).sum(axis=1) == 0] += 0.1
            c_r = -(yx.T / np.linalg.norm(yx, axis=1)).T
            c_c = np.vstack([c_r[:, 1], -c_r[:, 0]]).T
            localCoords[z[idz], y[idz], x[idz], :2, 1] = c_c[:, ::-1]
            localCoords[z[idz], y[idz], x[idz], :2, 2] = c_r[:, ::-1]
        return localCoords

    def getAhaModel(self, segmentation, myoLabel, rvLabel):
        segImg = sitkUtils.PullVolumeFromSlicer(segmentation)
        rvPoint = self._rvPoint(segmentation, myoLabel, rvLabel)
        lvCenter = self._lvCenter(segmentation, myoLabel, rvLabel, True)
        segImg = self._resample(segImg)
        myo = segImg == myoLabel
        myoDilated = sitk.GetArrayFromImage(sitk.BinaryDilate(myo, [5, 5, 0]))
        myoTransp = myoDilated.transpose((2, 1, 0))
        center = center_of_mass(myoTransp)
        myo = self._cropROI(myoTransp, center, [128, 128]).transpose((2, 1, 0))
        ahaLv = np.zeros_like(myo, dtype=np.uint8)
        self._add16segments(myo, lvCenter, rvPoint, ahaLv)
        self._add17segment(myo, ahaLv)
        return ahaLv

    def getStrain(self, motion, ahaModel, localCoords):
        # Accumulated strains: 17 indexes for AHA model, last index is global.
        iec_accum = np.zeros((18, len(motion) + 1))
        iel_accum = np.zeros((18, len(motion) + 1))
        ier_accum = np.zeros((18, len(motion) + 1))

        for t in range(len(motion)):
            df = sitk.GetArrayFromImage(motion[t])

            df = [df[..., 0], df[..., 1], df[..., 2] * 4]

            iec = np.zeros(ahaModel.shape, dtype=float)
            ier = np.zeros(ahaModel.shape, dtype=float)
            iel = np.zeros(ahaModel.shape, dtype=float)

            u, v, w = df[0], df[1], df[2]
            u_z, u_y, u_x = np.gradient(u)
            v_z, v_y, v_x = np.gradient(v)
            w_z, w_y, w_x = np.gradient(w)

            e_xx, e_xy, e_xz = u_x, 0.5 * (u_y + v_x), 0.5 * (u_z + w_x)
            e_yx, e_yy, e_yz = 0.5 * (u_y + v_x), v_y, 0.5 * (v_z + w_y)
            e_zx, e_zy, e_zz = 0.5 * (u_z + w_x), 0.5 * (v_z + w_y), w_z

            for r, c, j in zip(*np.where(ahaModel > 0)):
                c_l = localCoords[..., 0][r, c, j, :]  # x,y,z
                c_c = localCoords[..., 1][r, c, j, :]  # x,y,z
                c_r = localCoords[..., 2][r, c, j, :]  # x,y,z

                inf_e = np.array(
                    [
                        [e_xx[r, c, j], e_xy[r, c, j], e_xz[r, c, j]],
                        [e_yx[r, c, j], e_yy[r, c, j], e_yz[r, c, j]],
                        [e_zx[r, c, j], e_zy[r, c, j], e_zz[r, c, j]],
                    ]
                )
                iec[r, c, j] = np.dot(np.dot(c_c, inf_e), c_c)
                ier[r, c, j] = np.dot(np.dot(c_r, inf_e), c_r)
                iel[r, c, j] = np.dot(np.dot(c_l, inf_e), c_l)

            for j in range(17):
                rr, cc, jj = np.where(ahaModel == j + 1)
                iec_accum[j, t] = iec[rr, cc, jj].mean() * 100
                ier_accum[j, t] = ier[rr, cc, jj].mean() * 100
                iel_accum[j, t] = iel[rr, cc, jj].mean() * 100
            iec_accum[-1, t] += iec[ahaModel > 0].mean() * 100
            ier_accum[-1, t] += ier[ahaModel > 0].mean() * 100
            iel_accum[-1, t] += iel[ahaModel > 0].mean() * 100

        iec_accum = convolve1d(iec_accum, np.ones(4) / 4, axis=1)
        ier_accum = convolve1d(ier_accum, np.ones(4) / 4, axis=1)
        iel_accum = convolve1d(iel_accum, np.ones(4) / 4, axis=1)
        iec_accum[:, 0] = ier_accum[:, 0] = iel_accum[:, 0] = 0

        iecn = iec_accum[:, -1]
        iern = ier_accum[:, -1]
        ieln = iel_accum[:, -1]

        mc = iecn / len(motion)
        mr = iern / len(motion)
        ml = ieln / len(motion)

        iec_accum = iec_accum - mc[:, np.newaxis] * np.arange(len(motion) + 1)
        ier_accum = ier_accum - mr[:, np.newaxis] * np.arange(len(motion) + 1)
        iel_accum = iel_accum - ml[:, np.newaxis] * np.arange(len(motion) + 1)

        return ier_accum, iec_accum, iel_accum

    def getStrainSeries(self, direction, kind, zone):
        directionIdxs = {"R": 0, "C": 1, "L": 2}
        zones = {"G": (17, 17), "B": (0, 6), "M": (7, 13), "A": (13, 17)}
        seqId = self.getParameterNode().GetNodeReference("ImgSequence").GetID()
        strain = self._cache[("strain", seqId)]
        strain = strain[directionIdxs[direction[0]]]
        strain = strain[zones[zone[0]][0] : zones[zone[0]][1] + 1].mean(axis=0)
        return np.gradient(strain, axis=0) if "Rate" in kind else strain

    def getCalculatedResult(self, key):
        seqId = self.getParameterNode().GetNodeReference("ImgSequence").GetID()
        return self._cache.get((key, seqId), None)

    def _set_status(self, text):
        self.getParameterNode().SetParameter("Status", text)
        slicer.app.processEvents()

    def _resample(self, img, outputSize=[256, 256, 16]):
        dim = img.GetDimension()
        imgSize = np.asarray(img.GetSize())
        imgSpacing = np.asarray(img.GetSpacing())
        ref = sitk.Image(outputSize, img.GetPixelIDValue())
        ref.SetOrigin(img.GetOrigin())
        ref.SetDirection(img.GetDirection())
        ref.SetSpacing(imgSize / np.asarray(outputSize) * imgSpacing)
        identity = sitk.Transform(dim, sitk.sitkIdentity)
        return sitk.Resample(img, ref, identity, sitk.sitkNearestNeighbor)

    def _cropROI(self, img, center, size):
        sx, sy = size
        cx, cy, _ = np.asarray(center).astype(int)
        return img[cx - sx // 2 : cx + sx // 2, cy - sy // 2 : cy + sy // 2, ...]

    def _normalize(self, V, axis=(0, 1, 2)):
        mu = V.mean(axis=axis, keepdims=True)
        sd = V.std(axis=axis, keepdims=True)
        return (V - mu) / (sd + 1e-8)

    def _getOrientation(self, V, M, rvLabel):
        maxRv = np.argmax((M == rvLabel).sum(axis=(0, 1)))
        if maxRv > M.shape[2] // 2:
            return V[:, :, ::-1]
        return V

    def _rvPoint(self, segmentation, myoLabel, rvLabel):
        segImg = sitkUtils.PullVolumeFromSlicer(segmentation)
        segImg = self._resample(segImg)
        transposedSeg = sitk.GetArrayFromImage(segImg).transpose((2, 1, 0))
        lvCenter = center_of_mass(transposedSeg == myoLabel)
        croppedSeg = self._cropROI(transposedSeg, lvCenter, [128, 128])
        return center_of_mass(croppedSeg == rvLabel)[::-1]

    def _lvCenter(self, segmentation, myoLabel, rvLabel, by_slice=False):
        segImg = sitkUtils.PullVolumeFromSlicer(segmentation)
        segArray = sitk.GetArrayFromImage(segImg)
        myo = segArray == myoLabel
        myoTransp = myo.transpose((2, 1, 0))
        center = center_of_mass(myoTransp)
        myo = self._cropROI(myoTransp, center, [128, 128]).transpose((2, 1, 0))
        maxRv = np.argmax((segArray == rvLabel).sum(axis=(1, 2)))
        isInverted = maxRv > segArray.shape[0] // 2
        lvCenter = []
        mid = int(round(myo.shape[0] / 2))
        cm = center_of_mass(myo[mid, ...])
        for z in range(myo.shape[0] - 1, -1, -1):
            npixels = myo[z, ...].sum()
            if npixels > 0:
                if by_slice:
                    cm = center_of_mass(myo[z, ...])
                lvCenter.append([z, *cm])
        lvCenter = np.array(lvCenter)
        return lvCenter[::-1, :] if isInverted else lvCenter

    def _add16segments(self, myo, lvCenter, rvPoint, ahaLv):
        rv2lv = lvCenter[0][1:] - rvPoint[1:]
        z, y, x = np.where(myo)

        e1 = rv2lv / np.linalg.norm(rv2lv)
        e2 = np.array([e1[1], -e1[0]])
        tmatrix = np.matrix([e1, e2]).T

        zBase = int(round(lvCenter.shape[0] * 0.35))
        zMid = int(round(lvCenter.shape[0] * 0.70))

        anglesBaseMid = [
            (-120, -60),
            (-180, -120),
            (120, 180),
            (60, 120),
            (0, 60),
            (-60, 0),
        ]
        anglesAppex = [(-135, -45), (135, 180), (45, 135), (-45, 45)]

        for idz, z_i in enumerate(lvCenter[:, 0]):
            idx = z == z_i
            z_idx, y_i, x_i = z[idx], y[idx], x[idx]
            yx_i = (np.vstack([y_i, x_i]).T - lvCenter[idz, 1:]).T
            YX_i = np.array(tmatrix * yx_i).T
            ang = np.arctan2(YX_i[:, 1], YX_i[:, 0]) * 180 / np.pi
            if idz < zBase:
                ahaSegs, anglesLimits = range(1, 7), anglesBaseMid
            elif zBase < idz and idz < zMid:
                ahaSegs, anglesLimits = range(7, 13), anglesBaseMid
            else:
                ahaSegs, anglesLimits = range(13, 17), anglesAppex

            for ahaSeg, angleLimits in zip(ahaSegs, anglesLimits):
                ids = (angleLimits[0] <= ang) & (ang <= angleLimits[1])
                ahaLv[z_idx[ids], y_i[ids], x_i[ids]] = ahaSeg
                if angleLimits[0] == 135:  # seg 14 workaround
                    ids = (-180 <= ang) & (ang <= -135)
                    ahaLv[z_idx[ids], y_i[ids], x_i[ids]] = ahaSeg

    def _add17segment(self, myo, ahaLv):
        myoFilled = []
        for idx in range(myo.shape[0]):
            myoFilled.append(binary_fill_holes(myo[idx]))
        myoFilled = np.asarray(myoFilled).astype("uint8")
        epi = sitk.GetImageFromArray(myoFilled)
        myoImg = sitk.GetImageFromArray(myo.astype("int8"))
        endo = sitk.Cast(epi, sitk.sitkInt8) - myoImg
        Iendo = sitk.GetArrayFromImage(endo)
        z = np.any(Iendo, axis=(1, 2))
        zmin_endo, _ = np.where(z)[0][[0, -1]]
        Iendo_minz = Iendo[zmin_endo]
        z = np.any(myo, axis=(1, 2))
        zmin_myo, _ = np.where(z)[0][[0, -1]]
        for z in range(zmin_myo, zmin_endo):
            ahaLv[z] = 17 * (Iendo_minz > 0)
