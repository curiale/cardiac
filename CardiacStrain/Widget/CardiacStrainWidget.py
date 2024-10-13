import importlib
from functools import partial

import qt
import slicer
import vtk
import numpy as np
import SimpleITK as sitk
from Logic.CardiacStrainLogic import CardiacStrainLogic
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from slicer.util import VTKObservationMixin

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    pass  # See Installation UI


class CardiacStrainWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._deps = {
            "tensorflow": "2.7.0", 
            "matplotlib": None, 
            "scikit-image": None, 
            "nibabel": None,
        }
        self._charts = [{}, {}, {}, {}]
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        if not self._checkImports():
            self.initializeInstallationUi()

        else:
            self.initializeUi()
            self.initializeLogic()
            self.initializeObservers()
            self.initializeConnections()
            self.initializeParameterNode()

    def cleanup(self):
        self.removeObservers()

    def enter(self):
        if self._checkImports():
            self.initializeParameterNode()

    def exit(self):
        if self._checkImports():
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )

    def onSceneStartClose(self, caller, event):
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        if self.parent.isEntered:
            self.initializeParameterNode()

    def setParameterNode(self, inputParameterNode):
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode,
            )
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        self._updatingGUIFromParameterNode = True

        imgNodeReference = self._parameterNode.GetNodeReference("ImgSequence")
        segNodeReference = self._parameterNode.GetNodeReference("SegSequence")

        self.ui.imgSelector.setCurrentNode(imgNodeReference)
        self.ui.segSelector.setCurrentNode(segNodeReference)

        self.ui.runStrainButton.enabled = imgNodeReference and segNodeReference

        strainIsReady = self._parameterNode.GetParameter("hasStrain") == "True"
        self.ui.strainReportButton.enabled = strainIsReady
        self.ui.visualizationTabs.enabled = strainIsReady
        self.ui.saveStrainButton.enabled = strainIsReady
        self.ui.saveMotionButton.enabled = strainIsReady
        self.ui.saveAHAButton.enabled = strainIsReady

        self.ui.statusLabel.setText(self._parameterNode.GetParameter("Status"))

        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()

        self._parameterNode.SetNodeReferenceID(
            "ImgSequence", self.ui.imgSelector.currentNodeID
        )
        self._parameterNode.SetNodeReferenceID(
            "SegSequence", self.ui.segSelector.currentNodeID
        )
        self._parameterNode.SetParameter("MyoLabel", self.ui.myoLabel.text)
        self._parameterNode.SetParameter("RVLabel", self.ui.rvLabel.text)
        self._parameterNode.SetParameter("Status", self.ui.statusLabel.text)
        self._parameterNode.EndModify(wasModified)

    def onLoadImg(self):
        filenames = qt.QFileDialog.getOpenFileNames()
        if len(filenames) == 0:
            slicer.util.messageBox("Select at least one file.")
            return

        seqName = qt.QInputDialog.getText(qt.QWidget(), "", "Sequence name:")
        if not seqName:
            slicer.util.messageBox("Can't create an empty name sequence.")
            return

        self.logic.loadImages(filenames, seqName)
        self.ui.statusLabel.setText("Load segmentations.")

    def onLoadSeg(self):
        seq = self.ui.imgSelector.currentNode()
        if seq is None:
            slicer.util.messageBox("You should select a sequence first.")
            return

        filenames = qt.QFileDialog.getOpenFileNames()
        if len(filenames) == 1:
            self.logic.loadSegmentation(filenames, seq.GetName())
            self.ui.statusLabel.setText("Run strain analysis.")
        elif len(filenames) == seq.GetNumberOfDataNodes():
            raise NotImplementedError
        else:
            slicer.util.messageBox("Wrong quantity of segmentation files")
            return

    def onRunStrain(self):
        self.logic.runStrainPipeline()
        self.setVisualizations()

    def onReport(self):
        filename = qt.QFileDialog.getSaveFileName()
        if not filename:
            slicer.util.messageBox("Select a file.")
            return
        seqName = self.ui.imgSelector.currentNode().GetName()
        directions = ["Radial", "Circumferential", "Longitudinal"]
        kinds = ["Strain", "Strain Rate"]
        zones = ["Global", "Apical", "Mid", "Basal"]

        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle(f"Strain Analysis report for {seqName}\n\n", fontsize=20)
        for j, kind in enumerate(kinds):
            axs[0, j].set_title(kind, loc="left", fontsize=15)
            axs[2, j].set_xlabel("Time [frames]", fontsize=12)
            for i, direction in enumerate(directions):
                axs[i, 0].set_ylabel(direction, fontsize=12)
                axs[i, j].grid()
                for z in zones:
                    s = self.logic.getStrainSeries(direction, kind, z)
                    axs[i, j].plot(s, label=z, lw=3 if z == "Global" else 0.5)
                axs[i, j].legend(loc="upper left")
        fig.tight_layout()

        plt.savefig(filename + ".pdf")

    def onSaveAha(self):
        aha = self.logic.getCalculatedResult("aha")
        if not aha:
            slicer.util.messageBox("Strain needs to be calculated first")
            return
        filename = qt.QFileDialog.getSaveFileName()
        if not filename:
            slicer.util.messageBox("Select a file.")
            return
        np.save(filename + ".aha", aha[0])
        np.save(filename + ".lc", aha[1])

    def onSaveMotion(self):
        motion = self.logic.getCalculatedResult("motion")
        if not motion:
            slicer.util.messageBox("Strain needs to be calculated first")
            return
        filename = qt.QFileDialog.getSaveFileName()
        if not filename:
            slicer.util.messageBox("Select a file.")
            return
        sitk.WriteImage(sitk.JoinSeries(motion), filename + ".mhd")

    def onSaveStrain(self):
        strain = self.logic.getCalculatedResult("strain")
        if not strain:
            slicer.util.messageBox("Strain needs to be calculated first")
            return
        filename = qt.QFileDialog.getSaveFileName()
        if not filename:
            slicer.util.messageBox("Select a file.")
            return
        np.save(filename, np.asarray(strain))

    def setVisualizations(self):
        lm = slicer.app.layoutManager()
        if self.ui.visualizationTabs.currentIndex == 0:
            lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpPlotView)
            cvns = slicer.mrmlScene.GetNodesByClass("vtkMRMLPlotViewNode")
            cvns.UnRegister(None)
            cvns.InitTraversal()
            cvn = cvns.GetNextItemAsObject()
            kind = self.ui.singlePlotTypeSelector.currentText
            direction = self.ui.singlePlotDirectionSelector.currentText
            region = self.ui.singlePlotRegionSelector.currentText
            self._plotStrain(kind, direction, region, cvn, 0)

        if self.ui.visualizationTabs.currentIndex == 1:
            lm.setLayout(
                slicer.vtkMRMLLayoutNode.SlicerLayoutThreeOverThreePlotView
            )
            cvns = slicer.mrmlScene.GetNodesByClass("vtkMRMLPlotViewNode")
            cvns.UnRegister(None)
            cvns.InitTraversal()
            cvn = cvns.GetNextItemAsObject()
            kind = self.ui.triplePlotTypeSelector1.currentText
            direction = self.ui.triplePlotDirectionSelector1.currentText
            region = self.ui.triplePlotRegionSelector1.currentText
            self._plotStrain(kind, direction, region, cvn, 1)

            cvn = cvns.GetNextItemAsObject()
            kind = self.ui.triplePlotTypeSelector2.currentText
            direction = self.ui.triplePlotDirectionSelector2.currentText
            region = self.ui.triplePlotRegionSelector2.currentText
            self._plotStrain(kind, direction, region, cvn, 2)

            cvn = cvns.GetNextItemAsObject()
            kind = self.ui.triplePlotTypeSelector3.currentText
            direction = self.ui.triplePlotDirectionSelector3.currentText
            region = self.ui.triplePlotRegionSelector3.currentText
            self._plotStrain(kind, direction, region, cvn, 3)

    def initializeInstallationUi(self):
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Installation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        self.ui.installAll.connect(
            "clicked(bool)", partial(self._installDep, "All")
        )
        for name, version in self._deps.items():
            button = getattr(self.ui, name)
            if importlib.util.find_spec(name):
                button.enabled = False
                button.setText("Already installed")
            else:
                dep = name + "==" + version if version else name
                button.connect("clicked(bool)", partial(self._installDep, dep))
        self.ui.restartSlicer.connect("clicked(bool)", slicer.util.restart)

    def initializeUi(self):
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/CardiacStrain.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

    def initializeObservers(self):
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.StartCloseEvent,
            self.onSceneStartClose,
        )
        self.addObserver(
            slicer.mrmlScene,
            slicer.mrmlScene.EndCloseEvent,
            self.onSceneEndClose,
        )

    def initializeLogic(self):
        self.logic = CardiacStrainLogic()
        self.logic.modelPath = self.resourcePath("Models/motion.h5")

    def initializeConnections(self):
        self.ui.imgSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.updateParameterNodeFromGUI,
        )
        self.ui.segSelector.connect(
            "currentNodeChanged(vtkMRMLNode*)",
            self.updateParameterNodeFromGUI,
        )
        self.ui.myoLabel.connect(
            "textChanged(QString*)",
            self.updateParameterNodeFromGUI,
        )
        self.ui.rvLabel.connect(
            "textChanged(QString*)",
            self.updateParameterNodeFromGUI,
        )
        self.ui.loadImgButton.connect("clicked(bool)", self.onLoadImg)
        self.ui.loadSegButton.connect("clicked(bool)", self.onLoadSeg)
        self.ui.runStrainButton.connect("clicked(bool)", self.onRunStrain)
        self.ui.saveStrainButton.connect("clicked(bool)", self.onSaveStrain)
        self.ui.saveMotionButton.connect("clicked(bool)", self.onSaveMotion)
        self.ui.saveAHAButton.connect("clicked(bool)", self.onSaveAha)
        self.ui.strainReportButton.connect("clicked(bool)", self.onReport)
        self.ui.visualizationTabs.connect(
            "currentChanged(int)", self.setVisualizations
        )
        for child in self.ui.visualizationTabs.findChildren(qt.QComboBox):
            child.connect("currentIndexChanged(int)", self.setVisualizations)

    def initializeParameterNode(self):
        self.setParameterNode(self.logic.getParameterNode())

    def _plotBullseye(self, kind, direction, region, cvn, chartIdx):
        pass

    def _plotStrain(self, kind, direction, region, cvn, chartIdx):
        series = self.logic.getStrainSeries(direction, kind, region)
        cn = slicer.util.plot(
            series,
            show=False,
            nodes=self._charts[chartIdx],
        )
        cn.SetTitle(f"{kind} - {region}")
        cn.GetNthPlotSeriesNode(0).SetName(f"{direction}")
        cn.SetXAxisTitle("Time [frames]")
        cvn.SetPlotChartNodeID(cn.GetID())

    def _checkImports(self):
        for name, _ in self._deps.items():
            if importlib.util.find_spec(name) is None:
                return False
        return True

    def _installDep(self, dep="All", event=None, caller=None):
        if dep == "All":
            deps = [f"{n}=={v}" if v else n for n, v in self._deps.items()]
            self.ui.status.text = "Installing and restarting"
            slicer.util.pip_install(deps)
            slicer.util.restart()
        else:
            self.ui.status.text = f"Installing {dep}"
            slicer.util.pip_install(dep)
            self.ui.status.text = f"{dep} installed"
