"""
File: CardiacSegmentation.py
Author: Ariel Hernán Curiale and Lucca Dellazopa
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
    Module for Myocardial segmentation and Volumetric cuantification.
    To disable Test and Reload Widget element go to Settings -> Developer and
    unclick the option Enable developer mode.
"""

# Slicer modules
import os
import sys
import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleWidget
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleTest
from slicer.util import VTKObservationMixin


# Extra modules
import numpy as np
from scipy import ndimage
import time
from pathlib import Path
# Hack to avoid problems with embeded python
# https://bugs.python.org/issue32573
if not hasattr(sys, 'argv'):
    sys.argv = ['']

# Module libraries
from src.settings import CENTER_MODEL_PATH, SEG_MODEL_PATH, LOGIC_IC
from src.settings import CENTER_MODEL_INPUT_SHAPE, SEG_MODEL_INPUT_SHAPE
from src.settings import LABEL_TISSUES
from src.utils import measures, images

from src.dependencies import Dependencies

dependencies = Dependencies()

if dependencies.check():
    # All dependencies installed
    from skimage import transform

    # Tensorflow imports
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from src.models.custom_objects import c_o as custom_objects_imported
    from src.models import architecture
    # Check if there is available GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Set cardiac base path
base_path = Path(
    os.path.realpath(__file__).split(
        os.path.basename(os.path.realpath(__file__)))[0])


#
# CardiacSegmentation Module
#
class CardiacSegmentation(ScriptedLoadableModule, VTKObservationMixin):
    """
    CardiacSegmentation Module class. This class define the segmentation module
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.parent.title = 'CardIAc Segmentation'
        self.parent.categories = ['CardIAc']
        self.parent.dependencies = []
        self.parent.contributors = ['Lucca Dellazoppa and Ariel H. Curiale']
        self.parent.helpText = '''
    Module for biventricular automatic segmentation and cardiac quantification
    in MRI.  For details and instructions, see
    https://gitlab.com/Curiale/cardiac.'''
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = '''
    This module is the reult of the work related to the final prject of the
    student Lucca Dellazoppa directed by PhD. Ariel H. Curiale. Now is
    maintained by Ariel H. Curiale.'''


#
# CardiacSegmentationWidget
#


class CardiacSegmentationWidget(ScriptedLoadableModuleWidget):
    """
    This class define the GUI for the cardIAc segmentation module
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self._dep = dependencies

    def setup(self):
        """
        Method to set up the complete GUI Widget.
        """
        # Set scene initial conditions
        # self.onGetOpacity() # TODO
        self.setNNModelFile()

        # Set up usual GUI in case dependencies are installed
        if not self._dep.check():
            self.startInstallationLayout()

        else:
            # Set up Widget's general class
            ScriptedLoadableModuleWidget.setup(self)
            self.logic = CardiacSegmentationLogic(self)

            # Define horizontal line to use in GUI
            class QHLine(qt.QFrame):

                def __init__(self):
                    super(QHLine, self).__init__()
                    self.setFrameShape(qt.QFrame.HLine)
                    self.setFrameShadow(qt.QFrame.Sunken)

            # Starts GUI
            self.startCollapsibleButtons()
            self.startLayouts()

            self.startSequenceSection()
            self.startSegmentationSection(QHLine)
            self.startQuantificationSection(QHLine)

            # Add vertical spacer
            self.layout.addStretch(1)

    def startInstallationLayout(self):
        """
        Method that starts different widget GUI in case 'skimage' and
        'tensorflow' are not installed in slicer's python bin.
        """

        self.installationCollapsibleButton = ctk.ctkCollapsibleButton()
        self.installationCollapsibleButton.text = 'Install dependencies'
        self.installationCollapsibleButton.collapsed = False

        self.layout.addWidget(self.installationCollapsibleButton)

        self.installationFormLayout = qt.QFormLayout(
            self.installationCollapsibleButton)

        self.installDependenciesText = qt.QLabel()

        msg = 'To use this module, you need to install its dependencies:\n'

        ldep = self._dep.missing_libraries
        for k in ldep[:-1]:
            msg += '\t\t%s\n' % k
        msg += '\t\t%s' % ldep[-1]

        self.installDependenciesText.setText(msg)
        self.installDependenciesText.setAlignment(qt.Qt.AlignCenter)
        self.installationFormLayout.addWidget(self.installDependenciesText)

        # Try to install dependencies automatically
        self.installDependenciesButton = qt.QPushButton('Install dependencies')
        self.installDependenciesButton.enabled = True
        self.installationFormLayout.addWidget(self.installDependenciesButton)

        self.installDependenciesButton.connect('clicked(bool)',
                                               self.onInstallDependencies)
        self.installLayoutStatus = qt.QLabel('')
        self.installationFormLayout.addWidget(self.installLayoutStatus)

    def onInstallDependencies(self):
        self._dep.install_libraries()
        lib_msg = ''
        for k in self._dep:
            if self._dep.libraries[k]['installed']:
                lib_msg += '%s: installed\n' % k
            else:
                lib_msg += '%s: not installed\n' % k
            self.installLayoutStatus.setText(lib_msg)

        msg = 'Installation of dependencies done. \n'
        if self._dep.installed:
            self.succesMessage(msg + 'All the libraries were installed and '
                               '3DSlicer will restart to update changes.')
            slicer.app.restart()
        else:
            self.succesMessage(
                msg + 'Cannot instal one or more libraries. '
                'Try to install them manually inside the Python CLI: '
                '"slicer.util.pip_install("tensorflow")"')

    def startCollapsibleButtons(self):
        """
        Method to start up widget collapsible buttons of the three sections:
        'sequences', 'segmentation' and 'quantification'
        """
        self.seqCollButton = ctk.ctkCollapsibleButton()
        self.seqCollButton.text = 'Create Sequence'
        self.layout.addWidget(self.seqCollButton)

        self.segmCollButton = ctk.ctkCollapsibleButton()
        self.segmCollButton.text = 'Segmentation'
        self.layout.addWidget(self.segmCollButton)

        self.visCollButton = ctk.ctkCollapsibleButton()
        self.visCollButton.text = 'Cardiac Quantification'
        self.visCollButton.collapsed = True
        self.layout.addWidget(self.visCollButton)

    def startLayouts(self):
        """
        Set up GUI's Layouts for the three sections of the module.
        """
        self.sequenceLayout = qt.QGridLayout(self.seqCollButton)
        self.segmentacionLayout = qt.QGridLayout(self.segmCollButton)
        self.visLayout = qt.QGridLayout(self.visCollButton)
        self.visLayout.setColumnMinimumWidth(2, 2)

    def startSequenceSection(self):
        """
        Set up GUI's Layout for 'Sequence' section
        """

        self.seqFromToLabel = qt.QLabel('Sequence from / to:')
        self.sequenceLayout.addWidget(self.seqFromToLabel, 0, 0, 1, 1)

        self.seqFromList = slicer.qMRMLNodeComboBox()
        self.seqFromList.setToolTip('First volume of the sequence')
        self.seqFromList.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.seqFromList.selectNodeUponCreation = True
        self.seqFromList.addEnabled = False
        self.seqFromList.removeEnabled = False
        self.seqFromList.noneEnabled = True
        self.seqFromList.showHidden = True
        self.seqFromList.showChildNodeTypes = False
        self.seqFromList.setMRMLScene(slicer.mrmlScene)
        self.seqFromList.setToolTip('First image of the sequence.')
        self.sequenceLayout.addWidget(self.seqFromList, 0, 1, 1, 1)

        self.seqToList = slicer.qMRMLNodeComboBox()
        self.seqToList.setToolTip('Last volume of the sequence')
        self.seqToList.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.seqToList.selectNodeUponCreation = True
        self.seqToList.addEnabled = False
        self.seqToList.removeEnabled = False
        self.seqToList.noneEnabled = True
        self.seqToList.showHidden = True
        self.seqToList.showChildNodeTypes = False
        self.seqToList.setMRMLScene(slicer.mrmlScene)
        self.seqToList.setToolTip('Last image of the sequence.')
        self.sequenceLayout.addWidget(self.seqToList, 0, 2, 1, 1)

        self.seqCreateLabel = qt.QLabel('Sequence options: ')
        self.sequenceLayout.addWidget(self.seqCreateLabel, 1, 0, 1, 1)

        self.seqCreateButton = qt.QPushButton('Create sequence')
        self.seqCreateButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/createSeq.png'))
        self.seqCreateButton.setToolTip(
            'Create a new sequence from the first '
            'volume selected to the last volume selected.')
        self.sequenceLayout.addWidget(self.seqCreateButton, 1, 1, 1, 1)

        # TODO ---> testing
        self.seqDeleteButton = qt.QPushButton('Delete Sequences')
        self.seqDeleteButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/trash.png'))
        self.seqDeleteButton.setToolTip('Delete all cardIAc sequences created')
        self.sequenceLayout.addWidget(self.seqDeleteButton, 1, 2, 1, 1)

        self.seqSeqSelectorLabel = qt.QLabel('Select a sequence: ')
        self.sequenceLayout.addWidget(self.seqSeqSelectorLabel, 2, 0)

        self.seqSeqSelectorList = slicer.qMRMLNodeComboBox()
        self.seqSeqSelectorList.setToolTip(
            'Sequence used to segment. You can '
            'select "Create new sequence" to create an empty sequence.')
        self.seqSeqSelectorList.nodeTypes = ['vtkMRMLSequenceNode']
        self.seqSeqSelectorList.selectNodeUponCreation = True
        self.seqSeqSelectorList.addEnabled = True
        self.seqSeqSelectorList.removeEnabled = False
        self.seqSeqSelectorList.noneEnabled = False
        self.seqSeqSelectorList.showHidden = True
        self.seqSeqSelectorList.showChildNodeTypes = False
        self.seqSeqSelectorList.setMRMLScene(slicer.mrmlScene)
        self.sequenceLayout.addWidget(self.seqSeqSelectorList, 2, 1, 1, 2)

        self.seqShowListOfSequencesCheckbox = qt.QCheckBox('Show sequences')
        self.seqShowListOfSequencesCheckbox.setToolTip(
            'Tap to see a list of current sequences in 3DSlicer '
            '(both from cardIAc or other modules). Only first and last data '
            'nodes are shown.')
        self.sequenceLayout.addWidget(self.seqShowListOfSequencesCheckbox, 3,
                                      0, 1, 1)

        # Connections
        self.seqCreateButton.clicked.connect(self.onCreateCardIAcSequence)
        self.seqShowListOfSequencesCheckbox.clicked.connect(
            self.onShowCreatedSequences)
        self.seqDeleteButton.clicked.connect(self.onDeleteCardIAcSequences)

    def startSegmentationSection(self, QHLine):
        """
        Set up GUI's Layout for 'Segmentation' section
        """
        self.segmIntervalLabel = qt.QLabel('Images to segment: ')
        self.segmentacionLayout.addWidget(self.segmIntervalLabel, 1, 0)

        self.segmTwoVolumesList = qt.QComboBox()
        self.segmTwoVolumesList.addItem('Complete sequence')
        self.segmTwoVolumesList.addItem('ED and ES volumes')
        self.segmTwoVolumesList.addItem('Unic volume')
        self.segmTwoVolumesList.setToolTip(
            'Select if you wish to segment the complete sequence, two volumes'
            'of the sequence (i.e. ED and ES) or a unic volume.')
        self.segmentacionLayout.addWidget(self.segmTwoVolumesList, 1, 1, 1, 2)

        self.segmTwoVolumesLabel = qt.QLabel('ED vol. / ES vol.: ')
        self.segmTwoVolumesLabel.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmTwoVolumesLabel, 2, 0)

        self.segmFirstImage = slicer.qMRMLNodeComboBox()
        self.segmFirstImage.setVisible(0)
        self.segmFirstImage.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.segmFirstImage.selectNodeUponCreation = True
        self.segmFirstImage.addEnabled = False
        self.segmFirstImage.removeEnabled = False
        self.segmFirstImage.noneEnabled = False
        self.segmFirstImage.showHidden = True
        self.segmFirstImage.showChildNodeTypes = False
        self.segmFirstImage.setMRMLScene(slicer.mrmlScene)
        self.segmFirstImage.setToolTip('First volume of the sequence.')
        self.segmentacionLayout.addWidget(self.segmFirstImage, 2, 1, 1, 1)

        self.segmSecondImage = slicer.qMRMLNodeComboBox()
        self.segmSecondImage.setVisible(0)
        self.segmSecondImage.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.segmSecondImage.selectNodeUponCreation = True
        self.segmSecondImage.addEnabled = False
        self.segmSecondImage.removeEnabled = False
        self.segmSecondImage.noneEnabled = False
        self.segmSecondImage.showHidden = True
        self.segmSecondImage.showChildNodeTypes = False
        self.segmSecondImage.setMRMLScene(slicer.mrmlScene)
        self.segmSecondImage.setToolTip('Second volume of the sequence.')
        self.segmentacionLayout.addWidget(self.segmSecondImage, 2, 2, 1, 1)

        self.segmOneVolumeLabel = qt.QLabel('Volume: ')
        self.segmOneVolumeLabel.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmOneVolumeLabel, 2, 0)

        self.segmOneVolume = slicer.qMRMLNodeComboBox()
        self.segmOneVolume.setVisible(0)
        self.segmOneVolume.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.segmOneVolume.selectNodeUponCreation = True
        self.segmOneVolume.addEnabled = False
        self.segmOneVolume.removeEnabled = False
        self.segmOneVolume.noneEnabled = False
        self.segmOneVolume.showHidden = True
        self.segmOneVolume.showChildNodeTypes = False
        self.segmOneVolume.setMRMLScene(slicer.mrmlScene)
        self.segmOneVolume.setToolTip('Select which volume to segment.')
        self.segmentacionLayout.addWidget(self.segmOneVolume, 2, 1, 1, 2)

        self.segmCenterManuallyButtonCheckbox = qt.QCheckBox(
            "(?) Heart's center:")
        self.segmCenterManuallyButtonCheckbox.setToolTip('Check for help')
        self.segmentacionLayout.addWidget(
            self.segmCenterManuallyButtonCheckbox, 3, 0, 1, 1)

        self.segmCenterManuallyButton = qt.QPushButton(' Select Center')
        self.segmCenterManuallyButton.setToolTip(
            '''Manual selection of the heart's center in case that automatic
            segmentation goes wrong.''')
        self.segmCenterManuallyButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/SelectCenter2.png'))
        self.segmentacionLayout.addWidget(self.segmCenterManuallyButton, 3, 1,
                                          1, 2)

        self.segmVisibleCheckbox = qt.QCheckBox('Show segmentation')
        self.segmVisibleCheckbox.setChecked(True)
        self.segmVisibleCheckbox.setToolTip(
            'Toggle between segmentations visible or hidden in the views.')
        self.segmentacionLayout.addWidget(self.segmVisibleCheckbox, 4, 0)

        self.segmSegmentationButton = qt.QPushButton('Segmentation')
        self.segmSegmentationButton.setToolTip(
            'When clicked, old cardIAc segmentations will be removed. Make '
            'sure to save before making new segmentations.')
        self.segmSegmentationButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/Segment.png'))
        self.segmentacionLayout.addWidget(self.segmSegmentationButton, 4, 1, 1,
                                          2)

        self.segmStatusLabel = qt.QLabel('Segmentation status:')
        self.segmentacionLayout.addWidget(self.segmStatusLabel, 5, 0)

        self.segmStatusIndicator = qt.QLineEdit()
        self.segmStatusIndicator.setToolTip(
            'After segmentation, a "success" message will be shown here.')
        self.segmStatusIndicator.setEnabled(0)
        self.segmentacionLayout.addWidget(self.segmStatusIndicator, 5, 1, 1, 2)

        self.segmOpacityLabel = qt.QLabel("Label's opacity")
        self.segmentacionLayout.addWidget(self.segmOpacityLabel, 6, 0, 1, 1)

        self.segmSliderOpacity = qt.QSlider(qt.Qt.Horizontal)
        self.segmSliderOpacity.setValue(100)
        self.segmentacionLayout.addWidget(self.segmSliderOpacity, 6, 1, 1, 2)

        self.segmSegmentationOptionsLabel = qt.QLabel('Segmentation options:')
        self.segmentacionLayout.addWidget(self.segmSegmentationOptionsLabel, 7,
                                          0, 1, 1)

        self.segmSaveButton = qt.QPushButton('Save Segm')
        self.segmSaveButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/save.png'))
        self.segmentacionLayout.addWidget(self.segmSaveButton, 7, 1, 1, 1)

        self.segmDeleteSegmButton = qt.QPushButton('Delete Segm')
        self.segmDeleteSegmButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/trash.png'))
        self.segmDeleteSegmButton.setToolTip(
            'Remove all cardIAc segmentations done to clean the scene.')
        self.segmentacionLayout.addWidget(self.segmDeleteSegmButton, 7, 2, 1,
                                          1)

        # Aditional options
        self.segmEditSegmentationCheckbox = qt.QCheckBox('Edit segmentation')
        self.segmentacionLayout.addWidget(self.segmEditSegmentationCheckbox, 8,
                                          0, 1, 1)

        self.segmCNNsOptionsCheckbox = qt.QCheckBox('Import model options')
        self.segmentacionLayout.addWidget(self.segmCNNsOptionsCheckbox, 8, 1,
                                          1, 1)

        # Options begining (hidden)
        self.segmHiddenLine1 = QHLine()
        self.segmHiddenLine1.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmHiddenLine1, 9, 0, 1, 3)

        # Edit Segmentations options (hidden)
        self.segmLabelToEditLabel = qt.QLabel(' Select label to edit:')
        self.segmLabelToEditLabel.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmLabelToEditLabel, 10, 0, 1,
                                          1)

        self.segmLabelToEditSelector = slicer.qMRMLNodeComboBox()
        self.segmLabelToEditSelector.setVisible(0)
        self.segmLabelToEditSelector.nodeTypes = ['vtkMRMLLabelMapVolumeNode']
        self.segmLabelToEditSelector.selectNodeUponCreation = True
        self.segmLabelToEditSelector.addEnabled = False
        self.segmLabelToEditSelector.removeEnabled = False
        self.segmLabelToEditSelector.noneEnabled = True
        self.segmLabelToEditSelector.showHidden = False
        self.segmLabelToEditSelector.showChildNodeTypes = False
        self.segmLabelToEditSelector.setMRMLScene(slicer.mrmlScene)
        self.segmLabelToEditSelector.setToolTip(
            'Label to edit manually (Type allowed: vtkMRMLLabelMapVolumeNode)')
        self.segmentacionLayout.addWidget(self.segmLabelToEditSelector, 10, 1,
                                          1, 2)

        self.segmOpenEditorLabel = qt.QLabel(' Edition options:')
        self.segmOpenEditorLabel.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmOpenEditorLabel, 11, 0, 1,
                                          1)

        self.segmEditSegmButton = qt.QPushButton('Edit')
        self.segmEditSegmButton.setToolTip(
            'Click to open "Segment Editor" '
            'and edit the volume selected (label)')
        self.segmEditSegmButton.setVisible(0)
        self.segmEditSegmButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/editSegm.png'))
        self.segmentacionLayout.addWidget(self.segmEditSegmButton, 11, 1, 1, 1)

        self.segmFinishEditionButton = qt.QPushButton('Save Edition')
        self.segmFinishEditionButton.setToolTip(
            'Changes will be overwritten in the label selected.')
        self.segmFinishEditionButton.setVisible(0)
        self.segmFinishEditionButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/EditionFinished.png'))
        self.segmentacionLayout.addWidget(self.segmFinishEditionButton, 11, 2,
                                          1, 1)

        # CNNs models options (hidden)
        self.segmCurrentModelLabel = qt.QLabel(' Current segm. model:')
        self.segmCurrentModelLabel.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmCurrentModelLabel, 14, 0, 1,
                                          1)

        self.segmCurrentModel = qt.QLineEdit()
        self.segmCurrentModel.setVisible(0)
        self.segmCurrentModel.setEnabled(0)
        self.segmCurrentModel.text = '  ' + self.getNNModelName()
        self.segmentacionLayout.addWidget(self.segmCurrentModel, 14, 1, 1, 2)

        self.segmImportNNButtonLabel = qt.QLabel(' Import pre trained Model:')
        self.segmImportNNButtonLabel.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmImportNNButtonLabel, 15, 0,
                                          1, 1)

        self.segmImportNNButton = qt.QPushButton(' Import Model')
        self.segmImportNNButton.setToolTip(
            'Load a CNN segmentation model from local disk.')
        self.segmImportNNButton.setIcon(
            qt.QIcon(base_path / 'Resources/Icons/import.png'))
        self.segmImportNNButton.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmImportNNButton, 15, 1, 1, 2)

        # Hidden end
        self.segmHelpCheckbox = qt.QCheckBox('(?) Help')
        self.segmHelpCheckbox.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmHelpCheckbox, 16, 0, 1, 1)

        self.segmHiddenLine2 = QHLine()
        self.segmHiddenLine2.setVisible(0)
        self.segmentacionLayout.addWidget(self.segmHiddenLine2, 17, 0, 1, 3)

        # Connections
        self.segmVisibleCheckbox.clicked.connect(self.onSegmVisibleCheckbox)
        self.segmTwoVolumesList.connect('currentIndexChanged(int)',
                                        self.onSegmentTwoVolumesLayout)
        self.segmSegmentationButton.clicked.connect(self.onApplySegmentation)
        self.segmEditSegmentationCheckbox.clicked.connect(
            self.onShowEditSegmOptions)
        self.segmCNNsOptionsCheckbox.clicked.connect(
            self.onShowImportModelOptions)
        self.segmImportNNButton.clicked.connect(self.onImportNNModel)
        self.segmSaveButton.clicked.connect(self.onSaveSegm)
        self.segmDeleteSegmButton.clicked.connect(
            self.onDeleteCardIAcLabelNodes)
        self.segmSliderOpacity.valueChanged.connect(self.onChangeOpacity)
        self.segmCenterManuallyButton.clicked.connect(
            self.onPlaceFiducialPointInHeartsCenter)
        # self.segmRefreshButton.clicked.connect(self.onRefreshModule)
        self.segmHelpCheckbox.clicked.connect(self.onShowHelpMessage)
        self.segmCenterManuallyButtonCheckbox.clicked.connect(
            self.onShowHelpMessage)
        self.segmEditSegmButton.clicked.connect(self.onEditSegmentation)
        self.segmFinishEditionButton.clicked.connect(
            self.onFinishEditSegmentation)

    def startQuantificationSection(self, QHLine):
        self.visDiastoleLabel = qt.QLabel('End-Diastole:')
        self.visLayout.addWidget(self.visDiastoleLabel, 0, 0, 1, 1)

        self.visDiastoleFrame = slicer.qMRMLNodeComboBox()
        self.visDiastoleFrame.nodeTypes = ['vtkMRMLLabelMapVolumeNode']
        self.visDiastoleFrame.selectNodeUponCreation = True
        self.visDiastoleFrame.addEnabled = False
        self.visDiastoleFrame.removeEnabled = False
        self.visDiastoleFrame.noneEnabled = True
        self.visDiastoleFrame.showHidden = False
        self.visDiastoleFrame.showChildNodeTypes = False
        self.visDiastoleFrame.setMRMLScene(slicer.mrmlScene)
        if self.visDiastoleFrame.currentNode():
            self.visDiastoleFrame.setToolTip(
                self.visDiastoleFrame.currentNode.GetName())
        else:
            self.visDiastoleFrame.setToolTip('Select end-diastolic frame.')
        self.visLayout.addWidget(self.visDiastoleFrame, 0, 1, 1, 1)

        self.visSystoleLabel = qt.QLabel('End-Systole:')
        self.visLayout.addWidget(self.visSystoleLabel, 0, 2, 1, 1)

        self.visSystoleFrame = slicer.qMRMLNodeComboBox()
        self.visSystoleFrame.nodeTypes = ['vtkMRMLLabelMapVolumeNode']
        self.visSystoleFrame.selectNodeUponCreation = True
        self.visSystoleFrame.addEnabled = False
        self.visSystoleFrame.removeEnabled = False
        self.visSystoleFrame.noneEnabled = True
        self.visSystoleFrame.showHidden = False
        self.visSystoleFrame.showChildNodeTypes = False
        self.visSystoleFrame.setMRMLScene(slicer.mrmlScene)
        if self.visSystoleFrame.currentNode():
            self.visSystoleFrame.setToolTip(
                self.visSystoleFrame.currentNode.GetName())
        else:
            self.visSystoleFrame.setToolTip('Select end-systolic frame.')
        self.visLayout.addWidget(self.visSystoleFrame, 0, 3, 1, 1)

        # --
        self.visInput1Label = qt.QLabel('Weight [kg]')
        self.visLayout.addWidget(self.visInput1Label, 1, 0, 1, 1)

        self.visInput1 = qt.QLineEdit()
        self.visLayout.addWidget(self.visInput1, 1, 1, 1, 1)

        self.visInput2Label = qt.QLabel('Height [cm]')
        self.visLayout.addWidget(self.visInput2Label, 1, 2, 1, 1)

        self.visInput2 = qt.QLineEdit()
        self.visLayout.addWidget(self.visInput2, 1, 3, 1, 1)

        # --
        self.visCalcButton = qt.QPushButton('Calculate')
        self.visLayout.addWidget(self.visCalcButton, 2, 0, 1, 2)

        self.visExportButton = qt.QPushButton('Export')
        self.visLayout.addWidget(self.visExportButton, 2, 2, 1, 2)

        # Line -----
        Line = QHLine()
        self.visLayout.addWidget(Line, 3, 0, 1, 4)

        self.visLVTitle = qt.QLabel('Left ventricle quantification')
        self.visLayout.addWidget(self.visLVTitle, 4, 0, 1, 2)

        self.visRVTitle = qt.QLabel('Right ventricle quantification')
        self.visLayout.addWidget(self.visRVTitle, 4, 2, 1, 2)

        # Line -----
        Line = QHLine()
        self.visLayout.addWidget(Line, 5, 0, 1, 4)

        # --
        self.visLVVLabel_ED = qt.QLabel('ED LVV[mL]:')
        self.visLayout.addWidget(self.visLVVLabel_ED, 6, 0, 1, 1)
        self.visLayout.setColumnMinimumWidth(0, 85)

        self.visLVVValue_ED = qt.QLabel('   --')
        self.visLayout.addWidget(self.visLVVValue_ED, 6, 1, 1, 1)
        self.visLayout.setColumnMinimumWidth(1, 80)

        self.visRVVLabel_ED = qt.QLabel('ED RVV[mL]:')
        self.visLayout.addWidget(self.visRVVLabel_ED, 6, 2, 1, 1)
        self.visLayout.setColumnMinimumWidth(2, 85)

        self.visRVVValue_ED = qt.QLabel('   --')
        self.visLayout.addWidget(self.visRVVValue_ED, 6, 3, 1, 1)
        self.visLayout.setColumnMinimumWidth(3, 80)

        # --
        self.visLVVLabel_ES = qt.QLabel('ES LVV[mL]:')
        self.visLayout.addWidget(self.visLVVLabel_ES, 7, 0, 1, 1)

        self.visLVVValue_ES = qt.QLabel('   --')
        self.visLayout.addWidget(self.visLVVValue_ES, 7, 1, 1, 1)

        self.visRVVLabel_ES = qt.QLabel('ES RVV[mL]:')
        self.visLayout.addWidget(self.visRVVLabel_ES, 7, 2, 1, 1)

        self.visRVVValue_ES = qt.QLabel('   --')
        self.visLayout.addWidget(self.visRVVValue_ES, 7, 3, 1, 1)

        # --
        self.visLVMassLabel = qt.QLabel('LV Mass[g]:')
        self.visLayout.addWidget(self.visLVMassLabel, 8, 0, 1, 1)

        self.visLVMassValue = qt.QLabel('   --')
        self.visLayout.addWidget(self.visLVMassValue, 8, 1, 1, 1)

        self.visRVMassLabel = qt.QLabel('RV Mass[g]:')
        self.visLayout.addWidget(self.visRVMassLabel, 8, 2, 1, 1)

        self.visRVMassValue = qt.QLabel('   --')
        self.visLayout.addWidget(self.visRVMassValue, 8, 3, 1, 1)

        # --
        self.visLVEjectionFractionLabel = qt.QLabel('LV EF[%]:')
        self.visLayout.addWidget(self.visLVEjectionFractionLabel, 9, 0, 1, 1)

        self.visLVEjectionFractionValue = qt.QLabel('   --')
        self.visLayout.addWidget(self.visLVEjectionFractionValue, 9, 1, 1, 1)

        self.visRVEjectionFractionLabel = qt.QLabel('RV EF[%]:')
        self.visLayout.addWidget(self.visRVEjectionFractionLabel, 9, 2, 1, 1)

        self.visRVEjectionFractionValue = qt.QLabel('   --')
        self.visLayout.addWidget(self.visRVEjectionFractionValue, 9, 3, 1, 1)

        # --
        self.visLayout.addWidget(QHLine(), 10, 0, 1, 4)

        # Hidden from ----------------------
        # --
        self.visLVVLabel_ED_BSA = qt.QLabel('ED LVV[mL/m2]:')
        self.visLVVLabel_ED_BSA.setVisible(0)
        self.visLayout.addWidget(self.visLVVLabel_ED_BSA, 11, 0, 1, 1)

        self.visLVVValue_ED_BSA = qt.QLabel('   --')
        self.visLVVValue_ED_BSA.setVisible(0)
        self.visLayout.addWidget(self.visLVVValue_ED_BSA, 11, 1, 1, 1)

        self.visRVVLabel_ED_BSA = qt.QLabel('ED RVV[mL/m2]:')
        self.visRVVLabel_ED_BSA.setVisible(0)
        self.visLayout.addWidget(self.visRVVLabel_ED_BSA, 11, 2, 1, 1)

        self.visRVVValue_ED_BSA = qt.QLabel('   --')
        self.visRVVValue_ED_BSA.setVisible(0)
        self.visLayout.addWidget(self.visRVVValue_ED_BSA, 11, 3, 1, 1)

        # --
        self.visLVVLabel_ES_BSA = qt.QLabel('ES LVV[mL/m2]:')
        self.visLVVLabel_ES_BSA.setVisible(0)
        self.visLayout.addWidget(self.visLVVLabel_ES_BSA, 12, 0, 1, 1)

        self.visLVVValue_ES_BSA = qt.QLabel('   --')
        self.visLVVValue_ES_BSA.setVisible(0)
        self.visLayout.addWidget(self.visLVVValue_ES_BSA, 12, 1, 1, 1)

        self.visRVVLabel_ES_BSA = qt.QLabel('ES RVV[mL/m2]:')
        self.visRVVLabel_ES_BSA.setVisible(0)
        self.visLayout.addWidget(self.visRVVLabel_ES_BSA, 12, 2, 1, 1)

        self.visRVVValue_ES_BSA = qt.QLabel('   --')
        self.visRVVValue_ES_BSA.setVisible(0)
        self.visLayout.addWidget(self.visRVVValue_ES_BSA, 12, 3, 1, 1)

        # --
        self.visLVMassLabel_BSA = qt.QLabel('LV Mass[g/m2]:')
        self.visLVMassLabel_BSA.setVisible(0)
        self.visLayout.addWidget(self.visLVMassLabel_BSA, 13, 0, 1, 1)

        self.visLVMassValue_BSA = qt.QLabel('   --')
        self.visLVMassValue_BSA.setVisible(0)
        self.visLayout.addWidget(self.visLVMassValue_BSA, 13, 1, 1, 1)

        self.visRVMassLabel_BSA = qt.QLabel('RV Mass[g/m2]:')
        self.visRVMassLabel_BSA.setVisible(0)
        self.visLayout.addWidget(self.visRVMassLabel_BSA, 13, 2, 1, 1)

        self.visRVMassValue_BSA = qt.QLabel('   --')
        self.visRVMassValue_BSA.setVisible(0)
        self.visLayout.addWidget(self.visRVMassValue_BSA, 13, 3, 1, 1)

        LineHid = QHLine()
        LineHid.setVisible(0)
        self.visLayout.addWidget(LineHid, 14, 0, 1, 4)
        # Hidden to ----------------------

        self.visCleanBiomarkersButton = qt.QPushButton('Clear values')
        self.visLayout.addWidget(self.visCleanBiomarkersButton, 15, 0, 1, 1)

        # Connections
        self.visCalcButton.clicked.connect(self.onCalculateBioMarkers)
        self.visExportButton.clicked.connect(self.onExportBioMarkers)
        self.visCleanBiomarkersButton.clicked.connect(self.onCleanBiomarkers)

    def setNNModelFile(self,
                       center_path=CENTER_MODEL_PATH,
                       seg_path=SEG_MODEL_PATH):
        """
        Set the neural network model file path used to segment.
        """
        self.model_center = Path(center_path)
        self.model_segment = Path(seg_path)
        print('CardIAc center model loaded: ', center_path)
        print('CardIAc segmentation model loaded: ', seg_path)

    def getNNModelFile(self):
        """
        Returns both center detection and segmentation model's paths.
        """
        return self.model_center, self.model_segment

    def getNNModelName(self):
        """
        Returns segmentation model's name (MODEL_NAME.h5)
        """
        return self.model_segment.parts[-1]

    def onCreateCardIAcSequence(self):
        """
        Creates a sequence with the background images uploaded.
        """

        if self.seqFromList.currentNode(
        ) is None or self.seqToList.currentNode() is None:
            self.errorMessage(
                '''Can't create the sequence: Check images selected.''')
            return

        # Delete previous sequences
        # TODO: Allow to have more than one sequence
        self.onDeleteCardIAcSequences()

        # Create the original sequence and put it in the sequence selector
        check_flag, mssge, sequenceNode = self.logic.createCardIAcSequence()

        if check_flag == 0:
            self.errorMessage(mssge)
            return
        elif check_flag == 1:
            self.seqSeqSelectorList.setCurrentNode(sequenceNode)

    def onSegmVisibleCheckbox(self):
        """
        Toggle between visible and hidden segmentation
        """

        # No segmentation in the scene
        if not slicer.mrmlScene.GetNodesByClass(
                'vtkMRMLLabelMapVolumeNode').GetItemAsObject(0):
            return

        # Get segmentation node in views
        segmInViews = self.logic.getNodeFromViews()

        # Get layout manager and views controller
        layoutManager = slicer.app.layoutManager()
        sliceViewName = layoutManager.sliceViewNames()[0]
        controller = layoutManager.sliceWidget(sliceViewName).sliceController()

        # Want visible
        if self.segmVisibleCheckbox.checked and segmInViews:
            controller.setLabelMapHidden(0)

        # Want hidden
        if not self.segmVisibleCheckbox.checked and segmInViews:
            controller.setLabelMapHidden(1)

    def showLabelOutline(self):
        """
        Show the label segmentation outlined
        """
        layoutManager = slicer.app.layoutManager()
        for name in layoutManager.sliceViewNames():
            controller = layoutManager.sliceWidget(name).sliceController()
            controller.showLabelOutline(True)

    def errorMessage(self, msj='Error...'):
        """
        Creates a qt pop up message of error, with 'msj' text argument.
        """
        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setText(msj)
        msg.setWindowTitle('Error')
        msg.exec_()

    def succesMessage(self, msj='Ok', title='Success'):
        """
        Creates a qt pop up message, with 'msj' text argument and 'title' text
        argument.
        """
        msg = qt.QMessageBox()
        msg.setText(msj)
        msg.setWindowTitle(title)
        msg.exec_()

    def onSegmentTwoVolumesLayout(self):
        """
        Toggle between different visualizations in GUI, depending user's
        selection: 'Complete sequence', 'ED and ES volumes' or 'Unic volume'.
        """
        if self.segmTwoVolumesList.currentText == 'Complete sequence':
            self.segmTwoVolumesLabel.setVisible(0)
            self.segmFirstImage.setVisible(0)
            self.segmSecondImage.setVisible(0)
            self.segmOneVolumeLabel.setVisible(0)
            self.segmOneVolume.setVisible(0)

        if self.segmTwoVolumesList.currentText == 'ED and ES volumes':
            self.segmTwoVolumesLabel.setVisible(1)
            self.segmFirstImage.setVisible(1)
            self.segmSecondImage.setVisible(1)
            self.segmOneVolumeLabel.setVisible(0)
            self.segmOneVolume.setVisible(0)

        if self.segmTwoVolumesList.currentText == 'Unic volume':
            self.segmTwoVolumesLabel.setVisible(0)
            self.segmFirstImage.setVisible(0)
            self.segmSecondImage.setVisible(0)
            self.segmOneVolumeLabel.setVisible(1)
            self.segmOneVolume.setVisible(1)

    def onApplySegmentation(self):
        """
        Connect with logic's method 'applySegmentation' depending user's
        selection for segmentation (one volume, interval or complete sequence).
        """

        # TODO: Refactor: move the logic to the Logic class

        # No sequence and no Volume to segment
        if (self.seqSeqSelectorList.currentNode() is None and
            (self.segmTwoVolumesList.currentText == 'Complete sequence'
             or self.segmTwoVolumesList.currentText == 'ED and ES volumes')):
            self.errorMessage('First select a sequence or volume to segment.')

        # No model selected
        elif not os.path.exists(self.model_segment) or not os.path.exists(
                self.model_center):
            self.errorMessage(
                'A model is required to segment, import one.\n %s\n %s' %
                (self.model_center, self.model_segment))

        # Check if a unic volume is going to be segmented. And ignore the
        # sequence
        elif self.segmTwoVolumesList.currentText == 'Unic volume':
            node1 = self.segmOneVolume.currentNode()
            node2 = None
            segType = 'vol'
            # Check the node type and do not allow to segment cardiac proxies
            if self.logic.isCardIAcSequenceProxy(node1.GetName()):
                self.errorMessage('Proxys not allowed to segment. Select '
                                  'a correct volume and try again.')
            elif node1.GetClassName() != 'vtkMRMLScalarVolumeNode':
                self.errorMessage(
                    'Only Volumes are allowed to segment. Select '
                    'a correct volume and try again.')
        elif self.segmTwoVolumesList.currentText == 'Complete sequence':
            # Full sequence to be segmented
            selectedSequence = self.seqSeqSelectorList.currentNode()
            # Check that all the nodes in the seq are ScalarVolumes
            scalar_volumes = True
            for i in range(selectedSequence.GetNumberOfDataNodes()):
                node_i = selectedSequence.GetNthDataNode(i)
                if node_i.GetClassName() != 'vtkMRMLScalarVolumeNode':
                    scalar_volumes = False
                    break
            # Empty external sequence
            if selectedSequence.GetNumberOfDataNodes() == 0:
                self.errorMessage(
                    'Selected sequence is empty. Edit a '
                    'sequence in "Sequences" module and try again.')
            elif scalar_volumes:
                node1 = selectedSequence.GetNthDataNode(0)
                node2 = selectedSequence.GetNthDataNode(
                    selectedSequence.GetNumberOfDataNodes() - 1)
                segType = 'seq'
            else:
                self.errorMessage(
                    'Nodes of the sequence must be ScalarVolumeNodes, not'
                    '"%s". Check sequence selected and try again.' %
                    node_i.GetClassName())

        elif self.segmTwoVolumesList.currentText == 'ED and ES volumes':
            # Two volumes of the sequence selected (ED and ES)
            selectedSequence = self.seqSeqSelectorList.currentNode()
            # Empty external sequence
            if selectedSequence.GetNumberOfDataNodes() == 0:
                self.errorMessage(
                    'Selected sequence is empty. Edit a '
                    'sequence in "Sequences" module and try again.')
            else:
                node1 = self.segmFirstImage.currentNode()
                node2 = self.segmSecondImage.currentNode()
                name1 = node1.GetName()
                name2 = node2.GetName()
                # Check that node1 and node2 are in seq
                seq_nodes = [
                    selectedSequence.GetNthDataNode(i).GetName()
                    for i in range(selectedSequence.GetNumberOfDataNodes())
                ]
                if (name1 not in seq_nodes) or (name2 not in seq_nodes):
                    self.errorMessage(
                        'Check volumes of interval: they must be part of the '
                        'sequence selected.')
                elif node1.GetClassName() != 'vtkMRMLScalarVolumeNode':
                    self.errorMessage(
                        'Nodes of the sequence must be ScalarVolumeNodes, not '
                        '"%s". Check sequence selected and try again.' %
                        node1.GetClassName())
                elif node2.GetClassName() != 'vtkMRMLScalarVolumeNode':
                    self.errorMessage(
                        'Nodes of the sequence must be ScalarVolumeNodes, not '
                        '"%s". Check sequence selected and try again.' %
                        node2.GetClassName())
                elif (self.logic.isCardIAcSequenceProxy(name1)
                      or self.logic.isCardIAcSequenceProxy(name2)):
                    self.errorMessage('Proxys not allowed to segment. Select '
                                      'a correct volume and try again.')
                elif node1 == node2:
                    # If the same volume selected for first and last, dont
                    # create sequence
                    segType = 'vol'
                    node2 = None
                elif selectedSequence.GetNumberOfDataNodes() == 2:
                    segType = 'seq'
                    # If the sequence has only two nodes, SegType = 'seq'
                    # instead of 'seq_short'
                else:
                    segType = 'seq_short'
                    # Segment two volumes from a large sequence

        # Nodes are ok, Check now if the segmentatino is already there
        if self.logic.CardiacSegmentationExists(node1, node2, segType):
            self.errorMessage('Segmentation already done. If you want to '
                              'continue remove the segmentation and try again')
        else:
            if segType != 'vol':
                self.enableSequenceCreation(enabled=False)
            status = self.logic.applySegmentation(firstVolume=node1,
                                                  lastVolume=node2,
                                                  SegType=segType)

            # Once the segmentation ends properly we set the set ED and ES
            if (status == 0 and segType != 'vol'):
                nodes = self.logic.ed_es_nodes
                if nodes['ED'] is not None and nodes['ES'] is not None:
                    self.visDiastoleFrame.setCurrentNode(nodes['ED'])
                    self.visSystoleFrame.setCurrentNode(nodes['ES'])

    def enableSequenceCreation(self, enabled=True):
        """
        Toggle between enabled/disabled the options to create sequences
        """
        if enabled:
            self.seqFromList.setEnabled(1)
            self.seqToList.setEnabled(1)
            self.seqSeqSelectorList.setEnabled(1)
            self.seqCreateButton.setEnabled(1)
            self.seqDeleteButton.setEnabled(1)

        if not enabled:
            self.seqFromList.setEnabled(0)
            self.seqToList.setEnabled(0)
            self.seqSeqSelectorList.setEnabled(0)
            self.seqCreateButton.setEnabled(0)
            self.seqDeleteButton.setEnabled(0)

    def onShowEditSegmOptions(self):
        """
        Show or hide 'EditSegmentation' advanced option in "Segmentation"
        section when checked by user. Only one advanced option can be shown at
        a time.
        """

        # Show Edit Segm options
        if self.segmEditSegmentationCheckbox.checked:

            # Hide other otions
            self.segmCNNsOptionsCheckbox.setChecked(0)
            self.onShowImportModelOptions()
            self.segmHiddenLine1.setVisible(1)
            self.segmHiddenLine2.setVisible(1)
            self.segmLabelToEditLabel.setVisible(1)
            self.segmLabelToEditSelector.setVisible(1)
            self.segmOpenEditorLabel.setVisible(1)
            self.segmFinishEditionButton.setVisible(1)
            self.segmHelpCheckbox.setVisible(1)
            self.segmEditSegmButton.setVisible(1)

        # Hide more options
        else:
            self.segmHiddenLine1.setVisible(0)
            self.segmHelpCheckbox.setVisible(0)
            self.segmHiddenLine2.setVisible(0)
            self.segmLabelToEditLabel.setVisible(0)
            self.segmLabelToEditSelector.setVisible(0)
            self.segmOpenEditorLabel.setVisible(0)
            self.segmFinishEditionButton.setVisible(0)
            self.segmHelpCheckbox.setVisible(0)
            self.segmEditSegmButton.setVisible(0)

    def onShowImportModelOptions(self):
        """
        Show or hide 'ImportModel' advanced option in "Segmentation" section
        when checked by user. Only one advanced option can be shown at a time.
        """
        # Show more options
        if self.segmCNNsOptionsCheckbox.checked:

            # Hide other otions
            self.segmEditSegmentationCheckbox.setChecked(0)
            self.onShowEditSegmOptions()
            self.segmHiddenLine1.setVisible(1)
            self.segmImportNNButtonLabel.setVisible(1)
            self.segmImportNNButton.setVisible(1)
            self.segmCurrentModelLabel.setVisible(1)
            self.segmCurrentModel.setVisible(1)
            self.segmHelpCheckbox.setVisible(1)
            self.segmHiddenLine2.setVisible(1)

        # Hide more options
        else:
            self.segmHiddenLine1.setVisible(0)
            self.segmImportNNButtonLabel.setVisible(0)
            self.segmImportNNButton.setVisible(0)
            self.segmCurrentModelLabel.setVisible(0)
            self.segmCurrentModel.setVisible(0)
            self.segmHelpCheckbox.setVisible(0)
            self.segmHiddenLine2.setVisible(0)

    def onSaveSegm(self):
        """
        Save all cardIAc LabelMapVolumeNodes created in dir specified (.nrrd)
        """

        # Check that there are labels
        if self.logic.getNumberOfCardIAcLabels() == 0:
            self.errorMessage("There's no segmentation created: Can't save.")
            return

        dir_path = qt.QFileDialog.getExistingDirectory()

        # Return if cancel (dir_path == '')
        if not dir_path:
            return

        # Get label's names
        n, labelNamesList = self.logic.getNumberOfCardIAcLabels(
            getLabelNames=True)

        try:
            for i, labelName in enumerate(labelNamesList):

                labelNode_i = slicer.mrmlScene.GetNodesByClassByName(
                    'vtkMRMLLabelMapVolumeNode', labelName).GetItemAsObject(0)
                storageNode = labelNode_i.GetNthStorageNode(0)
                fileName = os.path.join(dir_path, labelName + '.nrrd')
                storageNode.SetFileName(fileName)
                storageNode.WriteData(labelNode_i)

            self.succesMessage(
                'CardIAc labels from segmentation were correctly saved!',
                'Save finished')

        except Exception:
            # Remove any trash created in case something went wrong
            for i, labelName in enumerate(labelNamesList):

                fileName = os.path.join(dir_path, labelName + '.nrrd')
                os.remove(fileName)

            self.errorMessage('An error ocurred: Nothing was saved.')

    def onImportNNModel(self):
        """
        Stores a new CNM segmentation model path ('.h5' format) provided by
        user.
        """
        # Open folder
        model_segm_file = qt.QFileDialog.getOpenFileName(
            None, 'Path for Segmentation Model', '', 'HDF (*.h5)')

        # Return if canceled
        if not model_segm_file:
            return

        self.setNNModelFile(seg_path=Path(model_segm_file))

        name = self.getNNModelName()
        self.segmCurrentModel.text = name
        self.succesMessage('Local model "{}" is now loaded!'.format(name))

    def onDeleteCardIAcSequences(self):
        """
        Clearthe sequences (backg and seg) created by cardIAc
        """
        if self.logic.deleteCardIAcSequence():
            # Center views
            slicer.util.resetSliceViews()
            # Show up and center 3D view
            self.logic.showViewsIn3DSection()

    def onDeleteCardIAcLabelNodes(self):
        """
        Clear the cardIAc segmentations
        """
        # TODO: Allow to kep previous segmentations
        if self.logic.deleteCardiacSegmentation():
            self.enableSequenceCreation(enabled=True)
            self.resetSegmentationStatus()
            # Clear measures
            self.onCleanBiomarkers()

    def resetSegmentationStatus(self):
        self.segmStatusIndicator.text = ''

    def onChangeOpacity(self):
        # TODO get prev scene opacity instead of setting default always
        slicer.util.setSliceViewerLayers(
            labelOpacity=self.segmSliderOpacity.value / 100.)

    def onCalculateBioMarkers(self):
        """
        Connects with logic's method 'calculatBioMarkers' to get EF, Volumes
        and Myo mass. Then, prints values in 'Quantification' section.
        """

        # Check that ED and ES frames were provided
        if (self.visDiastoleFrame.currentNode() is None
                or self.visSystoleFrame.currentNode() is None):
            self.errorMessage(
                'End-Diastolic and End-Systolic frames must be provided. ')
            return

        ed_node = self.visDiastoleFrame.currentNode()
        es_node = self.visSystoleFrame.currentNode()

        # TODO: Remove logic from widget !!!

        # If weight and Height not provided --> BSA = 1:
        # Get Weight and Height (check that are floats)
        # BSA = sqrt((h*w)/3600) or BSA = 1 (default)
        try:
            w = float(self.visInput1.text)
            h = float(self.visInput2.text)

            # Avoid w or h < 0:
            if w <= 0 or h <= 0:
                self.errorMessage('W and h must be greater than 0.')
                return

            BSA = np.sqrt((h * w) / 3600)
        except Exception:
            if self.visInput1.text == '' or self.visInput2.text == '':
                BSA = 1
            else:
                w = self.visInput1.text
                h = self.visInput2.text
                self.errorMessage(
                    '"Weight" and "Height" must be numbers. Instead, "%s" '
                    'and "%s" were inserted.' % (w, h))
                return

        # Calculate biomarkers (floats) [LVV_ED, LVV_ES, RVV_ED, RVV_ES,
        # LV_mass, RV_mass, LV_EF, RV_EF]
        biomarkers = self.logic.calculateBioMarkers(ed_node, es_node)

        # Print biomks in GUI
        self.setBioMarkers(biomarkers, BSA)

    def onExportBioMarkers(self):
        """
        Stores biomarkers previously obtained in '.csv' file. Path provided by
        user.
        """

        # Get dir path to export
        wdw = qt.QFileDialog()
        wdw.setWindowTitle(
            'Select folder to export CardIAc quantification values:')
        dir_path = wdw.getExistingDirectory()

        # Return if cancel (dir_path == '')
        if not dir_path:
            return

        # Get biomarkers shown on the GUI. Return if biomarkers weren't
        # calculated yet
        list_of_markers = self.getBioMarkers()
        if list_of_markers is None:
            return

        # Convert to path using Pathlib library
        dir_path = Path(dir_path)

        # Unpack and convert to numpy arrays
        biomarkers = np.array(list_of_markers[0])
        biomarkers_BSA = np.array(list_of_markers[1])

        try:
            # Save biomarkers file
            file_biomks = open(dir_path / 'cardIAc_quantification.csv', 'w')
            file_biomks.write(self.visLVVLabel_ED.text +
                              '\t{:0.2f}\n'.format(biomarkers[0]))
            file_biomks.write(self.visLVVLabel_ES.text +
                              '\t{:0.2f}\n'.format(biomarkers[1]))
            file_biomks.write(self.visRVVLabel_ED.text +
                              '\t{:0.2f}\n'.format(biomarkers[2]))
            file_biomks.write(self.visRVVLabel_ES.text +
                              '\t{:0.2f}\n'.format(biomarkers[3]))
            file_biomks.write(self.visLVMassLabel.text +
                              '\t{:0.2f}\n'.format(biomarkers[4]))
            file_biomks.write(self.visRVMassLabel.text +
                              '\t{:0.2f}\n'.format(biomarkers[5]))
            file_biomks.write(self.visLVEjectionFractionLabel.text +
                              '\t{:0.2f}\n'.format(biomarkers[6]))
            file_biomks.write(self.visRVEjectionFractionLabel.text +
                              '\t{:0.2f}\n'.format(biomarkers[7]))
            file_biomks.close()

            # Save biomarkers normalized file
            if biomarkers_BSA.size != 0:
                file_biomks_BSA = open(
                    dir_path / 'cardIAc_quantification_BSA.csv', 'w')
                file_biomks_BSA.write(self.visLVVLabel_ED_BSA.text +
                                      '\t{:0.2f}\n'.format(biomarkers_BSA[0]))
                file_biomks_BSA.write(self.visLVVLabel_ES_BSA.text +
                                      '\t{:0.2f}\n'.format(biomarkers_BSA[1]))
                file_biomks_BSA.write(self.visRVVLabel_ED_BSA.text +
                                      '\t{:0.2f}\n'.format(biomarkers_BSA[2]))
                file_biomks_BSA.write(self.visRVVLabel_ES_BSA.text +
                                      '\t{:0.2f}\n'.format(biomarkers_BSA[3]))
                file_biomks_BSA.write(self.visLVMassLabel_BSA.text +
                                      '\t{:0.2f}\n'.format(biomarkers_BSA[4]))
                file_biomks_BSA.write(self.visRVMassLabel_BSA.text +
                                      '\t{:0.2f}\n'.format(biomarkers_BSA[5]))
                file_biomks_BSA.close()

            # Print success message
            self.succesMessage(
                'CardIAc quantification values were exported correctly!')

        except Exception:
            # Remove trash files if were created:
            try:
                os.remove(dir_path / 'cardIAc_quantification.csv')
                os.remove(dir_path / 'cardIAc_quantification_BSA.csv')
            except Exception:
                pass
            self.errorMessage(
                'An error ocurred while exporting values. Use other directory')

    def getBioMarkers(self):
        """
        Function that get list with current biomarkers printed on the
        quantification section.
        Returns two lists: [biomarkers], [biomarkers_normalized_with_BSA] or
        empty list if not calculated.
        """
        # Check that quantification was made. Look ED LVV Value != '   --'
        if self.visLVVValue_ED.text == '   --':
            self.errorMessage(
                'There are no CardIAc quantification values to export!')
            return None

        biomarkers = []
        biomarkers_BSA = []

        # Get biomarkers not normalized
        biomarkers.append(float(self.visLVVValue_ED.text))
        biomarkers.append(float(self.visLVVValue_ES.text))
        biomarkers.append(float(self.visRVVValue_ED.text))
        biomarkers.append(float(self.visRVVValue_ES.text))
        biomarkers.append(float(self.visLVMassValue.text))
        # If RVMyoMass was not segmented, print NAN
        try:
            biomarkers.append(float(self.visRVMassValue.text))
        except Exception:
            biomarkers.append(np.nan)
        try:
            biomarkers.append(float(self.visLVEjectionFractionValue.text))
        except Exception:
            biomarkers.append(np.nan)
        try:
            biomarkers.append(float(self.visRVEjectionFractionValue.text))
        except Exception:
            biomarkers.append(np.nan)

        # Get biomarkers normalized (if they were calculated)
        if self.visLVVLabel_ED_BSA.isVisible():
            biomarkers_BSA.append(float(self.visLVVValue_ED_BSA.text))
            biomarkers_BSA.append(float(self.visLVVValue_ES_BSA.text))
            biomarkers_BSA.append(float(self.visRVVValue_ED_BSA.text))
            biomarkers_BSA.append(float(self.visRVVValue_ES_BSA.text))
            biomarkers_BSA.append(float(self.visLVMassValue_BSA.text))
            try:
                biomarkers_BSA.append(
                    float(self.visRVMassValue_BSA.text
                          ))  # If RVMyoMass was not segmented, print NAN #
            except Exception:
                biomarkers_BSA.append(np.nan)

        return [biomarkers, biomarkers_BSA]

    def setBioMarkers(self, biomarkers, BSA):
        """
        Function that prints biomarkers on the quantification section.
        Calculations must be done first.
        """

        # Convert to str. If any biomarker is 0, replace to '--'
        biomk_str = []
        for b in biomarkers:
            if b != 0.0:
                biomk_str.append(str(np.around(b, 2)))
            else:
                biomk_str.append('--')

        # Update visualization
        # [LVV_ED,LVV_ES,RVV_ED,RVV_ES,LV_mass,RV_mass,LV_EF,RV_EF]
        self.visLVVValue_ED.text = '   ' + biomk_str[0]
        self.visLVVValue_ES.text = '   ' + biomk_str[1]
        self.visRVVValue_ED.text = '   ' + biomk_str[2]
        self.visRVVValue_ES.text = '   ' + biomk_str[3]
        self.visLVMassValue.text = '   ' + biomk_str[4]
        self.visRVMassValue.text = '   ' + biomk_str[5]
        self.visLVEjectionFractionValue.text = '   ' + biomk_str[6]
        self.visRVEjectionFractionValue.text = '   ' + biomk_str[7]

        if BSA == 1:
            self.visLVVLabel_ED_BSA.setVisible(0)
            self.visLVVValue_ED_BSA.setVisible(0)
            self.visLVVLabel_ES_BSA.setVisible(0)
            self.visLVVValue_ES_BSA.setVisible(0)
            self.visRVVLabel_ED_BSA.setVisible(0)
            self.visRVVValue_ED_BSA.setVisible(0)
            self.visRVVLabel_ES_BSA.setVisible(0)
            self.visRVVValue_ES_BSA.setVisible(0)
            self.visLVMassLabel_BSA.setVisible(0)
            self.visLVMassValue_BSA.setVisible(0)
            self.visRVMassLabel_BSA.setVisible(0)
            self.visRVMassValue_BSA.setVisible(0)

        else:
            # Convert to str. If any biomarker is 0, replace to '--'
            biomk_str = []
            for b in biomarkers:
                if b != 0.0:
                    biomk_str.append(str(np.around(b / BSA, 2)))
                else:
                    biomk_str.append('--')

            self.visLVVValue_ED_BSA.text = '   ' + biomk_str[0]
            self.visLVVValue_ES_BSA.text = '   ' + biomk_str[1]
            self.visRVVValue_ED_BSA.text = '   ' + biomk_str[2]
            self.visRVVValue_ES_BSA.text = '   ' + biomk_str[3]
            self.visLVMassValue_BSA.text = '   ' + biomk_str[4]
            self.visRVMassValue_BSA.text = '   ' + biomk_str[5]
            self.visLVVLabel_ED_BSA.setVisible(1)
            self.visLVVValue_ED_BSA.setVisible(1)
            self.visLVVLabel_ES_BSA.setVisible(1)
            self.visLVVValue_ES_BSA.setVisible(1)
            self.visRVVLabel_ED_BSA.setVisible(1)
            self.visRVVValue_ED_BSA.setVisible(1)
            self.visRVVLabel_ES_BSA.setVisible(1)
            self.visRVVValue_ES_BSA.setVisible(1)
            self.visLVMassLabel_BSA.setVisible(1)
            self.visLVMassValue_BSA.setVisible(1)
            self.visRVMassLabel_BSA.setVisible(1)
            self.visRVMassValue_BSA.setVisible(1)

    def onCleanBiomarkers(self):
        """
        Clean quantification section
        """
        # Update visualization
        self.visLVVValue_ED.text = '   --'
        self.visLVVValue_ES.text = '   --'
        self.visRVVValue_ED.text = '   --'
        self.visRVVValue_ES.text = '   --'
        self.visLVMassValue.text = '   --'
        self.visRVMassValue.text = '   --'
        self.visLVEjectionFractionValue.text = '   --'
        self.visRVEjectionFractionValue.text = '   --'

        self.visLVVLabel_ED_BSA.setVisible(0)
        self.visLVVValue_ED_BSA.setVisible(0)
        self.visLVVLabel_ES_BSA.setVisible(0)
        self.visLVVValue_ES_BSA.setVisible(0)
        self.visRVVLabel_ED_BSA.setVisible(0)
        self.visRVVValue_ED_BSA.setVisible(0)
        self.visRVVLabel_ES_BSA.setVisible(0)
        self.visRVVValue_ES_BSA.setVisible(0)
        self.visLVMassLabel_BSA.setVisible(0)
        self.visLVMassValue_BSA.setVisible(0)
        self.visRVMassLabel_BSA.setVisible(0)
        self.visRVMassValue_BSA.setVisible(0)

    def onPlaceFiducialPointInHeartsCenter(self):

        # Hide segmentations
        slicer.util.setSliceViewerLayers(label=None)
        self.logic.placeFiducialPointInHeartsCenter()

    def onRefreshModule(self):
        """
        Refresh module using father's class method onReload()
        """
        self.onReload()

    def onShowHelpMessage(self):
        """
        Pop up help messages for advanced options in "Segmentation" section,
        depending each case.
        """

        editSegmentationHelp = (
            "Once automatic segmentation is finished, "
            "it's possible to use the Slicer's Segment Editor Module "
            "to perform manual editions on the labels obtained.\n"
            "First, select the label to edit and click the 'Edit' button. "
            "After edition is finished, click the 'Save Edition' button "
            "to save changes.")
        centerSelectionHelp = (
            "Automatic segmentation deppends on heart's "
            "center detection. It's possible that such detection fails.\n"
            "In that case, this sections allows user to place a markup in the "
            " center of the heart and redo the automatic segmentation.\n"
            "First, select the heart's center manually placing a markup. "
            'Then, press "Segment" button to finish. This will use sequences'
            'configurations selected.')
        importCNNModelsHelp = (
            "It's possible to import a pre trained CNN model"
            "for segmentation (.h5 admitted only):\n"
            "More information for the admitted models can be found in CardIAc "
            "Manual.\nSee https://gitlab.com/Curiale/cardiac.")

        if self.segmEditSegmentationCheckbox.checked:
            self.succesMessage(editSegmentationHelp, 'Help: Edit segmentation')
            self.segmHelpCheckbox.setChecked(0)
            return

        if self.segmCenterManuallyButtonCheckbox.checked:
            self.succesMessage(centerSelectionHelp, 'Help: Center selection')
            self.segmCenterManuallyButtonCheckbox.setChecked(0)
            return

        if self.segmCNNsOptionsCheckbox.checked:
            self.succesMessage(importCNNModelsHelp, 'Help: Center selection')
            self.segmHelpCheckbox.setChecked(0)
            return

    def onEditSegmentation(self):
        """
        Connect with logic's method 'editSegmentation' after label was selected
        to manual edition.
        """
        node = self.segmLabelToEditSelector.currentNode()
        # Check that label was selected
        if node is None:
            self.errorMessage('Select one label to edit manually.')
        elif self.logic.isCardIAcSequenceProxy(node.GetName()):
            # Avoid editing proxys
            self.errorMessage('Proxy nodes are not editable! Select one label '
                              'to edit manually.')
        elif self.logic.isCardIAcLabelMapVolume(node.GetName()):
            self.logic.editSegmentation()
        else:
            self.errorMessage(
                'This edition option is allowed for CardIAc labels only. '
                'To edit other labels, use Segmentation Module.')

    def onFinishEditSegmentation(self):
        """
        Connect with logic's method 'finishEditSegmentation' once manual
        edition was made by user. Pop ups message of "success".
        """

        # Check current segmentation being edited
        if self.logic.segmNodeInEdition is None:
            self.errorMessage('Please begin a new edition before saving.')
            return

        status = self.logic.finishEditSegmentation()
        name = self.segmLabelToEditSelector.currentNode().GetName()
        if status:
            self.errorMessage('"%s" label is not modified.' % name)
        else:
            self.succesMessage(
                '"%s" label has been correctly modified.' % name,
                'Edition finished!')

    def onShowCreatedSequences(self):
        """
        This function shows a list of created cardIAc sequences with its first
        and last volume.
        """

        if self.seqShowListOfSequencesCheckbox.checked:

            cardiacText = ''
            otherText = ''
            n_cardiac = 0
            n_others = 0

            # Get current sequences in the scene (cardIAc and others)
            sequencesInfo = self.logic.getCurrentSequencesInfo()

            # CardIAc sequences in slicer scene
            for seqID in sequencesInfo['cardiac'].keys():

                seqName = slicer.mrmlScene.GetNodeByID(seqID).GetName()
                n_cardiac += 1
                cardiacText += '  {}) '.format(n_cardiac)
                cardiacText += '[' + seqName + ']' + ': '

                # Empty sequence
                if sequencesInfo['cardiac'][seqID][0] == '' or sequencesInfo[
                        'cardiac'][seqID][1] == '':
                    cardiacText += '(Empty)' + '\n'

                else:
                    node1Name = sequencesInfo['cardiac'][seqID][0]
                    node2Name = sequencesInfo['cardiac'][seqID][1]
                    cardiacText += ('"' + node1Name + '"' + ' --> ' + '"' +
                                    node2Name + '"' + '\n')

            # Other sequences in slicer scene
            for seqID in sequencesInfo['others'].keys():

                seqName = slicer.mrmlScene.GetNodeByID(seqID).GetName()
                n_others += 1
                otherText += '  {}) '.format(n_others)
                otherText += '[' + seqName + ']' + ': '

                # Empty sequence
                if sequencesInfo['others'][seqID][0] == '' or sequencesInfo[
                        'others'][seqID][1] == '':
                    otherText += '(Empty)' + '\n'

                else:
                    node1Name = sequencesInfo['others'][seqID][0]
                    node2Name = sequencesInfo['others'][seqID][1]
                    otherText += ('"' + node1Name + '"' + ' --> ' + '"' +
                                  node2Name + '"' + '\n')

            # Build information message
            text = 'Sequences in scene:\n\n'
            text += 'CardIAc sequences\n' + 30 * '-' + '\n'
            text += cardiacText + '\n'
            text += 'Other sequences\n' + 30 * '-' + '\n'
            text += otherText

            self.succesMessage(text, 'Information')
            self.seqShowListOfSequencesCheckbox.setChecked(0)


#
# CardiacSegmentationLogic
#


class CardiacSegmentationLogic(ScriptedLoadableModuleLogic):
    """
    This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    """

    def __init__(self, parent):
        self._widgets = parent
        self._segmStatus = LOGIC_IC['segmStatus']
        self._cnn_input_label = LOGIC_IC['cnn_input_label']
        self._cnn_output_label = LOGIC_IC['cnn_output_label']
        self._cnn_roi_size = LOGIC_IC['cnn_roi_size']

        self._manualCenterFlag = False
        self._labelInEdition = None
        self._segmNodeInEdition = None
        self._segmentEditorNode = None
        self._active_backg_node = None

        self._ed_es_nodes = {'ED': None, 'ES': None}

        self._proxy_backg_name = LOGIC_IC['backg_name_default']
        self._proxy_seg_name = LOGIC_IC['seg_name_default']
        self._proxy_backg_short_name = LOGIC_IC['backg_name_short']
        # Sequences stores the sequences name created with the cardIAc in sets
        self._cardiac_sequences = {
            'backg': {
                'proxies': set(),
                'sequences': set(),
                'browsers': set()
            },
            'backg_short': {
                'proxies': set(),
                'sequences': set(),
                'browsers': set()
            },
            'seg': {
                'proxies': set(),
                'sequences': set(),
                'browsers': set()
            }
        }
        self._node_types = {
            'sequences': 'vtkMRMLSequenceNode',
            'browsers': 'vtkMRMLSequenceBrowserNode',
            'proxies_seg': 'vtkMRMLLabelMapVolumeNode',
            'proxies_backg': 'vtkMRMLScalarVolumeNode',
            'proxies_backg_short': 'vtkMRMLScalarVolumeNode',
        }
        # Check if the app was restarted and there are some cardiac_sequences
        # in the nodes
        self._load_current_cardiac_sequences()

    def _load_current_cardiac_sequences(self):
        """
        Load posible cardiac sequences already created
        """
        for key, val in self._cardiac_sequences.items():
            for cardiac_type in val:
                val = cardiac_type
                if val == 'proxies':
                    val += '_' + key
                node_type = self._node_types[val]

                elements = slicer.mrmlScene.GetNodesByClass(node_type)
                for i in range(elements.GetNumberOfItems()):
                    b_i = elements.GetItemAsObject(i)
                    b_name = b_i.GetName()
                    name_preffix = '_'.join(b_name.split('_')[:2])
                    if key == 'backg':
                        cardiac_prefix = self.proxy_backg_name
                    elif key == 'backg_short':
                        cardiac_prefix = self.proxy_backg_short_name
                    else:
                        cardiac_prefix = self.proxy_seg_name
                    if name_preffix == cardiac_prefix:
                        self._cardiac_sequences[key][cardiac_type].add(b_name)

    def _deleteCardIAcElement(self, class_name, cardiac_type, cae_type=None):
        """
        Remove  class_name nodes from the MRMLScene according to the
        cardiac_type.
        class_name: vtkMRMLSequenceNode,vtkMRMLSequenceBrowserNode,
                    vtkMRMLLabelMapVolumeNode or vtkMRMLScalarVolumeNode. (see
                    definition of self._node_types)
        cardiac_type: 'sequences', 'proxies' or 'browsers'

        NOTE: Cardiac elements starts with cardiac_img, cardiac_seg or
        cardiac_short (according to the proxy_..._name).

        Returns the number of elements removed
        """
        if cae_type is None:
            cae_type = ''
        ne = 0
        elements = slicer.mrmlScene.GetNodesByClass(class_name)
        n_type = len(cardiac_type.split('_'))
        for i in range(elements.GetNumberOfItems()):
            b_i = elements.GetItemAsObject(i)
            if (class_name == 'vtkMRMLColorTableNode'
                    or (class_name == 'vtkMRMLLabelMapVolumeNode'
                        and cae_type != 'proxies')):
                split_name = b_i.GetName().split('_')[-n_type:]
            else:
                split_name = b_i.GetName().split('_')[:n_type]
            b_prefix = '_'.join(split_name)
            if b_prefix == cardiac_type:
                slicer.mrmlScene.RemoveNode(b_i)
                ne += 1
        return ne

    @property
    def widgets(self):
        return self._widgets

    @property
    def segmStatus(self) -> int:
        return self._segmStatus

    @property
    def cnn_input_label(self) -> str:
        return self._cnn_input_label

    @property
    def cnn_output_label(self) -> str:
        return self._cnn_output_label

    @property
    def cnn_roi_size(self) -> int:
        return self._cnn_roi_size

    @property
    def manualCenterFlag(self) -> bool:
        """
        Flag used to know that the LV center was defined manyally
        """
        return self._manualCenterFlag

    @manualCenterFlag.setter
    def manualCenterFlag(self, value: bool):
        self._manualCenterFlag = value

    @property
    def labelInEdition(self):
        return self._labelInEdition

    @labelInEdition.setter
    def labelInEdition(self, value):
        self._labelInEdition = value

    @property
    def segmNodeInEdition(self):
        return self._segmNodeInEdition

    @segmNodeInEdition.setter
    def segmNodeInEdition(self, value):
        self._segmNodeInEdition = value

    @property
    def segmentEditorNode(self):
        return self._segmentEditorNode

    @segmentEditorNode.setter
    def segmentEditorNode(self, value):
        self._segmentEditorNode = value

    @property
    def active_backg_node(self):
        """
        Returns the active segmentation backg. The background can be a full
        (seq) or short (short_seq) sequence node or a volume (vol)
        """
        return self._active_backg_node

    @property
    def proxy_backg_name(self) -> str:
        return self._proxy_backg_name

    @property
    def proxy_seg_name(self) -> str:
        return self._proxy_seg_name

    @property
    def proxy_backg_short_name(self) -> str:
        return self._proxy_backg_short_name

    @property
    def ed_es_nodes(self) -> dict:
        """
        Returns ED and ES nodes
        """
        return self._ed_es_nodes

    def getCardIAcPrefix(self, name):
        """
        This method returns the cardIAc prefix according to the internal proxy
        name convention (proxy_backg_name, proxy_backg_short_name and
        proxy_seg_name)
        """
        # Identify the background and remove the prefix
        n_prefix_backg = len(self.proxy_backg_name.split('_'))
        n_prefix_backg_short = len(self.proxy_backg_short_name.split('_'))
        n_prefix_seg = len(self.proxy_seg_name.split('_'))
        # Identify the proxy name and get the preffix
        name_split = name.split('_')
        prefix_backg = '_'.join(name_split[:n_prefix_backg])
        prefix_backg_short = '_'.join(name_split[:n_prefix_backg_short])
        prefix_seg = '_'.join(name_split[:n_prefix_seg])
        prefix = None
        if prefix_backg == self.proxy_backg_name:
            prefix = self.proxy_backg_name
        elif prefix_backg_short == self.proxy_backg_short_name:
            prefix = self.proxy_backg_short_name
        elif prefix_seg == self.proxy_seg_name:
            prefix = self.proxy_seg_name
        return prefix

    def isCardIAcSequenceProxy(self, name):
        """
        Check if the node name is one of the cardIAc proxies
        """
        for val in self._cardiac_sequences.values():
            if name in val['proxies']:
                return True

    def CardiacSegmentationExists(self, node1, node2, segType):
        """
        Check if the if the segmentation exists
        """
        if segType == 'vol':
            # There is no seq
            node_name = node1.GetName() + '_' + self.proxy_seg_name
        elif segType == 'seq' or segType == 'seq_short':
            suffix = node1.GetName() + '_to_' + node2.GetName()
            node_name = (self.proxy_backg_name
                         if segType == 'seq' else self.proxy_backg_short_name)
            node_name += suffix
        nodes = slicer.mrmlScene.GetNodesByClassByName(
            'vtkMRMLLabelMapVolumeNode', node_name)
        return nodes.GetNumberOfItems() > 0

    def isCardIAcLabelMapVolume(self, name):
        """
        Check if the node name is one of the cardIAc LabelMapVolume
        segmentations. CardiacSegmentations ends with self.proxy_seg_prefix
        """
        n_suffix = len(self.proxy_seg_name.split('_'))
        name_suffix = '_'.join(name.split('_')[-n_suffix:])
        return name_suffix == self.proxy_seg_name

    def show_sequence(self, backg_name, label_name=None):
        """
        Set the sequence as the background or segmentation and show
        """
        # Get the volume node (ScalarVolumeNone)
        backg_node = slicer.mrmlScene.GetNodesByClassByName(
            'vtkMRMLScalarVolumeNode', backg_name).GetItemAsObject(0)
        kwargs = {'background': backg_node.GetID()}
        if label_name is not None:
            # Get the volume node to use as label
            label_node = slicer.mrmlScene.GetNodesByClassByName(
                'vtkMRMLScalarVolumeNode', label_name).GetItemAsObject(0)
            kwargs['label'] = label_node.GetID()
            # Set the active background only when the label is given. Avoid to
            # set up as active_backg the cardiac_img when no segmentation
            # was performed
            self._active_backg_node = backg_node
        slicer.util.setSliceViewerLayers(**kwargs)

        self.reset_views()

    def reset_views(self):
        """
        Reset the view and set to no interpolate
        """

        # Lock views
        self.lockSliceViews()
        # Center views
        slicer.util.resetSliceViews()
        # Show up and center 3D view
        self.showViewsIn3DSection()
        self.notInterpolate()
        slicer.app.restoreOverrideCursor()

    def deleteCardIAcSequence(self, backg=True, backg_short=True, seg=True):
        """
        Delete sequences, volumes and browsers created by cardIAc for backg,
        backg_short and or seg.
        """
        # TODO: Allow to have more than one cardiac Sequence
        #  if ne > 0:
        # If seqBrowser are deleted first, a new rare sequenceNode is
        # created.
        # Avoid deleting seqBrowser first
        # 1) Delete Sequence Nodes
        ne = self.deleteCardIAcSequenceElement('sequences',
                                               backg=backg,
                                               backg_short=backg_short,
                                               seg=seg)
        # 2) Delete proxies (Scalar and/or LabelMap) Nodes
        ne += self.deleteCardIAcSequenceElement('proxies',
                                                backg=backg,
                                                backg_short=backg_short,
                                                seg=seg)
        # 3) Delete Edition Nodes and volume seg.
        if seg:
            ne += self.deleteCardIAcManualEdtionNodes()
            ne += self._deleteCardIAcElement('vtkMRMLLabelMapVolumeNode',
                                             self.proxy_seg_name)
        # 4) Delete Browsers
        ne += self.deleteCardIAcSequenceElement('browsers',
                                                backg=backg,
                                                backg_short=backg_short,
                                                seg=seg)
        return ne

    def deleteCardIAcSequenceElement(self,
                                     element,
                                     backg=True,
                                     backg_short=True,
                                     seg=False):
        """
        Remove the elements created by CardIAc.
        element: 'sequences', 'proxies' or 'browsers'
        """

        def remove_all(element_key, key):
            n = 0
            node_type = (self._node_types[element_key + '_' +
                                          key] if element_key == 'proxies' else
                         self._node_types[element_key])

            while self._cardiac_sequences[key][element_key]:
                name = self._cardiac_sequences[key][element_key].pop()
                cardiac_type = '_'.join(name.split('_')[:2])
                n += self._deleteCardIAcElement(node_type,
                                                cardiac_type,
                                                cae_type=element_key)
            return n

        # Number of elements deleted
        ne = 0
        if backg:
            ne += remove_all(element, 'backg')
        if backg_short:
            ne += remove_all(element, 'backg_short')
        if seg:
            ne += remove_all(element, 'seg')
        return ne

    def deleteCardIAcManualEdtionNodes(self):
        """
        Delete cardIAc vtkMRMLColorTableNodes (these are
        generated when editing manually a cardIAc label).
        Returns the number of elements deleted
        """
        # Remove cardIAc SegmentationNode and SegmentationEditorNode
        ne = 0
        if self.segmNodeInEdition is not None:
            slicer.mrmlScene.RemoveNode(self.segmNodeInEdition)
            ne += 1
        if self.segmentEditorNode is not None:
            slicer.mrmlScene.RemoveNode(self.segmentEditorNode)
            ne += 1
        if ne > 0:
            self.resetCardIAcEditionFlags()
        return 0

    def deleteCardiacSegmentation(self):
        """
        Delete sequences, edition nodes and LabelMaps
        """
        n_short = len(self._cardiac_sequences['backg_short']['proxies'])
        backg_short = n_short > 0

        # 1) Remove manual segmentation
        ne = self.deleteCardIAcManualEdtionNodes()

        # 2) Remove cardiac seg sequences
        ne += self.deleteCardIAcSequenceElement('sequences',
                                                backg=False,
                                                backg_short=backg_short,
                                                seg=True)
        # 3) Delete proxies (Scalar and/or LabelMap) Nodes
        ne += self.deleteCardIAcSequenceElement('proxies',
                                                backg=False,
                                                backg_short=backg_short,
                                                seg=True)
        # 4) Remove cardiac browsers
        if backg_short:
            ne += self.deleteCardIAcSequenceElement('browsers',
                                                    backg=False,
                                                    backg_short=True,
                                                    seg=False)
        # 5) Remove cardiac LabelMap
        ne += self._deleteCardIAcElement('vtkMRMLLabelMapVolumeNode',
                                         self.proxy_seg_name)
        # 6) Remove cardIAc colorTableNodes (generated at manual edition)
        ne += self._deleteCardIAcElement('vtkMRMLColorTableNode',
                                         self.proxy_seg_name + '_ColorTable')
        # 7) Remove cardIAc Heart Center
        ne += self._deleteCardIAcElement('vtkMRMLMarkupsFiducialNode',
                                         'Heart Center')

        self.widgets.resetSegmentationStatus()
        return ne

    def getCardIAcElementNode(self, name, cardiac_sequence, cardiac_type):
        """
        This method return the cardiac element node according to the name.
        If name doesn't exist, then a new one is created.
        """
        nodes = slicer.mrmlScene.GetNodesByName(name)
        n_nodes = nodes.GetNumberOfItems()
        node_type = self._node_types[cardiac_type]

        idx = None
        for i in range(n_nodes):
            node_i = nodes.GetItemAsObject(i)
            if node_i.GetClassName() == node_type:
                idx = i
                break

        if idx is None:
            node = slicer.mrmlScene.AddNewNodeByClass(node_type)
            node.SetName(name)
            # Add the Sequence to our internal set
            self._cardiac_sequences[cardiac_sequence][cardiac_type].add(name)
        else:
            node = nodes.GetItemAsObject(idx)

        return node

    def createCardIAcSequence(self):
        """
        This function creates a sequence with the volumes loaded. New sequences
        are stored and named as 'NUMBERseqProxy_Backg_cardIAc', so having two
        cardIAc sequences will result in sequences nodes named as
        '1seqProxy_Backg_cardIAc' and '2seqProxy_Backg_cardIAc' (for proxys,
        SeqNodes and SeqBrowNodes).  Short sequences from selecting interval do
        not accumulate. Instead, prev shortSeq is deleted when a new
        one is created.
        -----------
        Returns:
        int flag: 0 if a problem ocurred or 1 if nothing went wrong
        str mssg: message with the problem description if flag == 0
        -----------
        """

        # Check for sequencesLogic module, depending slicer version
        if slicer.app.majorVersion * 100 + slicer.app.minorVersion < 411:
            # old slicer versions
            sequencesModule = slicer.modules.sequencebrowser
        else:
            sequencesModule = slicer.modules.sequences  # new slicer versions

        # First volume of sequence
        VolN_orig1 = self.widgets.seqFromList.currentNode()
        VolN_orig2 = self.widgets.seqToList.currentNode()
        VolN_orig1ID = VolN_orig1.GetID()
        VolN_orig2ID = VolN_orig2.GetID()

        # Sequence name
        seq_suffix = VolN_orig1.GetName() + '_to_' + VolN_orig2.GetName()
        seq_name = self.proxy_backg_name + '_' + seq_suffix

        # Check that sequence does not already exist. Avoid duplications
        if seq_name in self._cardiac_sequences['backg']['sequences']:
            return 0, ('CardIAc sequence from volume "%s" to volume "%s" '
                       'already exist. Nothing is done.') % (
                           VolN_orig1.GetName(), VolN_orig2.GetName()), None

        # Split output will be:
        # ['vtkMRMLScalarVolume' , 'Number of Node in main scene']
        VolN_orig1ID_split = VolN_orig1ID.split('Node')
        VolN_orig2ID_split = VolN_orig2ID.split('Node')

        frames = int(VolN_orig2ID_split[-1]) - int(VolN_orig1ID_split[-1])

        # Same node as first and last
        if frames == 0:
            return 0, ('No sequence created. The first and last images must '
                       'be different.'), None

        # First image bigger than last
        if frames < 0:
            return 0, ("Can't create the sequence: "
                       'The interval selected is not correct.'), None

        # Number of nodes for sequence. Add 1 to include first and last.
        frames += 1

        # Create new seqBrowNode and SeqNode
        SeqBrowN = self.getCardIAcElementNode(seq_name, 'backg', 'browsers')
        SeqN_orig = self.getCardIAcElementNode(seq_name, 'backg', 'sequences')

        # Create the sequence
        for frame_i in range(frames):
            NodeForSequence = slicer.mrmlScene.GetNodeByID(
                'vtkMRMLScalarVolumeNode' +
                str(frame_i + int(VolN_orig1ID_split[-1])))
            SeqN_orig.SetDataNodeAtValue(NodeForSequence, str(frame_i))

        # When the Observer is Added the proxy (ScalarVolumeNode) is created
        SeqBrowN.SetAndObserveMasterSequenceNodeID(SeqN_orig.GetID())
        # Add the proxy volume to our internal set
        self._cardiac_sequences['backg']['proxies'].add(seq_name)

        sequencesModule.widgetRepresentation().setActiveBrowserNode(SeqBrowN)

        self.show_sequence(seq_name)

        return 1, '', SeqN_orig

    def compute_lv_center(self, volume):
        """
        Detection of the LV center using the segmention of the middle slice.
        The method returns the center of the LV in numpy coordinates
        volume: a numpy array with shape (z, y, x) and it is downsampled to
        (middle,) + CENTER_MODEL_INPUT_SHAPE for example, (middle,64,64)
        """
        # Load the model
        path_center, path_segement = self.widgets.getNNModelFile()
        center_model = load_model(path_center,
                                  custom_objects=custom_objects_imported)
        slice_shape = volume.shape[1:]
        middleSlice = volume.shape[0] // 2
        middleSlice = volume[middleSlice]
        middleSlice = transform.resize(middleSlice,
                                       CENTER_MODEL_INPUT_SHAPE,
                                       mode='symmetric').astype(
                                           'float32')  # example (64,64)
        center = images.normalize_image(middleSlice)
        # center shape (64,64)
        center = center_model.predict(center[np.newaxis, ..., np.newaxis],
                                      verbose=2).squeeze()

        # Resize to original shape to compute CM
        center = np.round(
            transform.resize(center, slice_shape,
                             mode='symmetric')).astype(int)

        # Compute centroid of heart (instead of scaling)
        cm = np.array(ndimage.measurements.center_of_mass(center))
        return cm

    def ventricle_segmentation(self, volumes, lv_center, spacing):
        """
        Segmentation of the myocardial tissue, left and right ventricle
        volumes: a numpy array of shape (frames, z, y, x)
        lv_center: tuple, list or numpy array of shape (y, x)
        Returns: a numpy array with the same shape as volumes
        """
        # Detect ED and ES according to the max and min LV volume
        idx_ed, idx_es = (0, 0)

        # Load models, labels by default: rv(1), myo(2) lv(3)
        path_center, path_segement = self.widgets.getNNModelFile()
        seg_model = load_model(path_segement,
                               custom_objects=custom_objects_imported)

        # Remove extra inputs and extra outputs if they are found
        try:
            seg_model = architecture.remove_extra_inputs_outputs(
                seg_model, self.cnn_input_label, self.cnn_output_label)
        except Exception as e:
            print(e)
            pass

        # Tissue model outputs should be:
        # myo: 0
        # lv: 1
        # rv: 2
        ntissue = seg_model.output.shape[-1]
        if ntissue == 3:
            idx_myo = 0
            idx_lv = 1
            idx_rv = 2
        else:
            raise Exception('Model output must be 3')

        slice_shape = volumes.shape[2:]
        # TODO: Implement non isotropic resolution (dy != dx)
        # roi ; r = 'cnn_roi_size' mm for each side (90)
        # 90 [mm] // 1.25 [pix/mm] = 72 [pix]
        r = int(round(self.cnn_roi_size / spacing[0]))
        ycm, xcm = lv_center.round().astype(int)
        ydown, yup = ycm - r, ycm + r
        xleft, xright = xcm - r, xcm + r

        # rois initial shape  (frames, z, y, x)
        rois = volumes[..., ydown:yup, xleft:xright]
        # rois final shape  (frames*z, y, x)
        rois.shape = (np.prod(rois.shape[:2]), ) + rois.shape[2:]

        # Seg model shape (frames*z, 128, 128)
        data = np.zeros((rois.shape[0], ) + SEG_MODEL_INPUT_SHAPE)

        for slice_frame in range(rois.shape[0]):
            # iamge size # (128,128)
            I = transform.resize(rois[slice_frame],
                                 SEG_MODEL_INPUT_SHAPE,
                                 mode='symmetric').astype('float32')
            data[slice_frame] = images.normalize_image(I)

        # Second Model: Segmentation
        self.updateStatusBar('Segmentation in progress. Please, wait...')
        # Seg shape (frames*z,128,128,n_classes)
        seg = seg_model.predict(data[..., np.newaxis], verbose=2)
        self.updateStatusBar('Finishing final settings...')

        # ----------------------------------------------
        # Container for segmentated images
        # img seg shape  (frames*z, 256, 256)
        imgs_seg = np.zeros((rois.shape[0], ) + slice_shape, dtype=int)
        for i in range(seg.shape[0]):
            myo_i = seg[i, ..., idx_myo]
            lv_i = seg[i, ..., idx_lv]

            myo_i = np.round(
                transform.resize(myo_i, rois.shape[1:],
                                 mode='symmetric')).astype(int)
            lv_i = np.round(
                transform.resize(lv_i, rois.shape[1:],
                                 mode='symmetric')).astype(int)
            # TODO: Avoiding overlapping is not properly working
            # Avoid intersection between rv-myo and lv-myo classes
            lv_myo = lv_i + myo_i
            #  Correct the LV
            lv_i[lv_myo == 2] = 0
            # Right Ventricle
            if idx_rv is not None:
                rv_i = seg[i, ..., idx_rv]
                rv_i = np.round(
                    transform.resize(rv_i, rois.shape[1:],
                                     mode='symmetric')).astype(int)
                # Avoid intersection between rv-myo and lv-myo classes
                rv_myo = rv_i + myo_i
                #  Correct the RV
                rv_i[rv_myo == 2] = 0

            # Assign labels
            # TODO: Let the user select the labels, now is hardcoded
            # 1:RV, 2:myo, 3:LV
            myo_i = LABEL_TISSUES['myo'] * myo_i
            lv_i = LABEL_TISSUES['lv'] * lv_i
            I_seg = myo_i + lv_i
            if idx_rv is not None:
                rv_i = LABEL_TISSUES['rv'] * rv_i
                I_seg += rv_i

            imgs_seg[i, ydown:yup, xleft:xright] = I_seg
        imgs_seg.shape = volumes.shape  # (frames, z, y, x)

        # TODO: This can be optimized avoiding the for in the frames if before
        # we do a reshape.
        min_volumen = np.inf
        max_volumen = 0
        for i in range(len(imgs_seg)):
            lv_volumen = (imgs_seg[i] == LABEL_TISSUES['lv']).sum()
            if lv_volumen < min_volumen:
                min_volumen = lv_volumen
                idx_es = i
            if lv_volumen > max_volumen:
                max_volumen = lv_volumen
                idx_ed = i

        return imgs_seg, idx_ed, idx_es

    def show_time(self, t_start, t_end):
        delta_t = t_end - t_start
        if delta_t >= 3600:
            delta_t = delta_t / 3600
            self.updateStatusBar(
                'Finished correctly!! Elapsed time: {0:.1f} [hs]'.format(
                    delta_t))
        elif delta_t >= 60:
            delta_t = delta_t / 60
            self.updateStatusBar(
                'Finished correctly!! Elapsed time: {0:.1f} [min]'.format(
                    delta_t))
        elif delta_t < 60:
            self.updateStatusBar(
                'Finished correctly!! Elapsed time: {0:.1f} [sec]'.format(
                    delta_t))

    def applySegmentation(self, firstVolume, lastVolume=None, SegType=None):
        """
        Apply the segmention according to the type
        """

        # Initialization
        if lastVolume is None:
            SegType = 'vol'

        # Set 'In progress' messages
        slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        self.updateStatusBar('Setting initial configurations...')

        t_start = time.time()

        # ------------------------------------------------------------------
        # Segmentation over volume (no sequence)
        # ------------------------------------------------------------------
        if SegType == 'vol':
            # TODO: Test what Happend if a sequence seg was created before
            Vol2Segment_name = firstVolume.GetName()
            # Center detection (CNN first model or manually)
            volume = slicer.util.arrayFromVolume(firstVolume)  # (z,y,x)
            if self.manualCenterFlag:
                # We dont need to scale in this case, as it's selected from
                # views. But we need to transpose
                # convert from ITK/VTK heart position (x,y,z) to numpy (z,y,x)
                cm = self.heartCenterPosition[0:2][::-1]
                cm = np.array(cm)
            else:
                cm = self.compute_lv_center(volume)

            # Volume information
            spacing = firstVolume.GetSpacing()

            # volumes_seg is (1, z, y, x)
            volumes_seg, idx_ed, idx_es = self.ventricle_segmentation(
                volume[np.newaxis, ...], cm, spacing)
            t_end = time.time()
            self.show_time(t_start, t_end)
            # Before conver to vtk the segmentation needs to be in float
            # for a proper Slicer visualization
            volumes_seg = volumes_seg.astype(float)

            VolN_Seg = images.slicerNodeFromArray(
                volumes_seg[0], Vol2Segment_name + '_' + self.proxy_seg_name,
                firstVolume)

            # Setting up the active segmentation
            self._active_backg_node = firstVolume

            # Show the segmentation and Reset the views
            # Slicer views
            slicer.util.setSliceViewerLayers(background=firstVolume.GetID(),
                                             label=VolN_Seg.GetID())
            self.widgets.showLabelOutline()
            self.reset_views()
        else:
            # -------------------------------------------------------------
            # Segmentation over sequence
            # -------------------------------------------------------------
            name1 = firstVolume.GetName()
            name2 = lastVolume.GetName()
            seq_suffix = name1 + '_to_' + name2

            # Check for sequencesLogic module, depending slicer version
            if slicer.app.majorVersion * 100 + slicer.app.minorVersion < 411:
                # old slicer versions
                seqModule = slicer.modules.sequencebrowser
            else:
                seqModule = slicer.modules.sequences  # new slicer versions

            # Get the sequence selected
            selected_sequence = self.widgets.seqSeqSelectorList.currentNode()

            # TODO: Allow to use more than onw segmentation short and seq
            # Delete the cardiac sequence elements including the Browser
            self.deleteCardIAcSequence(backg=False, backg_short=True, seg=True)

            # Complete sequence -------------------------------------
            if SegType == 'seq':
                backg_name = 'backg'
                seq_name = self.proxy_backg_name + '_' + seq_suffix
                seq_node = selected_sequence
                frames = selected_sequence.GetNumberOfDataNodes()
                #  # Delete the cardiac sequence elements
                #  self.deleteCardIAcSequence(backg=False,
                #                             backg_short=False,
                #                             seg=True)
                firstVolume_idx = 0

            # Short sequence ----------------------------------------
            if SegType == 'seq_short':
                backg_name = 'backg_short'
                # Delete the cardiac sequence elements including the Browser
                #  self.deleteCardIAcSequence(backg=False,
                #                             backg_short=True,
                #                             seg=True)
                # Set up the browser and sequence node for the backg_short
                seq_name = self.proxy_backg_short_name + '_' + seq_suffix
                seq_node = self.getCardIAcElementNode(seq_name, 'backg_short',
                                                      'sequences')

                frames = 2
                # Search first volume index in the original sequence (as its
                # shorter, it may be different than 0)
                for idx in range(selected_sequence.GetNumberOfDataNodes()):
                    nodeInSeq_i = selected_sequence.GetNthDataNode(idx)
                    if name1 == nodeInSeq_i.GetName():
                        firstVolume_idx = idx
                    if name2 == nodeInSeq_i.GetName():
                        lastVolume_idx = idx

            seq_browser = self.getCardIAcElementNode(seq_name, backg_name,
                                                     'browsers')
            # Set up the seg. sequence Node
            label_seq_name = self.proxy_seg_name + '_' + seq_suffix
            label_seq_node = self.getCardIAcElementNode(
                label_seq_name, 'seg', 'sequences')

            # Create the volumes (frame, z, y, x) from the sequence
            # VTK/ITK shape is (x,y,z) and numpy is (z,y,x)
            volume_shape = firstVolume.GetImageData().GetDimensions()[::-1]
            volumes = np.zeros((frames, ) + volume_shape)
            # Get Volumes from the entire sequence
            if SegType == 'seq':
                for f in range(frames):
                    volumes[f] = slicer.util.arrayFromVolume(
                        selected_sequence.GetNthDataNode(f + firstVolume_idx))

            # Get the volumes from ED and ES, ie. short sequence
            if SegType == 'seq_short':
                volumes[0] = slicer.util.arrayFromVolume(
                    selected_sequence.GetNthDataNode(firstVolume_idx))
                volumes[1] = slicer.util.arrayFromVolume(
                    selected_sequence.GetNthDataNode(lastVolume_idx))

            if self.manualCenterFlag:
                # We dont need to scale in this case, as it's selected from
                # views. But we need to transpose.
                # convert from ITK/VTK heart position (x,y,z) to numpy (z,y,x)
                cm = np.array(self.heartCenterPosition[0:2][::-1])

            else:
                volume = slicer.util.arrayFromVolume(firstVolume)  # (z,y,x)
                cm = self.compute_lv_center(volume)

            # Volume information
            spacing = firstVolume.GetSpacing()
            # volumes_seg is (frames, z, y, x)
            volumes_seg, idx_ed, idx_es = self.ventricle_segmentation(
                volumes, cm, spacing)

            # Before conver to vtk the segmentation needs to be in float
            # for a proper Slicer visualization
            volumes_seg = volumes_seg.astype(float)

            t_end = time.time()
            self.show_time(t_start, t_end)

            for f in range(frames):

                if SegType == 'seq':
                    seqScene_i = selected_sequence.GetNthDataNode(f)
                elif SegType == 'seq_short':
                    if f == 0:
                        seqScene_i = selected_sequence.GetNthDataNode(
                            firstVolume_idx)
                    elif f == 1:
                        seqScene_i = selected_sequence.GetNthDataNode(
                            lastVolume_idx)

                name = (seqScene_i.GetName() + '_' + self.proxy_seg_name)

                VolN_Seg = images.slicerNodeFromArray(volumes_seg[f], name,
                                                      firstVolume)
                if idx_ed == f:
                    self._ed_es_nodes['ED'] = VolN_Seg
                if idx_es == f:
                    self._ed_es_nodes['ES'] = VolN_Seg

                label_seq_node.SetDataNodeAtValue(VolN_Seg, str(f))
                if SegType == 'seq_short':
                    seq_node.SetDataNodeAtValue(seqScene_i, str(f))

            # Sequences and sequenceBrowser settings
            seqModule.widgetRepresentation().setActiveBrowserNode(seq_browser)

            # When the Observer is Added the proxy (ScalarVolumeNode) is
            # created
            seq_browser.SetAndObserveMasterSequenceNodeID(seq_node.GetID())
            self._cardiac_sequences['seg']['proxies'].add(label_seq_name)

            # With the synchronization sequence proxy (ScalarVolumeNode) is
            # created if it doesn't exist
            seq_browser.AddSynchronizedSequenceNodeID(label_seq_node.GetID())
            self._cardiac_sequences[backg_name]['proxies'].add(seq_name)

            # Show Segmentation and Reset the views
            self.widgets.showLabelOutline()
            self.show_sequence(seq_name, label_name=label_seq_name)

        # Set some flags and final configs
        self.manualCenterFlag = False
        # Remove Heart Center
        self._deleteCardIAcElement('vtkMRMLMarkupsFiducialNode',
                                   'Heart Center')

        return 0

    def getNumberOfCardIAcLabels(self, getLabelNames=False):
        """
        Returns the number of cardIAc LabelMapVolumeNodes in the scene.
        --------
        Params:
        * getLabelNames: if True, a list with the names of labels is returned
        as second argument (n_labels,name_labels)
        --------
        """

        n_cardiac_labels = 0
        if getLabelNames:
            label_names = []

        # Identify cardIAc label nodes
        number_of_slicerLabelNodes = slicer.mrmlScene.GetNodesByClass(
            'vtkMRMLLabelMapVolumeNode').GetNumberOfItems()

        for i in range(number_of_slicerLabelNodes):
            labelNode_i = slicer.mrmlScene.GetNodesByClass(
                'vtkMRMLLabelMapVolumeNode').GetItemAsObject(i)

            # Node found
            if labelNode_i is not None:
                labelNode_i_Name = labelNode_i.GetName()

                # cardIAc labelMapVolNode found
                if labelNode_i_Name.endswith('cardIAc_seg'):
                    n_cardiac_labels += 1
                    if getLabelNames:
                        label_names.append(labelNode_i_Name)

        # Return information
        if getLabelNames:
            return n_cardiac_labels, label_names
        else:
            return n_cardiac_labels

    def lockSliceViews(self):
        # Set linked slice views in all existing slice composite nodes and in
        # the default node
        sliceCompositeNodes = slicer.util.getNodesByClass(
            'vtkMRMLSliceCompositeNode')
        defaultSliceCompositeNode = slicer.mrmlScene.GetDefaultNodeByClass(
            'vtkMRMLSliceCompositeNode')

        if not defaultSliceCompositeNode:
            defaultSliceCompositeNode = slicer.mrmlScene.CreateNodeByClass(
                'vtkMRMLSliceCompositeNode')
            slicer.mrmlScene.AddDefaultNode(defaultSliceCompositeNode)
        sliceCompositeNodes.append(defaultSliceCompositeNode)

        for sliceCompositeNode in sliceCompositeNodes:
            sliceCompositeNode.SetLinkedControl(True)

    def showViewsIn3DSection(self):
        # Show slices in 3D views
        layoutManager = slicer.app.layoutManager()
        for sliceViewName in layoutManager.sliceViewNames():
            controller = layoutManager.sliceWidget(
                sliceViewName).sliceController()
            controller.setSliceVisible(True)

        # Center views
        threeDWidget = layoutManager.threeDWidget(0)
        threeDView = threeDWidget.threeDView()
        threeDView.resetFocalPoint()

    def notInterpolate(self):
        for node in slicer.mrmlScene.GetNodesByClass(
                'vtkMRMLScalarVolumeDisplayNode'):
            node.SetInterpolate(0)

    def getNodeFromViews(self, layer='label'):
        """
        Returns the current node in views.
        It can be a segmentation 'vtkMRMLLabelMapVolumeNode' (layer = 'label')
        or a background 'vtkMRMLScalarVolumeNode' (layer = 'vol').
        """

        # Bad input label
        if layer != 'label' and layer != 'vol':
            raise Exception(
                '"label" or "vol" flags admitted. Get {} instead'.format(
                    layer))

        # Get layout manager
        lm = slicer.app.layoutManager()
        sliceLogic = lm.sliceWidget('Red').sliceLogic()

        # Get node
        if layer == 'vol':
            node = sliceLogic.GetLayerVolumeNode(0)
        if layer == 'label':
            node = sliceLogic.GetLayerVolumeNode(2)

        return node

    def calculateBioMarkers(self, ed_node, es_node):
        """
        Estimation of biomarkers based on End-Diast and End-Syst frames
        provided.
        Output: [LVV_ED, LVV_ES, RVV_ED, RVV_ES, LV_mass, RV_mass, LV_EF,
        RV_EF] (0:backg, 1:RV, 2:Myo, 3:LV)
        """
        # Set nodes as ED and ES for being consistent. User could change the
        # suggested volumes
        self._ed_es_nodes = {'ED': ed_node, 'ES': es_node}

        # Tissues labels
        #  backg_label = 0
        rv_label = LABEL_TISSUES['rv']
        myo_lv_label = LABEL_TISSUES['myo']
        lv_label = LABEL_TISSUES['lv']
        myo_rv_label = 4

        # Density of myo
        rho = measures.MYO_DENSITY

        # Calculate volume = (#voxels != 0).V_voxel_i
        ED_vol = self.widgets.visDiastoleFrame.currentNode()
        ES_vol = self.widgets.visSystoleFrame.currentNode()
        ED_array = slicer.util.arrayFromVolume(ED_vol)
        ES_array = slicer.util.arrayFromVolume(ES_vol)

        # Volume of voxel i: (spacing_x).(spacing_y).(spacing_z)
        V_i = np.prod(ED_vol.GetSpacing())  # mm3

        # Count number of voxels of each class
        nvox_ED_LV = np.sum(ED_array == lv_label)
        nvox_ES_LV = np.sum(ES_array == lv_label)
        nvox_ED_RV = np.sum(ED_array == rv_label)
        nvox_ES_RV = np.sum(ES_array == rv_label)
        nvox_Myo_LV = np.sum(ED_array == myo_lv_label)
        nvox_Myo_RV = np.sum(ED_array == myo_rv_label)

        # Volumes at ED and ES [mm3]
        LVV_ED = float(nvox_ED_LV * V_i)
        LVV_ES = float(nvox_ES_LV * V_i)
        RVV_ED = float(nvox_ED_RV * V_i)
        RVV_ES = float(nvox_ES_RV * V_i)

        # Stroke Volume SV
        LV_SV = LVV_ED - LVV_ES
        RV_SV = RVV_ED - RVV_ES

        # Ejection fraction SV/VV_ED
        try:
            LV_EF = LV_SV / LVV_ED * 100.0
        except Exception:
            LV_EF = 0.0
        try:
            RV_EF = RV_SV / RVV_ED * 100.0
        except Exception:
            RV_EF = 0

        # Myo Volume
        MyoVol_LV = nvox_Myo_LV * V_i
        MyoVol_RV = nvox_Myo_RV * V_i

        # Myo Mass = rho[g/mm3].V[mm3]
        LV_mass = float(rho * MyoVol_LV)
        RV_mass = float(rho * MyoVol_RV)

        # Proper units
        LVV_ED = LVV_ED * 10**-3  # [mL]
        LVV_ES = LVV_ES * 10**-3
        RVV_ED = RVV_ED * 10**-3
        RVV_ES = RVV_ES * 10**-3

        return [LVV_ED, LVV_ES, RVV_ED, RVV_ES, LV_mass, RV_mass, LV_EF, RV_EF]

    def getHeartsCenterFromClick(self, event, caller):
        """
        Description: This function waits for a PointPositionDefinedEvent on the
        Markup node for the RV, then it removes the markup, gets the position
        of RV and saves it to rv_pts dictionary.
        Event: markupsVolNode
        Caller: (?)
        """

        # We get the volume and markup node
        volumeNode = self.getNodeFromViews('vol')
        markupsNode = slicer.mrmlScene.GetNodesByClassByName(
            'vtkMRMLMarkupsFiducialNode', 'Heart Center').GetItemAsObject(0)

        # Then we get the world coordinates and transform it to ijk
        point_ras = [0, 0, 0, 1]
        num_fids = markupsNode.GetNumberOfFiducials()
        markupsNode.GetNthFiducialWorldCoordinates(num_fids - 1, point_ras)
        transform_ras_to_volume_ras = vtk.vtkGeneralTransform()
        point_volume_ras = transform_ras_to_volume_ras.TransformPoint(
            point_ras[0:3])

        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            None, volumeNode.GetParentTransformNode(),
            transform_ras_to_volume_ras)

        volume_ras_to_ijk = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(volume_ras_to_ijk)
        point_ijk = [0, 0, 0, 1]
        volume_ras_to_ijk.MultiplyPoint(np.append(point_volume_ras, 1.0),
                                        point_ijk)
        point_ijk = [int(round(c)) for c in point_ijk[0:3]]
        self.heartCenterPosition = point_ijk
        self.manualCenterFlag = True

        # Then we remove everyting, save the coordinates and close the widget
        # window.
        slicer.mrmlScene.RemoveNode(markupsNode)
        self.placeWidget.close()
        interactionNode = slicer.mrmlScene.GetNodeByID(
            'vtkMRMLInteractionNodeSingleton')
        interactionNode.SwitchToViewTransformMode()

        # User message: Segmentations begins now
        self.widgets.succesMessage(
            'Center was selected correctly! Now, click "Segmentation" '
            'button again to finish segmentation.', 'Center Selected')

    def placeFiducialPointInHeartsCenter(self):
        """
        Description: This function starts a markup node and sets a callback
        function with an observer.
        """

        # First we get the volume node, and create a new Markups Node
        volumeNode = self.getNodeFromViews('vol')
        if not volumeNode:
            self.widgets.errorMessage(
                'Views are empty! Set ED volume in views to select'
                "the heart's center and try again.")
            return

        markupsNode = slicer.mrmlScene.AddNewNodeByClass(
            'vtkMRMLMarkupsFiducialNode', 'Heart Center')

        self.placeWidget = slicer.qSlicerMarkupsPlaceWidget()
        self.placeWidget.minimumHeight = 15
        self.placeWidget.minimumWidth = 350
        self.placeWidget.setMRMLScene(slicer.mrmlScene)
        self.placeWidget.setCurrentNode(markupsNode)
        self.placeWidget.buttonsVisible = False
        self.placeWidget.placeButton().show()
        self.placeWidget.show()

        markupsNode.AddObserver(
            slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent,
            self.getHeartsCenterFromClick)

    def editSegmentation(self):

        # Delete possible previous segmentationNodes (cardIAcEdition) or keep
        # editing them
        prevSegmNode = slicer.mrmlScene.GetNodesByClassByName(
            'vtkMRMLSegmentationNode', 'cardIAcEdition').GetItemAsObject(0)

        if prevSegmNode:

            # Get prev segmentEditorNode (make sure that it wasnt removed by
            # Segmentation Module)
            prevSegmEditorNode = slicer.mrmlScene.GetNodesByClassByName(
                'vtkMRMLSegmentEditorNode',
                'cardIAcEdition').GetItemAsObject(0)
            if not prevSegmEditorNode:
                prevSegmEditorNode = slicer.vtkMRMLSegmentEditorNode()
                prevSegmEditorNode.SetName('cardIAcEdition')
                slicer.mrmlScene.AddNode(prevSegmEditorNode)

            labelInEditionName = self.labelInEdition.GetName()

            qm = qt.QMessageBox()
            msg = ("There's already"
                   'one label in edition ("%s"). Begin new edition?  \n'
                   'Select "Yes" for new edition (old changes will be lost).\n'
                   'Select "No" for keep editing "%s".' %
                   (labelInEditionName, labelInEditionName))
            ret = qm.question(None, 'Edition in progress', msg, qm.Yes | qm.No)

            # Create new segmentation node (cardIAcEdition)
            if ret == qm.Yes:
                slicer.mrmlScene.RemoveNode(prevSegmNode)
                slicer.mrmlScene.RemoveNode(prevSegmEditorNode)

                self.segmNodeInEdition = slicer.mrmlScene.AddNewNodeByClass(
                    'vtkMRMLSegmentationNode')
                self.segmNodeInEdition.SetName('cardIAcEdition')
                slicer.modules.segmentations.logic(
                ).ImportLabelmapToSegmentationNode(self.labelInEdition,
                                                   self.segmNodeInEdition)
                self.segmNodeInEdition.CreateClosedSurfaceRepresentation()

                self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
                self.segmentEditorNode.SetName('cardIAcEdition')
                slicer.mrmlScene.AddNode(self.segmentEditorNode)

            # Get prev segmentation node (cardIAcEdition) and keep editing
            elif ret == qm.No:
                self.segmNodeInEdition = prevSegmNode
                self.segmentEditorNode = prevSegmEditorNode

        # Get labelNode selected as there is no previous edition
        self.labelInEdition = self.widgets.segmLabelToEditSelector.currentNode(
        )

        # Create new segmNode and segmEditorNode if not prevSegmNode
        if not prevSegmNode:
            self.segmNodeInEdition = slicer.mrmlScene.AddNewNodeByClass(
                'vtkMRMLSegmentationNode')
            self.segmNodeInEdition.SetName('cardIAcEdition')
            slicer.modules.segmentations.logic(
            ).ImportLabelmapToSegmentationNode(self.labelInEdition,
                                               self.segmNodeInEdition)
            self.segmNodeInEdition.CreateClosedSurfaceRepresentation()

            self.segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
            self.segmentEditorNode.SetName('cardIAcEdition')
            slicer.mrmlScene.AddNode(self.segmentEditorNode)

        # Get correct master volume node (background use in edition)
        background_name = '_'.join(
            self.labelInEdition.GetName().split('_')[:-3])
        background = slicer.mrmlScene.GetNodesByClassByName(
            'vtkMRMLScalarVolumeNode', background_name).GetItemAsObject(0)

        segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
        segmentEditorWidget.setMRMLScene(slicer.mrmlScene)

        segmentEditorWidget.setMRMLSegmentEditorNode(self.segmentEditorNode)
        segmentEditorWidget.setSegmentationNode(self.segmNodeInEdition)
        segmentEditorWidget.setMasterVolumeNode(background)

        # Switch to Segment Editor Module.
        slicer.util.selectModule('SegmentEditor')

        # Lock views
        self.lockSliceViews()

        # Quit self.labelInEdition from views and show Segmentation.
        slicer.util.setSliceViewerLayers(label=None)

        layoutManager = slicer.app.layoutManager()
        sliceViewName = layoutManager.sliceViewNames()[0]
        controller = layoutManager.sliceWidget(sliceViewName).sliceController()
        controller.setSegmentationHidden(0)

        # slicer.mrmlScene.RemoveNode(self.labelInEdition)

    def resetCardIAcEditionFlags(self):
        """
        Reset flags to None. Call this method when manual edition is done.
        """
        self.segmentEditorNode = None
        self.segmNodeInEdition = None
        self.labelInEdition = None

    def getActiveSegmentationElements(self):
        """
        Return the active segmentation sequence and proxy
        """
        backg_node = self._active_backg_node
        seq_node = None
        label_node = None
        if backg_node is not None:
            node_name = backg_node.GetName()
            prefix = self.getCardIAcPrefix(node_name)
            # prefix is None only if the active backg is a volume
            if prefix is not None:
                # Remove cardiac backg preffix and add seg. preffix
                name_suffix = node_name.split(prefix + '_')[-1]
                seq_name = self.proxy_seg_name + '_' + name_suffix
                if seq_name in self._cardiac_sequences['seg']['sequences']:
                    seq_node = slicer.mrmlScene.GetNodesByClassByName(
                        'vtkMRMLSequenceNode', seq_name).GetItemAsObject(0)
                    label_node = slicer.mrmlScene.GetNodesByClassByName(
                        'vtkMRMLScalarVolumeNode', seq_name).GetItemAsObject(0)
        return seq_node, label_node

    def getCardiacSegmentationIndex(self, seq_node, name):
        """
        This method return the index inside the sequence node for a
        particular element name. If the name is not in seq_nodes returns None
        """
        idx = None
        for i in range(seq_node.GetNumberOfDataNodes()):
            node_i = seq_node.GetNthDataNode(i)
            if node_i.GetName() == name:
                idx = i
                break
        return idx

    def finishEditSegmentation(self):
        """
        This method update the segmentation sequence and set the active backg
        and label volumes in the views
        """

        # Export LabelMap from segmNodeInEdition to labelInEdition
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            self.segmNodeInEdition, self.labelInEdition)

        seq_node, label_node = self.getActiveSegmentationElements()
        if seq_node:
            idx = self.getCardiacSegmentationIndex(
                seq_node, self.labelInEdition.GetName())
            seq_node.UpdateDataNodeAtValue(self.labelInEdition, str(idx))
            slicer.util.setSliceViewerLayers(
                background=self._active_backg_node.GetID(),
                label=label_node.GetID())
            status = 0
        elif self._active_backg_node:
            # There is no sequencia, the segmentation was made for a unique
            # volume
            slicer.util.setSliceViewerLayers(
                background=self._active_backg_node.GetID(),
                label=self.labelInEdition.GetID())
            status = 0
        else:
            status = 1

        # Remove cardIAc segmNode and segmEditorNode
        self.deleteCardIAcManualEdtionNodes()
        return status

    def getCurrentSequencesInfo(self):
        """
        Returns a dictionary with cardIAc created sequences (for Label and
        Background, not included IntervalBackground)
        The dictionary with "n" cardIAc sequences found and "m" other sequences
        found has the form:
        dict = {
            'cardiac' : { seq_1_ID:[firstNodeName,lastNodeName], ...,
                            seq_n_ID:[firstNodeName,lastNodeName] },
            'others' : { seq_1_ID:[firstNodeName,lastNodeName], ...,
                            seq_m_ID:[firstNodeName,lastNodeName] }
        }
        """

        seq_info = {'cardiac': {}, 'others': {}}

        # Get all sequences
        sequences = slicer.mrmlScene.GetNodesByClass('vtkMRMLSequenceNode')
        for i in range(sequences.GetNumberOfItems()):
            seq_i = sequences.GetItemAsObject(i)
            name = seq_i.GetName()
            # Get first and last data nodes names
            if seq_i.GetNumberOfDataNodes() == 0:  # empty sequence
                node1 = ''
                node2 = ''
            else:
                node1 = seq_i.GetNthDataNode(0).GetName()
                node2 = seq_i.GetNthDataNode(seq_i.GetNumberOfDataNodes() -
                                             1).GetName()

            in_cardiac = False
            for k in ['backg', 'backg_short', 'seg']:
                in_cardiac = in_cardiac or (
                    name in self._cardiac_sequences[k]['sequences'])

            if in_cardiac:
                seq_info['cardiac'][seq_i.GetID()] = [node1, node2]
            else:
                seq_info['others'][seq_i.GetID()] = [node1, node2]

        return seq_info

    def updateStatusBar(self, text):
        self.widgets.segmStatusIndicator.setText(text)
        slicer.util.resetSliceViews()
        slicer.app.processEvents()  # force update


class CardiacSegmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    """

    def setUp(self):
        """
        Do whatever is needed to reset the state - typically a scene clear will
        be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """
        Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_CardiacSegmentation1()

    def test_CardiacSegmentation1(self):
        """
        Ideally you should have several levels of tests. At the lowest level
        tests should exercise the functionality of the logic with different
        inputs (both valid and invalid). At higher levels your tests should
        emulate the way the user would interact with your code and confirm that
        it still works the way you intended. One of the most important features
        of the tests is that it should alert other developers when their
        changes will have an impact on the behavior of your module. For
        example, if a developer removes a feature that you depend on, your test
        should break so they know that the feature is needed.
        """

        self.delayDisplay('Starting the test')
        #
        # first, get some data
        #
        import SampleData
        SampleData.downloadFromURL(
            nodeNames='FA',
            fileNames='FA.nrrd',
            uris='http://slicer.kitware.com/midas3/download?items=5767')
        self.delayDisplay('Finished with download and loading')

        volumeNode = slicer.util.getNode(pattern='FA')
        logic = CardiacSegmentationLogic()
        self.assertIsNotNone(logic.hasImageData(volumeNode))
        self.delayDisplay('Test passed!')
