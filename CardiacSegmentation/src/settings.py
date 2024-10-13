"""
File: settings.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

import os
from pathlib import Path

# CNN paths
BASE_PATH = os.path.realpath(__file__).split(
    os.path.basename(os.path.realpath(__file__)))[0]
BASE_PATH = Path(BASE_PATH).parent
MODEL_PATH = "pretrained_models"
CENTER_MODEL_NAME = "center_detection_net_v0.h5"
SEG_MODEL_NAME = "rlv_detection_net_v0.h5"

CENTER_MODEL_PATH = os.path.join(BASE_PATH, MODEL_PATH, CENTER_MODEL_NAME)
SEG_MODEL_PATH = os.path.join(BASE_PATH, MODEL_PATH, SEG_MODEL_NAME)

# Model input shape
CENTER_MODEL_INPUT_SHAPE = (64, 64)
SEG_MODEL_INPUT_SHAPE = (128, 128)

# Logic initial conditions
LOGIC_IC = {
    'segmStatus': -1,
    'backg_name_default': 'cardIAc_img',
    'seg_name_default': 'cardIAc_seg',
    'backg_name_short': 'cardIAc_short',
    'cnn_input_label': 'unet_input',
    'cnn_output_label': 'gvae_output',
    'cnn_roi_size': 90
}

# Tissues Settings
RV = 1
MYO = 2
LV = 3
LABEL_TISSUES = {'rv': RV, 'myo': MYO, 'lv': LV}
