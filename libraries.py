# load required libraries
print('Loading libraries...')
IS_COLAB = False

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import math
from PIL import Image
from pathlib import Path
import tensorflow as tf
from keras import backend as K
from tensorflow import keras
from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Reshape, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, ReLU, Concatenate, Add, UpSampling2D, Input
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras import Model
from keras import layers as Layers
from keras.initializers import RandomNormal
tf.config.run_functions_eagerly(True)
from sklearn.model_selection import train_test_split
import time
import copy
import random
import requests
import zipfile
import gzip
import shutil

CWD = os.getcwd()

if IS_COLAB:
    from pydrive.auth import GoogleAuth
    from google.colab import drive
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials


print('Loaded libraries successfuly!')