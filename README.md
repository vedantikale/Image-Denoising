# Image-Denoising
## Introduction

**Zero-Reference Deep Curve Estimation** or **Zero-DCE** formulates low-light image
enhancement as the task of estimating an image-specific
[*tonal curve*](https://en.wikipedia.org/wiki/Curve_(tonality)) with a deep neural network.
In this example, we train a lightweight deep network, **DCE-Net**, to estimate
pixel-wise and high-order tonal curves for dynamic range adjustment of a given image.

Zero-DCE takes a low-light image as input and produces high-order tonal curves as its output.
These curves are then used for pixel-wise adjustment on the dynamic range of the input to
obtain an enhanced image. The curve estimation process is done in such a way that it maintains
the range of the enhanced image and preserves the contrast of neighboring pixels. This
curve estimation is inspired by curves adjustment used in photo editing software such as
Adobe Photoshop where users can adjust points throughout an image’s tonal range.

Zero-DCE is appealing because of its relaxed assumptions with regard to reference images:
it does not require any input/output image pairs during training.
This is achieved through a set of carefully formulated non-reference loss functions,
which implicitly measure the enhancement quality and guide the training of the network.

## Downloading LOLDataset

The **LoL Dataset** has been created for low-light image enhancement. It provides 485
images for training and 15 for testing. Each image pair in the dataset consists of a
low-light input image and its corresponding well-exposed reference image.
"""
import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.layers import Input,Conv2D,Concatenate
from keras.models import Model
import tensorflow as tf

We use 300 low-light images from the LoL Dataset training set for training, and we use
the remaining 185 low-light images for validation. We resize the images to size `256 x
256` to be used for both training and validation. Note that in order to train the DCE-Net,
we will not require the corresponding enhanced images.
"""


