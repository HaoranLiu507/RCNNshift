# RCNNshift for Moving Object Tracking

## Introduction

This repository presents an implementation of RCNNshift on PyTorch, leveraging the powerful computational capabilities of the Graphics Processing Unit (GPU) to significantly hasten the batch processing of feature extraction from frames in a video. RCNNshift is derived from the widely recognized meanshift object tracking algorithm, incorporating a Random-coupled Neural Network (RCNN) or a Three-Dimensional Random-coupled Neural Network (3DRCNN) to extract features from a video and generate an 'ignition map' for each frame. These ignition maps are utilized as additional fourth-dimensional information alongside the RGB channels of the video, enabling meanshift tracking in this enhanced feature space. Additionally, to facilitate a comparative analysis with the original meanshift algorithm, this repository integrates OpenCV-based meanshift algorithms operating in the RGB and HSV color spaces. For a detailed exposition of the repository, comprising the RCNN architecture, feature extraction process, and a thorough evaluation, please refer to our research paper available at the following DOI: Not published yet.


![My GIF](demo.gif)


## Dataset

This project uses the TB-50 dataset for tracking demonstration. Place the folder containing the videos to be tracked in the same directory as 'main.py', modify 'Your video path' in 'main.py' to the folder name, and change 'video name' to the name of the video to be tracked within the folder. Note that the TB-50 dataset is not included in this repository, and videos are all in '.mp4' format.

Please download the dataset from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12526486.svg)](https://doi.org/10.5281/zenodo.12526486)

```
# Video path and name
path = os.path.abspath('Your video path')
name = 'video name'
```

## Requirements

To run this project, you will need to have the following libraries installed: matplotlib and PyTorch.


```
# To install numpy, run the following code:
pip install numpy

# To install numpy, run the following code:
pip install opencv-python

# To ensure that Torch is installed correctly, make sure CUDA is avaliable on your device
# Visit CUDA's website, choose the right version for your device (https://developer.nvidia.com/cuda-downloads)

# Then, install the corresponding version of PyTorch
# Visit the PyTorch's website and getting install command (https://pytorch.org/)
# For example, 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
The development environment for this project is python 3.11, cuda 11.8, and torch 2.1.0, and newer versions are expected to work as well.

## Files

This repository contains three Python files:

- `main.py`: This file is the main script for the project, which reads the video, extracts features, and performs tracking using the RCNNshift class.
- `RCNNshift.py`: This file implements a class named RCNNshift.
- `main_track_in_batch.py`: This file performs tracking in batch mode. All videos in the folder will be tracked in sequence. Put the ground truth of the location of ROI of every video in the folder *GroundTruth*, and the tracking results will be compared with the ground truth to calculate the tracking accuracy.

## Usage

To use this code, follow these steps:

1. Download or clone this repository.

2. Open the terminal and navigate to the project's root directory.

3. Set video path and tracking parameters, and run the following command to initiate moving object tracking:

   ```
   python main.py
   ```

4. The program will wait for your input of the location of ROI in the first frame if 'select_rect' is set to 'input'. Otherwise, if 'select_rect' is set to 'mouse', a window will pop out and wait for your ROI selection.

5. The program will output the location of ROI in each frame to the folder *TrackWindowResult*.

6. If 'perform' is set to 'local', the program will output the tracking result to the folder *TrackedVideo*.

Have fun with the RCNNshift :)

If you encounter any problems, please feel free to contact us via *Github Issues*, or simply via email: *liuhaoran@cdut.edu.cn*
    
