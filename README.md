# CubeSLAM

This repository is a fork of the official [CubeSLAM implementation](https://github.com/shichaoy/cube_slam) focusing on monocular SLAM _without_ ROS integration.

**CubeSLAM: Monocular 3D Object SLAM**, IEEE Transactions on Robotics 2019, S. Yang, S. Scherer  [**PDF**](https://arxiv.org/abs/1806.00557)

## Installation

### Requirements
CubeSLAM requires the following depedencies, versions included are those used for testing:
- OpenCV 3.2
- Eigen3
- [Pangolin](https://github.com/stevenlovegrove/Pangolin)
- PCL 1.2

### Compile
Compile the whole project using the CMake file in the root folder
```bash
mkdir build
cd build
cmake ..
make -j4 # Adjust CPU count as needed
```

## Data Preparation
Each scene to be processed must be fully contained in a folder structured as follow:
```
SCENE_FOLDER
  |-data
  |   |-bb_tracking
  |   |-vehicle_segmpas
  |-scenes
      |-times.txt
      |-images
      |-000000_right.png
```

`images` contains one image per frame of the sequence named `[frame_id].png` where `[frame_id` is a six-digit, 0-padded frame number, 
`times.txt` contains the timestamps of the frames in seconds, one per line, `000000_right.png` is the first frame captured from the right camera used for initialization.

`vehicle_segmpas` contains a 8-bit grayscale PNG image for every frame (named as the corresponding frame) containing a tracked vehicle. Each image if fully black with the exception of the pixel belonging to a vehicle,
which are assigned a color corresponding to the ID of the vehicle, hence the maximum number of vehicles handled per scene is 255 (it can be increased by using 16-bit images). 
`bb_tracking` contains a text file for every frame (named as the corresponding frame but with `.txt` extension) containing a tracked vehicle. 
It contains a line for each detected bounding box consisting of 7 fields: class, top-left corner, width and height, confidence score and vehicle ID. 
This data can be obtained using Mask RCNN with the exception of vehicle tracking.

## Running
Run CubeSLAM with
```bash
cube_slam SCENE_FOLDER/scenes SCENE_FOLDER/data PROJECT_ROOT/data/ORBvoc.bin SETTINGS_PATH [OTHER_SETTINGS_PATH...]
```
`cube_slam` should be in `PROJECT_ROOT/build/bin` and `PROJECT_ROOT` is the root folder of this repository.
Any number of settings files can be provided and duplicated parameters overwrite those in previous files, additionally individual settings can be overwritten
from the CLI by passing additional parameters in the form `PARAMETER=VALUE`. The settings for the KITTI setup are provided in `data/KITTI_mono.yaml`.
