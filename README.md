# Latency-generation-for-safe-critical-scenario
This repo transforms the hdf5 dataset generated in Carla City Simulator(>=0.9.10) into kitti formmatted data structure to better understand the effect of lantency in multiagent communication. The generated data can be directly used in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) model training.  

## Requirements
Before you get started, please install the following dependencies:
- python 3.x
- h5py
- numpy
- mayavi >= 4.72
- pyqt >5
- Download the example daatset folder [here](https://drive.google.com/file/d/1vnifcFfoUZfJCyd7sK0ep1lUb5M9V__o/view?usp=share_link)

## Demo 
The following gif displays a clear perception of surrounding environment with fused point cloud and ground truth bounding box without any latency.

![Scenario without Latency](without_lag.gif)

One of the example below shows the generated scenario if we assign 80 percent of the car 60 percent more of latency comparing to the groundtruth ones. As you may notice, the shape of some cars are outside of the bounding box, and some cars overlap which will make a misleading sign of some traffic accident

![Scenario with Latency](with_lag.gif)

## Data structure
The ground truth data of Carla City Simulation is under the following setup.
 - fps 10 
 - frames(/scenario) 100
 - number of vehicles 8 - 13
 - number of pedestrain 4/5 
 - range of lidar 100m
 - Carla City Map chosen 1,2,3,5,10

And the generated file folder will be under this structure.

- kitti
  - ImageSets
  - training
    - calib & velodyne & label_2 & image_2
  - testing
    - calib & velodyne & image_2


## Get Started 
Example comman for generation: 
```
python genDelat.py --filename m1v9p4.hdf5 --datatype training --count_60 80
```
filname chooses the ground truth file generate from the Carla City Simulator
datatype decides where the data will fall into in the structure above 
count_(20~100) the percentage of cars that will be affected by the latency in the scenario and it could be a combination of cars(eg. 40% of cars under 40% more latency and 20% of cars under 80% more latency)
