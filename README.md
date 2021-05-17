## ROS2 End-to-End Learning Model with SVL Simulator

References: https://www.svlsimulator.com/docs/

### Prerequisites
* Docker installed
* Atleast 8GB of Graphic card

### Instructions:
* `git clone --recurse-submodules https://github.com/nitesh4146/lanefollowing.git`
* `docker pull lgsvl/lanefollowing:latest`
* `cd lanefollowing/ros2_ws/src/lane_following/`
* `git clone https://github.com/nitesh4146/bumblebee.git`

* To build ROS2 packages go back to the root lanefollowing directory and build ros:  
`docker-compose up build_ros`

* Collect Data (The data will be recorded in bumblebee/data directory)  
`docker-compose up collect`  
and run the simulator

* Move the data into corresponding maps directory before training  

* Train your model  

* Drive using trained model (trained model must be stored in bumblebe/model directory)  
`docker-compose up drive`  
and run the simulator



### Directory Tree:   

lanefollowing  
│ └── ros2_ws  
│   └── src  
│       ├── lane_following  
│       │   ├── bumblebee  
│       │   │   ├── hdf5  
│       │   │   ├── model  
│       │   │   ├── README.md  
│       │   │   ├── data_loader.py  
│       │   │   ├── preprocess.py  
│       │   │   └── trainer.py  
│       │   │   ├── images  
│       │   │   │   └── bee.png  
│       │   │   ├── __init__.py  
│       │   │   ├── maps  
│       │   │   │   ├── Map1  
│       │   │   │   └── Map2  
│       │   ├── collect.py  
│       │   ├── drive.py  
│       │   ├── __init__.py  
│       │   ├── package.xml  
│       │   ├── params  
│       │   │   ├── collect_params.yaml  
│       │   │   ├── drive_params.yaml  
│       │   │   └── drive_visual_params.yaml  

