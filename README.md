## Implementation & Evaluation of End-to-End Learning for Autonomous Vehicles using SVL

### End-to-end project directory: `EECS-753-Project/ros2_ws/src/lane_following/e2e/`

### Prerequisites
* Docker installed
* Atleast 8GB of Graphic card

### Instructions:
* `https://github.com/nitesh4146/EECS-753-Project.git`
* `docker pull lgsvl/lanefollowing:latest`
* `cd lanefollowing/ros2_ws/src/lane_following/`

* To build ROS2 packages go back to the root lanefollowing directory and build ros:  
`docker-compose up build_ros`

* Collect Data (The data will be recorded in e2e/data directory)  
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
│       │   ├── e2e  
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


References: https://www.svlsimulator.com/docs/
