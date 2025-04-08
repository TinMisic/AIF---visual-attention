# An Active Inference Model of Covert and Overt Visual Attention

This project introduces a model of covert and overt visual attention through the framework of active inference, utilizing dynamic optimization of sensory precisions to minimize free-energy. The model determines visual sensory precisions based on both current environmental beliefs and sensory input, influencing attentional allocation in both covert and overt modalities. The project includes simulations of the Posner cueing task and a simple object tracking task.

**This research has been supported by the H2020 project AIFORS under Grant Agreement No 952275**

## Video examples

The visual sensory input is on the left, and the visual prediction with target (red arrow) and covert focus (green dot) beliefs is on the right. 

### The Posner cueing task

#### Endogenous cueing

The model is cued endogenously for 50 steps, and after a cue-target onset asynchrony period of 100 steps the target is shown.

<iframe width="560" height="315" src="https://www.youtube.com/embed/0CwjfhYo5eQ?si=ZPXYmipHABjOFr_k" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

#### Exogenous cueing

The model is cued exogenously for 50 steps, and after a cue-target onset asynchrony period of 100 steps the target is shown.

<iframe width="560" height="315" src="https://www.youtube.com/embed/KfKeA_bZ7O0?si=wMffTdiBdPnscePg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Overt attention - action enabled

 The model is cued is presented a static object and moves the camera to focus it in the center of the image.

 <iframe width="560" height="315" src="https://www.youtube.com/embed/Ay33udKygOo?si=TrSUHusLSf8WNw4p" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Simulation

### Active inference agent demo

To begin a simple instance of the active inference agent run the following 4 commands in seperate terminals:

1. `ros2 launch gazebo_ros gazebo.launch.py world:=worlds/static_test.world`
2. `ros2 run camera_orientation turn_cam`
3. `ros2 topic pub /needs std_msgs/msg/Float32MultiArray "{data: [0.0, 0.0,1.0]}"` Note: edit the first two elements for the cue position, and the third for cue strength
4. `ros2 run aif_model act_inf`

Usage:<br/>
    -**Enter** advances the simulation by one step<br/> 
    -**s** sets the automatic step counter and advances the simulation by the given steps<br/>
    -**c** runs the simulation for the set step count<br/>

### Automatic trials for tasks

To start the auto trial node for the Posner paradigm, or the overt attention trial, run the following 3 commands in seperate terminals:

1. `ros2 launch gazebo_ros gazebo.launch.py world:=worlds/static_test.world`
2. `ros2 run camera_orientation turn_cam`
3. `ros2 run aif_model auto_trial`

You can edit the following arguments for the last command:<br/>
    -__*trials*__: number of trials. Default is 1<br/>
    -__*init*__: step duration of the initialization phase. Default is 10<br/>
    -__*cue*__: step duration for the cue phase. Default is 50<br/>
    -__*coa*__: step duration for the cue-target onset asynchrony phase. Default is 100<br/>
    -__*max*__: maximum number of simulation steps after target onset. Default is 1000<br/>
    -__*endo*__: boolean value indicating if trial is endogenous. Default is True, set False for exogenous<br/>
    -__*valid*__: boolean value indicating if trial is valid. Default is True, set False for invalid<br/>
    -__*act*__: boolean value indicating if action is enabled in the trial. Default is False, set True for overt attention<br/>

## Requirements

ROS2 Humble Hawksbill

Python 3.10.xx

Numpy 1.26.4

OpenCV 4.5.4

Gazebo Simulator 11.10.2
