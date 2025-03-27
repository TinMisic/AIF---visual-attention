# An Active Inference Model of Covert and Overt Visual Attention

This project introduces a model of covert and overt visual attention through the framework of active inference, utilizing dynamic optimization of sensory precisions to minimize free-energy. The model determines visual sensory precisions based on both current environmental beliefs and sensory input, influencing attentional allocation in both covert and overt modalities.

**This research has been supported by the H2020 project AIFORS under Grant Agreement No 952275**

## Requirements

ROS2 Humble Hawksbill

Python 3.10.xx

Numpy 1.26.4

OpenCV 4.5.4

Gazebo SImulator 11.10.2


## Simulation

To begin a simple instance of the active inference agent run the following 4 commands in seperate terminals:

1. `ros2 launch gazebo_ros gazebo.launch.py world:=worlds/static_test.world`
2. `ros2 run camera_orientation turn_cam`
3. `ros2 topic pub /needs std_msgs/msg/Float32MultiArray "{data: [0.0, 0.0,1.0]}"` Note: edit the first two elements for the cue position, and the third for cue strength
4. `ros2 run aif_model act_inf`

Usage:<br/>
    -**Enter** advances the simulation by one step<br/> 
    -**s** sets the automatic step counter and advances the simulation by the given steps<br/>
    -**c** runs the simulation for the set step count<br/>

To start the auto trial node for the Posner paradigm, or the overt attention trial, run the following 3 commands in seperate terminals:

1. `ros2 launch gazebo_ros gazebo.launch.py world:=worlds/static_test.world`
2. `ros2 run camera_orientation turn_cam`
3. `ros2 run aif_model auto_trial`

You can edit the following arguments for the last command:<br/>
    -*trials*: number of trials. Default is 1<br/>
    -*init*: step duration of the initialization phase. Default is 10<br/>
    -*cue*: step duration for the cue phase. Default is 50<br/>
    -*coa*: step duration for the cue-target onset asynchrony phase. Default is 100<br/>
    -*max*: maximum number of simulation steps after target onset. Default is 1000<br/>
    -*endo*: boolean value indicating if trial is endogenous. Default is True, set False for exogenous<br/>
    -*valid*: boolean value indicating if trial is valid. Default is True, set False for invalid<br/>
    -*act*: boolean value indicating if action is enabled in the trial. Default is False, set True for overt attention<br/>
