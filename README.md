# An Active Inference Model of Covert and Overt Visual Attention

This project introduces a model of covert and overt visual attention through the framework of active inference, utilizing dynamic optimization of sensory precisions to minimize free-energy. The model determines visual sensory precisions based on both current environmental beliefs and sensory input, influencing attentional allocation in both covert and overt modalities.

This research has been supported by the H2020 project AIFORS under Grant Agreement No 952275

## Requirements

ROS2 Humble Hawksbill
Python 3.10.xx
Numpy 1.26.4
OpenCV 4.5.4

## Simulation

To begin a simple instance of the active inference agent launch the file xxx
Usage:
    -**Enter** advances the simulation by one step
    -**s** sets the automatic step counter and advances the simulation by the given steps
    -**c** runs the simulation for the set step count

To start the auto trial node for the Posner paradigm, or the overt attention trial, launch the file xxx
Edit the launch file to set the following trial arguments:
    -*trials*: number of trials. Default is 1
    -*init*: step duration of the initialization phase. Default is 10
    -*cue*: step duration for the cue phase. Default is 50
    -*coa*: step duration for the cue-target onset asynchrony phase. Default is 100
    -*max*: maximum number of simulation steps after target onset. Default is 1000
    -*endo*: boolean value indicating if trial is endogenous. Default is True, set False for exogenous
    -*valid*: boolean value indicating if trial is valid. Default is True, set False for invalid
    -*act*: boolean value indicating if action is enabled in the trial. Default is False, set True for overt attention
