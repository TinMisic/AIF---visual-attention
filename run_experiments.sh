#!/bin/bash

# List of commands to run consecutively
commands=(
    "echo 'Experiment 1: Endogenous Valid' && ros2 run aif_model auto_trial --trials 100 --init 10 --cue 50 --coa 600 --max 1000 --endo true --valid true --action false"
    "echo 'Experiment 2: Endogenous Invalid' && ros2 run aif_model auto_trial --trials 100 --init 10 --cue 50 --coa 600 --max 1000 --endo true --valid flase --action false"
    "echo 'Experiment 3: Exogenous Valid' && ros2 run aif_model auto_trial --trials 100 --init 10 --cue 50 --coa 600 --max 1000 --endo false --valid true --action false"
    "echo 'Experiment 4: Exogenous Invalid' && ros2 run aif_model auto_trial --trials 100 --init 10 --cue 50 --coa 600 --max 1000 --endo false --valid false --action false"
)

# Run each command sequentially
for cmd in "${commands[@]}"; do
    eval "$cmd"

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Command failed - $cmd"
        # exit 1  # Exit on failure
    fi
done
echo "All experiments executed successfully!"

