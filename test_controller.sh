#!/bin/bash

# Define the list of Python files and configurations
files=("test/test_controller.py")
configs=("conf/map_easy.json" "conf/map_custom.json" "conf/map_slam.json")

for file in "${files[@]}"; do
    for config in "${configs[@]}"; do
        # Extract the base names for file and config
        file_name=$(basename "$file" .py)
        config_name=$(basename "$config" .json)

        # Define the output directory, including the current timestamp
        current_time=$(date +"%Y%m%d_%H%M%S")
        output_dir="results/$file_name/$config_name/$current_time"

        # Create the output directory
        mkdir -p "$output_dir"

        # Run the Python script with the specified configuration and output directory
        run="python $file --conf $config --output_dir $output_dir"
        if [ $config_name == "map_easy" ]; then
            $run --dynamic
        elif [ $config_name == "map_slam" ]; then
            $run --map image_grid 
        else
            $run 
        fi
    done
done