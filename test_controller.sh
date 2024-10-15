#!/bin/bash

# Define the list of Python files and configurations
files=("test/test_controller.py")
configs=("conf/map_easy.json" "conf/map_easy_dynamic.json" "conf/map_custom.json" "conf/map_slam.json")

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
        if [ $config_name == "map_easy_dynamic" ]; then
            $run --dynamic
        elif [ $config_name == "map_slam" ]; then
            $run --map image_grid 
        else
            $run 
        fi
        
        # Move any .png files from results/test_controller to the output directory if they exist
        png_files="results/test_controller/*.png"
        if ls $png_files 1> /dev/null 2>&1; then
            mv results/test_controller/*.png "$output_dir"
        fi
    done
done