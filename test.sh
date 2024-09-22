#!/bin/bash

files=("test/test_route_planner.py" "test/test_controller.py")
configs=("conf/map_easy.json")

for file in ${files[@]}; do
    for config in ${configs[@]}; do
        python $file --conf $config

        file_name=`basename $file .py`
        config_name=`basename $config .json`
        rm -rf results/$file_name/$config_name
        mkdir -p results/$file_name/$config_name
        find results/$file_name/ -name "*.png" -exec mv {} results/$file_name/$config_name/ \;
    done
done