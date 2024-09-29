#!/bin/bash

files=("test/test_controller.py")
configs=("conf/map_easy.json" "conf/map_medium.json")

for file in ${files[@]}; do
    for config in ${configs[@]}; do
        file_name=`basename $file .py`
        config_name=`basename $config .json`
        rm -rf results/$file_name/$config_name
        mkdir -p results/$file_name/$config_name

        python $file --conf $config

        # find 명령어 결과를 배열로 안전하게 처리
        IFS=$'\n' png_files=($(find results/$file_name/ -name "*.png"))

        # png 파일이 있는 경우에만 이동 시도
        if [ ${#png_files[@]} -gt 0 ]; then
            for png_file in "${png_files[@]}"; do
                mv "$png_file" results/$file_name/$config_name/
            done
        fi
    done
done