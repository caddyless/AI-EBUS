#!/usr/bin/env bash

source activate py36
cd ..

CUDA_VISIBLE_DEVICES=0 python3.6 k-means.py -s --source ../data/video-dig/auto-selection-e --dataset video --backbone seincept
#CUDA_VISIBLE_DEVICES=0 python3.6 k-means.py -s --source ../data/video-dig/mix-E --dataset video --backbone seincept
