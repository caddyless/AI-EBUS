#!/usr/bin/env bash

source activate py36
cd ..

CUDA_VISIBLE_DEVICES=1 python3.6 pretrain.py -s -ag -resume -record --source ../data/new/crop_E --dataset folder --num_class 2 --net typical --backbone seincept --epoch 100 --data imagenet --num_class 1000 --lr 0.01 --batch 128 --loss CCE
