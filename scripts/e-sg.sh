#!/usr/bin/env bash

source activate py36
cd ..

python train.py -s -ag -record --source ../data/new/crop_E --dataset mix --num_class 2 --net mixnet --backbone unet --epoch 200 --data e --lr 0.001 --batch 8 --loss dice

python train.py -s -ag -record --source ../data/new/crop_E --dataset mix --num_class 2 --net mixnet --backbone r2attunet --epoch 200 --data e --lr 0.001 --batch 8 --loss dice

python train.py -s -ag -record --source ../data/new/crop_E --dataset mix --num_class 2 --net mixnet --backbone r2unet --epoch 200 --data e --lr 0.001 --batch 8 --loss dice

python train.py -s -ag -record --source ../data/new/crop_E --dataset mix --num_class 2 --net mixnet --backbone attunet --epoch 200 --data e --lr 0.001 --batch 8 --loss dice

