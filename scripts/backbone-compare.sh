#!/usr/bin/env bash

source activate py36
cd ..

python3.6 train.py -s -ag -record --source ../data/video-dig/auto-crop-intensity-none --dataset folder --num_class 2 --net typical --backbone seincept --epoch 250 --data e --lr 0.005 --batch 32 --loss CCE --loss_weight 0.66 --remark 1
python3.6 train.py -s -ag -record --source ../data/video-dig/auto-crop-intensity-pca --dataset folder --num_class 2 --net typical --backbone seincept --epoch 250 --data e --lr 0.005 --batch 32 --loss CCE --loss_weight 0.66 --remark 1
python3.6 train.py -s -ag -record --source ../data/video-dig/auto-crop-none-none --dataset folder --num_class 2 --net typical --backbone seincept --epoch 250 --data e --lr 0.005 --batch 32 --loss CCE --loss_weight 0.66 --remark 1
python3.6 train.py -s -ag -record --source ../data/video-dig/auto-crop-none-pca --dataset folder --num_class 2 --net typical --backbone seincept --epoch 250 --data e --lr 0.005 --batch 32 --loss CCE --loss_weight 0.66 --remark 1
python3.6 train.py -s -ag -record --source ../data/video-dig/auto-roi-intensity-none --dataset folder --num_class 2 --net typical --backbone seincept --epoch 250 --data e --lr 0.005 --batch 32 --loss CCE --loss_weight 0.66 --remark 1
python3.6 train.py -s -ag -record --source ../data/video-dig/auto-roi-intensity-pca --dataset folder --num_class 2 --net typical --backbone seincept --epoch 250 --data e --lr 0.005 --batch 32 --loss CCE --loss_weight 0.66 --remark 1
python3.6 train.py -s -ag -record --source ../data/video-dig/auto-roi-none-none --dataset folder --num_class 2 --net typical --backbone seincept --epoch 250 --data e --lr 0.005 --batch 32 --loss CCE --loss_weight 0.66 --remark 1
python3.6 train.py -s -ag -record --source --source ../data/video-dig/auto-roi-none-pca --dataset folder --num_class 2 --net typical --backbone seincept --epoch 250 --data e --lr 0.005 --batch 32 --loss CCE --loss_weight 0.66 --remark 1
