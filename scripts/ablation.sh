#!/usr/bin/env bash

cd ..
python union_train.py --data ./conf/train/compare/Ablation-study-F.yaml
python union_train.py --data ./conf/train/compare/Ablation-study-E.yaml
