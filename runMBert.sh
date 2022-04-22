#!/bin/bash

mkdir ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0

for Source in english spanish french
    do
    for seed in 1 2 3 4 5
        do
        nohup python -u GCNModel.py --Source ${Source} --RecordsDir ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0/${Source}2Others${seed}.txt 2>&1
        done
    done