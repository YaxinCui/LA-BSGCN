#!/bin/zsh


mkdir -p ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3/
nohup python -u GCNModel.py --Source french --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//french2Others1.txt 2>&1
nohup python -u GCNModel.py --Source french --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//french2Others2.txt 2>&1
nohup python -u GCNModel.py --Source french --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//french2Others3.txt 2>&1


nohup python -u GCNModel.py --Source english --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//english2Others1.txt 2>&1
nohup python -u GCNModel.py --Source english --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//english2Others2.txt 2>&1
nohup python -u GCNModel.py --Source english --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//english2Others3.txt 2>&1

nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//spanish2Others1.txt 2>&1
nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//spanish2Others2.txt 2>&1
nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 0 --GCNLayerNum 0 --NEREmbedDim 0 --POSEmbedDim 0 > ./RecordsDir/GCNEmb_0\|GCNNum_0\|NERDim_0\|POSDim_0\|Mode_3//spanish2Others3.txt 2>&1

