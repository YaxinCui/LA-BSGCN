#!/bin/zsh

# en->es

mkdir -p ./RecordsDir/GCNEmb_256\|GCNNum_5\|NERDim_6\|POSDim_9\|Mode_3/
nohup python -u GCNModel.py --Source english --GCNEmbedDim 256 --GCNLayerNum 5 --NEREmbedDim 6 --POSEmbedDim 9 > ./RecordsDir/GCNEmb_256\|GCNNum_5\|NERDim_6\|POSDim_9\|Mode_3//english2Others1.txt 2>&1
nohup python -u GCNModel.py --Source english --GCNEmbedDim 256 --GCNLayerNum 5 --NEREmbedDim 6 --POSEmbedDim 9 > ./RecordsDir/GCNEmb_256\|GCNNum_5\|NERDim_6\|POSDim_9\|Mode_3//english2Others2.txt 2>&1
nohup python -u GCNModel.py --Source english --GCNEmbedDim 256 --GCNLayerNum 5 --NEREmbedDim 6 --POSEmbedDim 9 > ./RecordsDir/GCNEmb_256\|GCNNum_5\|NERDim_6\|POSDim_9\|Mode_3//english2Others3.txt 2>&1

# en->fr
mkdir -p ./RecordsDir/GCNEmb_192\|GCNNum_4\|NERDim_14\|POSDim_5\|Mode_3/
nohup python -u GCNModel.py --Source english --GCNEmbedDim 192 --GCNLayerNum 4 --NEREmbedDim 14 --POSEmbedDim 5 > ./RecordsDir/GCNEmb_192\|GCNNum_4\|NERDim_14\|POSDim_5\|Mode_3//english2Others1.txt 2>&1
nohup python -u GCNModel.py --Source english --GCNEmbedDim 192 --GCNLayerNum 4 --NEREmbedDim 14 --POSEmbedDim 5 > ./RecordsDir/GCNEmb_192\|GCNNum_4\|NERDim_14\|POSDim_5\|Mode_3//english2Others2.txt 2>&1
nohup python -u GCNModel.py --Source english --GCNEmbedDim 192 --GCNLayerNum 4 --NEREmbedDim 14 --POSEmbedDim 5 > ./RecordsDir/GCNEmb_192\|GCNNum_4\|NERDim_14\|POSDim_5\|Mode_3//english2Others3.txt 2>&1


# es->en

mkdir -p ./RecordsDir/GCNEmb_96\|GCNNum_5\|NERDim_12\|POSDim_5\|Mode_3/
nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 96 --GCNLayerNum 5 --NEREmbedDim 12 --POSEmbedDim 5 > ./RecordsDir/GCNEmb_96\|GCNNum_5\|NERDim_12\|POSDim_5\|Mode_3//spanish2Others1.txt 2>&1
nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 96 --GCNLayerNum 5 --NEREmbedDim 12 --POSEmbedDim 5 > ./RecordsDir/GCNEmb_96\|GCNNum_5\|NERDim_12\|POSDim_5\|Mode_3//spanish2Others2.txt 2>&1
nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 96 --GCNLayerNum 5 --NEREmbedDim 12 --POSEmbedDim 5 > ./RecordsDir/GCNEmb_96\|GCNNum_5\|NERDim_12\|POSDim_5\|Mode_3//spanish2Others3.txt 2>&1


# es->fr

mkdir -p ./RecordsDir/GCNEmb_192\|GCNNum_5\|NERDim_8\|POSDim_3\|Mode_3/
nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 192 --GCNLayerNum 5 --NEREmbedDim 8 --POSEmbedDim 3 > ./RecordsDir/GCNEmb_192\|GCNNum_5\|NERDim_8\|POSDim_3\|Mode_3//spanish2Others1.txt 2>&1
nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 192 --GCNLayerNum 5 --NEREmbedDim 8 --POSEmbedDim 3 > ./RecordsDir/GCNEmb_192\|GCNNum_5\|NERDim_8\|POSDim_3\|Mode_3//spanish2Others2.txt 2>&1
nohup python -u GCNModel.py --Source spanish --GCNEmbedDim 192 --GCNLayerNum 5 --NEREmbedDim 8 --POSEmbedDim 3 > ./RecordsDir/GCNEmb_192\|GCNNum_5\|NERDim_8\|POSDim_3\|Mode_3//spanish2Others3.txt 2>&1


# fr->en

mkdir -p ./RecordsDir/GCNEmb_384\|GCNNum_2\|NERDim_8\|POSDim_3\|Mode_3/
nohup python -u GCNModel.py --Source french --GCNEmbedDim 384 --GCNLayerNum 2 --NEREmbedDim 8 --POSEmbedDim 3 > ./RecordsDir/GCNEmb_384\|GCNNum_2\|NERDim_8\|POSDim_3\|Mode_3//french2Others1.txt 2>&1
nohup python -u GCNModel.py --Source french --GCNEmbedDim 384 --GCNLayerNum 2 --NEREmbedDim 8 --POSEmbedDim 3 > ./RecordsDir/GCNEmb_384\|GCNNum_2\|NERDim_8\|POSDim_3\|Mode_3//french2Others2.txt 2>&1
nohup python -u GCNModel.py --Source french --GCNEmbedDim 384 --GCNLayerNum 2 --NEREmbedDim 8 --POSEmbedDim 3 > ./RecordsDir/GCNEmb_384\|GCNNum_2\|NERDim_8\|POSDim_3\|Mode_3//french2Others3.txt 2>&1

# fr->es

mkdir -p ./RecordsDir/GCNEmb_320\|GCNNum_3\|NERDim_6\|POSDim_9\|Mode_3/
nohup python -u GCNModel.py --Source french --GCNEmbedDim 320 --GCNLayerNum 3 --NEREmbedDim 6 --POSEmbedDim 9 > ./RecordsDir/GCNEmb_320\|GCNNum_3\|NERDim_6\|POSDim_9\|Mode_3//french2Others1.txt 2>&1
nohup python -u GCNModel.py --Source french --GCNEmbedDim 320 --GCNLayerNum 3 --NEREmbedDim 6 --POSEmbedDim 9 > ./RecordsDir/GCNEmb_320\|GCNNum_3\|NERDim_6\|POSDim_9\|Mode_3//french2Others2.txt 2>&1
nohup python -u GCNModel.py --Source french --GCNEmbedDim 320 --GCNLayerNum 3 --NEREmbedDim 6 --POSEmbedDim 9 > ./RecordsDir/GCNEmb_320\|GCNNum_3\|NERDim_6\|POSDim_9\|Mode_3//french2Others3.txt 2>&1
