# coding: utf-8
# jupyter nbconvert --to script BaseModel.ipynb

import torch
import torch.utils.data as data


from ArgumentParser.GCNArgParser import GCNArgumentParser
from Dataset.SemEval16Task5Dataset import SemEvalXMLDataset

argument = GCNArgumentParser()
dataParams = argument.parse_args()

dataParams.PretrainModel="bert-base-multilingual-cased"
# Dataset
trainSourceDataset = SemEvalXMLDataset(phrase="Train", language=dataParams.Source)

trainDataset, trialDataset = data.random_split(trainSourceDataset, [int(len(trainSourceDataset)*0.9), len(trainSourceDataset)-int(len(trainSourceDataset)*0.9)])

testEnDataset = SemEvalXMLDataset(phrase="Test", language="english")
testEsDataset = SemEvalXMLDataset(phrase="Test", language="spanish")
testFrDataset = SemEvalXMLDataset(phrase="Test", language="french")

# DataLoader
from CollateFn.CollateFnXLMRGCN import CollateFnXLMRGCN
collateFnXLMRGCN = CollateFnXLMRGCN(pretrained_model_name_or_path=dataParams.PretrainModel)
#collateFnXLMRGCN = CollateFnXLMRGCN(pretrained_model_name_or_path="bert-base-multilingual-cased")

from torch.utils.data import DataLoader

trainDataLoader = DataLoader(trainDataset, batch_size=dataParams.Batchsize, collate_fn=collateFnXLMRGCN.collate_fn, shuffle=False, drop_last=False)
trialDataLoader = DataLoader(trialDataset, batch_size=dataParams.Batchsize, collate_fn=collateFnXLMRGCN.collate_fn, shuffle=False, drop_last=False)

testEnDataLoader = DataLoader(testEnDataset, batch_size=dataParams.Batchsize, collate_fn=collateFnXLMRGCN.collate_fn, shuffle=False, drop_last=False)
testEsDataLoader = DataLoader(testEsDataset, batch_size=dataParams.Batchsize, collate_fn=collateFnXLMRGCN.collate_fn, shuffle=False, drop_last=False)
testFrDataLoader = DataLoader(testFrDataset, batch_size=dataParams.Batchsize, collate_fn=collateFnXLMRGCN.collate_fn, shuffle=False, drop_last=False)

dataLoader = {
    'train': trainDataLoader,
    'trial': trialDataLoader,
    'testEn': testEnDataLoader,
    'testEs': testEsDataLoader,
    'testFr': testFrDataLoader    
}

from Model.GCNModel import GCNModel
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNModel(gcn_mode=dataParams.GCNModelMode, gcn_dim=dataParams.GCNEmbedDim, gcn_num=dataParams.GCNLayerNum, stanzaTokenNerDim=dataParams.NEREmbedDim, stanzaTokenPosDim=dataParams.POSEmbedDim, pretrained_model=dataParams.PretrainModel).to(DEVICE)

# 优化层
from torch import optim
from torch.optim import lr_scheduler

optimizer = optim.Adam(model.parameters(), lr=dataParams.LearningRate)
scheduler = lr_scheduler.StepLR(optimizer, 4, gamma=0.7)

from torch.nn import NLLLoss
criterion = NLLLoss()

from ModelSummary.ModelOutputsRecord import ModelOutputsRecord

modelOutputsRecord = ModelOutputsRecord(dataParams = dataParams, phrases=['train', 'trial', 'testEn', 'testEs', 'testFr'])
from ModelSummary.EvalATE import EvalATE

from Model.GCNModelRun import GCNModelRun
for epoch in range(dataParams.TrainEpochs):
    print('*'*40 + ' '*10 + str(epoch) + ' '*10 + "*"*40)
    
    for phrase in ['train', 'trial', 'testEn', 'testEs', 'testFr']:
        print("\n"+"+"*20+' '*20 + phrase + ' '*20 + '+'*20 + '\n')
        epochModelOutputs = GCNModelRun.runEpochModel(model, criterion, dataLoader[phrase], optimizer, scheduler, isTrain=(phrase=='train'), DEVICE=DEVICE)
        evalResultDict = modelOutputsRecord.addEpochModelOutputs(epochModelOutputs, phrase=phrase)
        print(EvalATE.strEvalResultDict(evalResultDict))
    
    bestEvalResultDict = modelOutputsRecord.analyseModel()
    print("best iter is ", bestEvalResultDict['iter'])
    print('train: ', EvalATE.strEvalResultDict(bestEvalResultDict['train']))
    print('trial: ', EvalATE.strEvalResultDict(bestEvalResultDict['trial']))
    print('testEn:', EvalATE.strEvalResultDict(bestEvalResultDict['testEn' ]))
    print('testEs:', EvalATE.strEvalResultDict(bestEvalResultDict['testEs' ]))
    print('testFr:', EvalATE.strEvalResultDict(bestEvalResultDict['testFr' ]))

modelOutputsRecord.dump()