from Model.ModelRun import ModelRun
from CollateFn.CollateFnXLMRGCN import CollateFnXLMRGCN
import torch

class GCNModelRun(ModelRun):
    @classmethod
    def runBatchModel(cls, batchDataEncode, model, criterion, DEVICE="cpu"):
        # 返回loss, PredLabelList, TrueLabelList
        batchDataEncode = CollateFnXLMRGCN.to(batchDataEncode, DEVICE)
        
        batchFlattenIds = batchDataEncode['batchLabelsEncode']['batchFlattenIds']
        
        batchModelOutputs = model(**batchDataEncode)
        batchTrueLabelsTensor = torch.LongTensor(batchFlattenIds).to(DEVICE)
        batchLoss = criterion(batchModelOutputs['tokenLogSoftmax'], batchTrueLabelsTensor)
        batchModelOutputs['batchLoss'] = batchLoss

        tokenLogSoftmax = batchModelOutputs.pop('tokenLogSoftmax')

        batchModelOutputs['tokenLogSoftmax'] = tokenLogSoftmax.detach().cpu().numpy()
        batchModelOutputs['batchLabelsEncode'] = batchDataEncode['batchLabelsEncode']
        batchModelOutputs['batchTextTokensLength'] = batchDataEncode['batchTokenizerEncode']['batchTextTokensLength']
        batchModelOutputs['batchRawData'] = batchDataEncode['batchRawData']

        return batchModelOutputs