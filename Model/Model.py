from torch import nn
from transformers import AutoModel
import torch
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, pretrained_model = "bert-base-multilingual-cased", label_num=len("BIEOS")):
        super(BaseModel, self).__init__()
        self.label_num = label_num

        self.model = AutoModel.from_pretrained(pretrained_model, return_dict=True, output_hidden_states=True)
        self.classifier = nn.Linear(768, label_num)
        
    def forward(self, *args, **kwargs):

        batchLengths = kwargs['batchTokenizerEncode']['batchTextTokensLength']
        batchTextEncodePlus = kwargs['batchTokenizerEncode']['batchTextEncodePlus']
        model_outputs = self.model(**batchTextEncodePlus)
        
        token_embeddings = torch.cat([(model_outputs.last_hidden_state)[i][1:1+length] for i, length in enumerate(batchLengths)], dim=0)
        
        tokenLogSoftmax = F.log_softmax(self.classifier(F.relu(token_embeddings)), dim=1)
        
        modelOutDir = {'tokenLogSoftmax': tokenLogSoftmax}
        return modelOutDir