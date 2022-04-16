from torch import nn
from transformers import AutoModel
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 相比于GCNModel，TSNE部分，将所有token的embedding存储起来了。

class GCNModel(nn.Module):

    def __init__(self, pretrained_model = "xlm-roberta-base", label_num=len("BIEOS"), gcn_dim=0, gcn_num=0, gcn_mode=0, stanzaTokenNerDim=0, stanzaTokenPosDim=4):
        super(GCNModel, self).__init__()
        self.label_num = label_num
        self.gcn_dim = gcn_dim
        self.gcn_num = gcn_num
        self.gcn_mode = gcn_mode
        self.stanzaTokenNerDim = stanzaTokenNerDim
        self.stanzaTokenPosDim = stanzaTokenPosDim

        self.model = AutoModel.from_pretrained(pretrained_model, return_dict=True, output_hidden_states=True)
        self.classifier = nn.Linear(768 + gcn_dim * (int(gcn_mode==3)+1) + stanzaTokenNerDim + stanzaTokenPosDim, label_num)

        if gcn_dim:
            self.addGCNConv(gcn_dim=gcn_dim, gcn_num=gcn_num, gcn_mode=gcn_mode)

        if stanzaTokenNerDim:
            self.addTokenNerInfo(stanzaTokenNerDim)

        if stanzaTokenPosDim:
            self.addTokenPosInfo(stanzaTokenPosDim)
        
        

    def addGCNConv(self, gcn_dim, gcn_num=3, gcn_mode=1):
        self.EmbeddingCompress = nn.Linear(768, gcn_dim * (int(gcn_mode==3)+1))

        self.GCNList = [GCNConv(gcn_dim * (int(gcn_mode==3)+1), gcn_dim).cuda() for i in range(gcn_num * (int(gcn_mode==3)+1))]

    def addTokenPosInfo(self, stanzaTokenPosDim):
        self.uposEmbedding = nn.Embedding(17, stanzaTokenPosDim)

    def addTokenNerInfo(self, stanzaTokenNerDim):
        self.nerEmbedding = nn.Embedding(19 * 4 + 1, stanzaTokenNerDim)
        

    def forward(self, *args, **kwargs):

        batchLengths = kwargs['batchTokenizerEncode']['batchTextTokensLength']
        batchTextEncodePlus = kwargs['batchTokenizerEncode']['batchTextEncodePlus']

        batchTokenEdgesTensor = kwargs['batchGCNInfo']['batchTokenEdgesTensor']
        batchStanzaTokensInfo = kwargs['batchGCNInfo']['batchStanzaTokensInfo']
        batchMapLocation = kwargs['batchGCNInfo']['batchMapLocation']

        model_outputs = self.model(**batchTextEncodePlus)
        
        token_embeddings = torch.cat([(model_outputs.last_hidden_state)[i][1:1+length] for i, length in enumerate(batchLengths)], dim=0)

        if self.gcn_dim or self.stanzaTokenNerDim or self.stanzaTokenPosDim:
            stanza_token_embeddings = []
            for i, (length, tokenEdgeTensor, tokenInfo, mapLocation) in enumerate(zip(batchLengths, batchTokenEdgesTensor, batchStanzaTokensInfo, batchMapLocation)):
                xlmr_embedding = (model_outputs.last_hidden_state)[i][1:1+length]
                
                stanza_token_embeddings.append(self.concatWordInformation(xlmr_embedding, tokenEdgeTensor, tokenInfo, mapLocation, self.gcn_mode, self.gcn_num))
            
            stanza_token_embeddings = torch.cat(stanza_token_embeddings)
            token_embeddings = torch.cat([token_embeddings, stanza_token_embeddings], dim=1)

        tokenLogSoftmax = F.log_softmax(self.classifier(F.relu(token_embeddings)), dim=1)

        modelOutDir = {'tokenLogSoftmax': tokenLogSoftmax, 'token_embeddings': token_embeddings.cpu().detach().numpy()}
        return modelOutDir
        
    def concatWordInformation(self, offset1_token_embedding, tokenEdgeTensor, tokenInfo, mapLocation, gcn_mode, gcn_num):
        # 获得词向量
        token_embedding = torch.cat([torch.mean(offset1_token_embedding[mapList],  dim=0, keepdim=True) for mapList in mapLocation['mapStanza2Xlmr']]) # word_len x 768
        # 获得图联系信息
        token_embedding_info = []

        # 获得图神经网络的向量
        if self.gcn_dim:
            gcn_embeddings = self.getGCNEmbedding(token_embedding, tokenEdgeTensor, gcn_mode, gcn_num)
            token_embedding_info.append(gcn_embeddings)

        if self.stanzaTokenPosDim:
            upos_embeddings = self.uposEmbedding(tokenInfo['upossId'])
            token_embedding_info.append(upos_embeddings)

        if self.stanzaTokenNerDim:
            ner_embeddings = self.nerEmbedding(tokenInfo['nersId'])        
            token_embedding_info.append(ner_embeddings)

        token_embedding_info = torch.cat(token_embedding_info, dim=1)
        token_embedding_info = F.relu(token_embedding_info)
                
        token_append_embeddings = []
        # 将词向量转换为token向量
        for mapList in mapLocation['mapXlmr2Stanza']:
            if len(mapList) != 0:
                token_append_embeddings.append(torch.mean(token_embedding_info[mapList], dim=0, keepdim=True))
            else:
                token_append_embeddings.append(torch.zeros((1, 2 * self.gcn_dim + self.stanzaTokenNerDim + self.stanzaTokenPosDim)))
        token_append_embeddings = torch.cat(token_append_embeddings, dim=0)

        return token_append_embeddings

    def getGCNEmbedding(self, token_embedding, tokenEdgeTensor, gcn_mode, gcn_num):
        token_biedges = tokenEdgeTensor['token_biedges']
        token_up_edges = tokenEdgeTensor['token_up_edges']
        token_down_edges = tokenEdgeTensor['token_down_edges']

        token_embedding = self.EmbeddingCompress(token_embedding)
        
        for i in range(gcn_num):
            if gcn_mode==0:
                token_embedding = self.GCNList[i](token_embedding, token_biedges)
            elif gcn_mode==1:
                token_embedding = self.GCNList[i](token_embedding, token_up_edges)
            elif gcn_mode==2:
                token_embedding = self.GCNList[i](token_embedding, token_down_edges)
            elif gcn_mode==3:
                token_embedding = torch.cat([self.GCNList[2*i](token_embedding, token_down_edges), self.GCNList[2*i+1](token_embedding, token_up_edges)], dim=1)
            token_embedding = F.relu(token_embedding)                
                
        return token_embedding