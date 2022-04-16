from CollateFn.CollateFnXLMR import CollateFnXLMR
import torch
from CollateFn.GCNUtils import *
import copy

class CollateFnXLMRGCN(CollateFnXLMR):

    def __init__(self, pretrained_model_name_or_path) -> None:
        super().__init__(pretrained_model_name_or_path)


    def collate_fn(self, batchDataSet):
        batchDataEncode = super().collate_fn(batchDataSet)
        
        batchLanguage = batchDataEncode['batchRawData']['batchLanguage']
        batchSentenceId = batchDataEncode['batchRawData']['batchSentenceId']
        batchOffsets = batchDataEncode['batchTokenizerEncode']['batchTokensOffset']
        batchDataEncode['batchGCNInfo'] = self.getBatchGCNInfo(batchLanguage, batchSentenceId, copy.deepcopy(batchOffsets))

        return batchDataEncode

    def getBatchGCNInfo(self, batchLanguage, batchSentenceId, batchOffsets):
        batchStanzaDoc = self.getbatchStanzaDoc(batchLanguage, batchSentenceId)
        batchTokenEdgesTensor = []
        batchStanzaTokensInfo = []
        batchMapLocation = []
        
        for iter, (offsets, doc) in enumerate(zip(batchOffsets, batchStanzaDoc)):
            stanzaTokens = sum([sentence.tokens for sentence in doc.sentences], [])

            mapLocation = self.getMapLocation(offsets, stanzaTokens)
            tokenInfo = self.getStanzaTokensInfo(stanzaTokens)
            tokenEdgeTensor = self.getStanzaTokenEdges(stanzaTokens)

            batchTokenEdgesTensor.append(tokenEdgeTensor)
            batchStanzaTokensInfo.append(tokenInfo)
            batchMapLocation.append(mapLocation)
        
        batchGCNInfo = {
            'batchTokenEdgesTensor': batchTokenEdgesTensor,
            'batchStanzaTokensInfo': batchStanzaTokensInfo,
            'batchMapLocation': batchMapLocation
        }
        return batchGCNInfo

    @classmethod
    def to(self, batchDataEncode, DEVICE):
        super().to(batchDataEncode, DEVICE)
        # 把张量放入cpu或gpu
        for i in range(len(batchDataEncode['batchGCNInfo']['batchTokenEdgesTensor'])):
            batchDataEncode['batchGCNInfo']['batchTokenEdgesTensor'][i]['token_biedges'] = batchDataEncode['batchGCNInfo']['batchTokenEdgesTensor'][i]['token_biedges'].to(DEVICE)
            batchDataEncode['batchGCNInfo']['batchTokenEdgesTensor'][i]['token_up_edges'] = batchDataEncode['batchGCNInfo']['batchTokenEdgesTensor'][i]['token_up_edges'].to(DEVICE)
            batchDataEncode['batchGCNInfo']['batchTokenEdgesTensor'][i]['token_down_edges'] = batchDataEncode['batchGCNInfo']['batchTokenEdgesTensor'][i]['token_down_edges'].to(DEVICE)

        for i in range(len(batchDataEncode['batchGCNInfo']['batchStanzaTokensInfo'])):
            batchDataEncode['batchGCNInfo']['batchStanzaTokensInfo'][i]['upossId'] = batchDataEncode['batchGCNInfo']['batchStanzaTokensInfo'][i]['upossId'].to(DEVICE)
            batchDataEncode['batchGCNInfo']['batchStanzaTokensInfo'][i]['nersId'] = batchDataEncode['batchGCNInfo']['batchStanzaTokensInfo'][i]['nersId'].to(DEVICE)
        
        return batchDataEncode


    def getStanzaTokensInfo(self, tokens):
        uposList = []
        nerList = []
        for token in tokens:
            if len(token.words)==1:
                uposList.append(StanzaDocPickle.upos2id[token.words[0].upos])
            else:
                uposList.append(StanzaDocPickle.upos2id["X"])
            nerList.append(StanzaDocPickle.ner2id[token.ner])
        upossId = torch.LongTensor(uposList)            
        nersIds = torch.LongTensor(nerList)
        token_info = {
            'upossId': upossId, 
            'nersId': nersIds
        }
        return token_info

    def getStanzaTokenEdges(self, tokens):
        id2token = {}
        for token_index, token in enumerate(tokens):
            for word in token.words:
                id2token[word.id]=token_index

        token_up_edges = []
        for token in tokens:
            for word in token.words:
                token_up_edges.append((id2token[word.id], id2token[word.id]))
                if word.head != 0:
                    token_up_edges.append((id2token[word.id], id2token[word.head]))

        token_down_edges = [(e, s) for (s, e) in token_up_edges]

        token_biedges = sum([token_down_edges, token_up_edges], [])
        token_up_edges = torch.LongTensor(token_up_edges).t()
        token_down_edges = torch.LongTensor(token_down_edges).t()
        token_biedges = torch.LongTensor(token_biedges).t()

        token_edges_tensor = {
            'token_biedges': token_biedges,
            'token_up_edges': token_up_edges,
            'token_down_edges': token_down_edges
        }
        return token_edges_tensor
    
    def getMapLocation(self, xlmrOffsets, stanzaTokens):
        # 生成token对应的句法信息-->获得gcn前的word embedding-->获得图连接信息-->获得gcn前的词向量
        xlmrLocations = [Location(offset[0], offset[1]) for offset in xlmrOffsets]
        stanzaLocations = [Location(token.start_char, token.end_char) for token in stanzaTokens]

        mapXlmr2Stanza, mapStanza2Xlmr = mapOffsets(xlmrLocations, stanzaLocations)
        mapLocation = {
            'mapXlmr2Stanza': mapXlmr2Stanza, 
            'mapStanza2Xlmr': mapStanza2Xlmr
        }
        return mapLocation

    def getbatchStanzaDoc(self, batchLanguage, batchSentenceId):
        batchStanzaDoc = []
        for language, sentenceId in zip(batchLanguage, batchSentenceId):
            doc = StanzaDocPickle.getDoc(language=language, sentenceId=sentenceId)
            batchStanzaDoc.append(doc)
        return batchStanzaDoc
