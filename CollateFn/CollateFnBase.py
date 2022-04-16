from abc import abstractmethod

class CollateFnBase:
    BIEOS = "BIEOS"
    ATElabel2id = {}
    ATEid2label = {}
    for id, label in enumerate(BIEOS):
        ATElabel2id[label] = id
        ATEid2label[id] = label

    label2id = ATElabel2id
    id2label = ATEid2label
    
    @classmethod
    def getRawData(cls, batchDataSet):
        # 记录下原始数据
        batchDataSet =list(zip(*batchDataSet))
        # 原始数据顺序为 text, UniOpinions, phrase, language, domain, reviewId, sentenceId
        batchRawData = {}
        rawBatchDataKeys = ["batchText", "batchOpinions", "batchPhrase", "batchLanguage", 
                                            "batchDomain", "batchReviewId", "batchSentenceId"]
        for i, key in enumerate(rawBatchDataKeys):
            batchRawData[key] = batchDataSet[i]
        batchRawData['batchText'] = list(batchRawData['batchText'])
        return batchRawData

    @abstractmethod
    def collate_fn(self, batchDataSet):
        pass


class CollateFnTransformer(CollateFnBase):
    
    @classmethod
    def collate_fn(self):
        pass

    @classmethod
    def getBatchBIEOSLabels(self, batchTokensOffset, batchOpinons):
        # offset是前闭后开
        batchLabels = []
        batchOpinionsOffsets = []
        for (offsets, opionions) in (zip(batchTokensOffset, batchOpinons)):
            labels = []
            opinionsOffsets = []
            message = []
            opionions = list(opionions)
            # 从offset列表从后往前推理对应的标签
            while 0 < len(opionions):
                while opionions[-1].end <= offsets[-1][0]:
                    labels.append('O')
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    offsets.pop()
                if offsets[-1][0] <= opionions[-1].begin:
                    labels.append("S")
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionsOffsets.append(offsets[-1])
                    offsets.pop()
                    opionions.pop()
                else:
                    # 说明是B I E
                    labels.append("E")
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionOffsetEnd = offsets[-1][1]
                    offsets.pop()
                    while opionions[-1].begin < offsets[-1][0]:
                        labels.append("I")
                        message.append(str(offsets[-1]) + " is " + labels[-1])
                        offsets.pop()
                    labels.append("B")
                    message.append(str(offsets[-1]) + " is " + labels[-1])
                    opinionOffsetBegin = offsets[-1][0]
                    opinionsOffsets.append((opinionOffsetBegin, opinionOffsetEnd))
                    offsets.pop()
                    opionions.pop()
            labels.extend(['O'] * len(offsets))
            labels.reverse()
            batchLabels.append(labels)
            opinionsOffsets.reverse()
            batchOpinionsOffsets.append(opinionsOffsets)
            message.reverse()

        return batchLabels

    @classmethod
    def getXLMRTokenizerEncode(self):
        pass

    @classmethod
    def getBatchLabelsEncode(self):
        pass
