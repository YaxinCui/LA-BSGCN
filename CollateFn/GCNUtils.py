from Dataset.SemEval16Task5Dataset import SemEvalXMLDataset
import pickle
import stanza
from tqdm import tqdm

class StanzaDocPickle():
    docPickleFolder = "Dataset/StanzaDocPickle/"

    stanzaUposWordList = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    stanzaEntityList = ["MISC", "PER", "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]
    upos2id = {}
    id2upos = {}
    for i, upos in enumerate(stanzaUposWordList):
        upos2id[upos]=i
        id2upos[id]=upos

    ner2id = {}
    id2ner = {}
    ner = []
    for head_char in "BIES":
        for entity in stanzaEntityList:
            ner.append(head_char+"-"+entity)
    ner.append("O")

    for i, ner in enumerate(ner):
        ner2id[ner]=i
        id2ner[i]=ner

    @classmethod
    def getDoc(cls, language, sentenceId):
        docPickleFile = cls.docPickleFolder+"/"+language.lower() + "/"+language+sentenceId+".pkl"
        with open(docPickleFile, "rb") as f:
            doc = pickle.load(f)
        return doc

    def preDoc(self):
        self.stanzaNlp = {  
            'english':stanza.Pipeline('en', verbose=False), 
            'spanish':stanza.Pipeline('es', verbose=False),
            'french':stanza.Pipeline('fr', verbose=False)
        }

        for language in ["english", "spanish", "french"]:
            for phrase in ["Train", "Trial", "Test"]:
                for domain in ["Restaurant"]:
                    dataset = SemEvalXMLDataset(phrase=phrase, language=language)
                    self.dumpDocPickle(dataset)

    def dumpDocPickle(self, dataset):
        
        for example in tqdm(dataset):
            text, UniOpinions, phrase, language, domain, reviewId, sentenceId = example
            doc = self.stanzaNlp[language](text)
            docPicklePath = f"./Dataset/StanzaDocPickle/{language}/{language}{sentenceId}.pkl"
            with open(docPicklePath, "wb") as f:
                pickle.dump(doc, f)

class Location:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return 'Location(%s, %s)' % (self.start, self.end)

    def __repr__(self):
        return self.__str__()

def mapOffsets(locations0, locations1) -> tuple:
    map0to1 = [[] for i in range(len(locations0))]
    map1to0 = [[] for i in range(len(locations1))]
    
    map0to1_index = map1to0_index = 0

    while map0to1_index < len(locations0) or map1to0_index < len(locations1):
        if locations0[map0to1_index].end == locations1[map1to0_index].end:
            map0to1[map0to1_index].append(map1to0_index)
            map1to0[map1to0_index].append(map0to1_index)

            map0to1_index += 1
            # 特殊情况
            # 当前一个分词为 _ 即空格时，后一个分词为某个独立标点，_ 跟标点的offset会一样，这时假设映射相同
            # 例子 My boyfriend had Prime Rib it was good .
            if map0to1_index < len(locations0) and str(locations0[map0to1_index]) == str(locations0[map0to1_index-1]):
                map0to1[map0to1_index].append(map1to0_index)
                map0to1_index += 1

            map1to0_index += 1
            
        else:
            if locations0[map0to1_index].end <= locations1[map1to0_index].start:
                # 不交叉，情况1
                map0to1_index += 1
                if map0to1_index < len(locations0) and str(locations0[map0to1_index]) == str(locations0[map0to1_index-1]):
                    map0to1[map0to1_index].append(map1to0_index)
                    map0to1_index += 1


            elif locations1[map1to0_index].end <= locations0[map0to1_index].start:
                # 不交叉，情况2
                map1to0_index += 1
            else:# 交叉
                if locations0[map0to1_index].end < locations1[map1to0_index].end:
                    map0to1[map0to1_index].append(map1to0_index)
                    map1to0[map1to0_index].append(map0to1_index)

                    map0to1_index += 1
                    if map0to1_index < len(locations0) and str(locations0[map0to1_index]) == str(locations0[map0to1_index-1]):
                        map0to1[map0to1_index].append(map1to0_index)
                        map0to1_index += 1

                else:
                    map0to1[map0to1_index].append(map1to0_index)
                    map1to0[map1to0_index].append(map0to1_index)
                        
                    map1to0_index += 1
        
    return map0to1, map1to0