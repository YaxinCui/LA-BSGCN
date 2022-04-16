from ArgumentParser.BaseArgParser import BaseArgumentParser

class GCNArgumentParser(BaseArgumentParser):
    def __init__(self) -> None:
        super().__init__()
        self.addGCNArgument()
        self.addNerArgument()
        self.addPosArgument()
    
    def parse_args(self):
        if self.getEnv()=="notebook":
            dataParams = self.parser.parse_args([])
        else:
            dataParams = self.parser.parse_args()
        print(dataParams)
        assert dataParams.GCNModelMode in [0, 1, 2, 3]
        return dataParams

    def addGCNArgument(self):
        self.parser.add_argument('--GCNModelMode',type=int, default=3, help='GCN Mode: 0:undirected edge, 1:up directed, 2:down directed, 3:two directed')
        self.parser.add_argument('--GCNEmbedDim',type=int, default=0, help='GCN Embedding Dim')
        self.parser.add_argument('--GCNLayerNum',type=int, default=0, help='GCN Layer Num')


    def addNerArgument(self):
        self.parser.add_argument('--NEREmbedDim',type=int, default=0, help='NER Embedding Dim')

    def addPosArgument(self):
        self.parser.add_argument('--POSEmbedDim',type=int, default=0, help='POS Embedding Dim')
