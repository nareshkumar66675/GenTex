from sklearn import preprocessing
import string

class TextHelper(object):
    """description of class"""


    def __init__(self, Encoder = None):
        self.Encoder = preprocessing.LabelEncoder()
        self.TextList=[]
        self.TextLabels=[]
        self.EncodedTextList=[]

    def PreProcessText(self,text):
        text = self.TextProcess(text)
        self.TextList = list(filter(None,text.split()))
        #self.TextList = list(map(lambda x:x.lower(),self.TextList))
        textListNoDup = list(dict.fromkeys(self.TextList))
        self.TextLabels = self.Encoder.fit(textListNoDup)
        self.EncodedTextList = self.Encoder.transform(self.TextList)


    def TextProcess(self,text):
        remPunct = str.maketrans('', '', string.punctuation)
        text = text.lower()
        text = text.replace(',',' COMMA')
        text = text.replace('.',' PERIOD')
        text = text.translate(remPunct)
        return text