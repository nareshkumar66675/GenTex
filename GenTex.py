import pandas as pd
from Helper import HMMParam
from Helper.ViterbiHelper import *
import string
import numpy as np
import sys
import os.path
from sklearn import preprocessing
import operator
import pickle

dataSetFolder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

textPath = dataSetFolder+ os.path.join('\GenTex\Dataset\lorem.txt')
with open(textPath, 'r') as textFile:
    textData=textFile.read().replace('\n', '')

remPunct = str.maketrans('', '', string.punctuation)

textData = textData.translate(remPunct)

textList = list(filter(None,textData.split()))

textList = list(map(lambda x:x.lower(),textList))

textListNoDup = list(dict.fromkeys(textList))

le = preprocessing.LabelEncoder()
le.fit(textListNoDup)

encodedTextList = le.transform(textList)


HMMValues = HMMParam.HMM()

HMMValues.SetParams(encodedTextList)

bPath,prob = viterbi(HMMValues)

pLabels = le.inverse_transform(bPath)


print('End')
