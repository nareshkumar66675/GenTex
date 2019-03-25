import pandas as pd
from Helper import TextHelper
from Helper import HMMParam
from Helper.ViterbiHelper import *
import numpy as np
import sys
import os.path
import pickle

dataSetFolder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

textPath = dataSetFolder+ os.path.join('\GenTex\Dataset\lorem.txt')
with open(textPath, 'r') as textFile:
    textData=textFile.read().replace('\n', '')

textHelper = TextHelper.TextHelper()

textHelper.PreProcessText(textData)

HMMValues = HMMParam.HMM()

HMMValues.SetParams(textHelper.EncodedTextList)

bPath,prob = viterbi(HMMValues)

pLabels = textHelper.Encoder.inverse_transform(bPath)


print('End')
