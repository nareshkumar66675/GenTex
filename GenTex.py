import pandas as pd
from Helper import TextHelper
from Helper import HMMParam
from Helper.ViterbiHelper import *
import numpy as np
import sys
import os.path
import pickle


def GetSeqAndLabels():
    with open(dataSetFolder+ os.path.join('\GenTex\Models\Table.pkl'),'rb') as f:
        vTable = pickle.load(f)

    with open(dataSetFolder+ os.path.join('\GenTex\Models\Text.pkl'),'rb') as f:
        textHelper = pickle.load(f)

    seq,prob = GetSequence(vTable)
    pLabels = textHelper.Encoder.inverse_transform(seq)

    return pLabels,seq

def GenerateText():

    pLabels,seq = GetSeqAndLabels()
    print("Generated Text from Text Corpus")
    print("")
    print(' '.join(pLabels[:100]))

def PredictText():
    print("Text Prediction")
    pLabels,seq = GetSeqAndLabels()
    while True:
        word = input('Enter a word from text corpus (E to Exit): ')

        if str.lower(word) == "e":
            break
        else:
            try:
                wIndx = np.where(pLabels==word)[0]
                if(wIndx[0] < len(pLabels)-1):
                    print(pLabels[wIndx[0]+1])
            except IndexError:
                print("Word not Found")

def ReTrain():
    print("Retraining Model")
    textPath = dataSetFolder+ os.path.join('\GenTex\Dataset\lorem.txt')
    with open(textPath, 'r') as textFile:
        textData=textFile.read().replace('\n', ' ')

    textHelper = TextHelper.TextHelper()

    textHelper.PreProcessText(textData)

    HMMValues = HMMParam.HMM()

    HMMValues.SetParams(textHelper.EncodedTextList)

    vTable = viterbi(HMMValues)

    with open(dataSetFolder+ os.path.join('\GenTex\Models\Table.pkl'),'wb+') as f:
        pickle.dump(vTable, f)

    with open(dataSetFolder+ os.path.join('\GenTex\Models\Text.pkl'),'wb+') as f:
        pickle.dump(textHelper, f)

    print("Model Retraining Completed")

print("Text Generation/Prediction using HMM")


dataSetFolder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

while True:
    print("1. Generate Text Sequence")
    print("2. Predict Text")
    print("3. Re-Train Model")

    userChoice = (input('Select one option from above : '))

    if userChoice == "1":
        GenerateText()
    elif userChoice == "2":
        PredictText()
    elif userChoice == "3":
        ReTrain()
    else:
        choice = input('Enter Valid Option. Press Y to Restart and N to Exit: ')
        if str.lower(choice) == 'n':
            sys.exit("User Exited")
        else:
            continue




print('End')

