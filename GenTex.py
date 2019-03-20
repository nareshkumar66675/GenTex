import pandas as pd
import string
import matplotlib.pyplot as plt 
import numpy as np
import sys
import os.path
from sklearn import preprocessing


#def transition_matrix(transitions):
#    nStates = 1+ max(transitions) #number of states

#    tranMat = [[0]*nStates for _ in range(nStates)]

#    for (i,j) in zip(transitions,transitions[1:]):
#        tranMat[i][j] += 1

#    # Find Probabilities
#    for row in tranMat:
#        s = sum(row)
#        if s > 0:
#            row[:] = [f/s for f in row]
#    return tranMat

def ComputeTranMatrix(transitions):
    #transitions = np.array([1,3,4,5,1,3,5,3,0,1,2])
    nStates = 1+max(transitions)

    tranMat = np.zeros(shape=(nStates,nStates))

    for i in range(0,nStates):
        fndEl = list(np.where(transitions==i)[0])
        for el in fndEl:
            if(el +1 != len(transitions)):
                tranMat[i,transitions[el+1]] += 1/len(fndEl)

    return tranMat


def findProb(transitions):
    transitions = np.array([1,3,4,5,1,3,5,3,0,1,2])
    tranDist = sorted(set(transitions))

    tranProb = []
    length= len(transitions)
    for x in tranDist:
        tranProb.append(len(np.where(transitions==x)[0])/length)

    return tranProb

findProb([1,2])
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
tranMatrix = ComputeTranMatrix(le.transform(textList))

emissMatrix = tranMatrix.transpose(1,0)

startProb = findProb(textList)

for row in tranMatrix: print(' '.join('{0:.2f}'.format(x) for x in row))

print('End')
