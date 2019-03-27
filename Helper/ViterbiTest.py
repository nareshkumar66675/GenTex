import numpy as np

def GetStartsWithIndex(textList,character):
    index = []
    for i in range(len(textList)):
        if textList[i].lower().startswith(character):
            index.append(i)
    return index


def ComputeTranMatrixCharacter(textList):
    textList = ['an','ca','dsd','ef','ab','ca','ed','cd','z','an','boo']
    nStates = 26

    tranMat = np.zeros(shape=(nStates,nStates))
    alphabet = list(map(chr, range(97, 123)))
    for i in range(0,nStates):
        fndEl = GetStartsWithIndex(textList,alphabet[i])
        for el in fndEl:
            if el +1 != len(textList):
                jIndex = alphabet.index(textList[el+1][0])
                tranMat[i,jIndex] += 1/len(fndEl)

    return tranMat


#def ComputeEmissMatrix(textList):
