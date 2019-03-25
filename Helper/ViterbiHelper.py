import numpy as np
from Helper import HMMParam


def FindListProb(encodedTextList):
    #transitions = np.array([1,3,4,5,1,3,5,3,0,1,2])
    tranDist = sorted(set(encodedTextList))

    tranProb = []
    length= len(encodedTextList)
    for x in tranDist:
        tranProb.append(len(np.where(encodedTextList==x)[0])/length)

    return tranProb

def ComputeTranMatrix(encodedTextList):
    #transitions = np.array([1,3,4,5,1,3,5,3,0,1,2])
    nStates = 1+max(encodedTextList)

    tranMat = np.zeros(shape=(nStates,nStates))

    for i in range(0,nStates):
        fndEl = list(np.where(encodedTextList==i)[0])
        for el in fndEl:
            if(el +1 != len(encodedTextList)):
                tranMat[i,encodedTextList[el+1]] += 1/len(fndEl)

    return LaplaceSmooth(tranMat,nStates)

def LaplaceSmooth(TranMatrix,TotCount, lamb = 1 ):
    TranMatrix[TranMatrix!=0]= TranMatrix[TranMatrix!=0] + (1/TotCount)
    TranMatrix[TranMatrix==0]=1/TotCount
    return TranMatrix

def ConvertMatToJFormat(matrix):
    JMatrix = {}
    for i,row in zip(list(range(np.size(matrix,axis=0))),matrix):
        temp = {}
        for j,val in zip(list(range(np.size(row,axis=0))),row):
            temp[j]=val
        JMatrix[i]=temp
    return JMatrix

def viterbi(HMMParam):
    vTable = [{}]
    for st in HMMParam.States:
        vTable[0][st] = {"0": HMMParam.InitialProb[st] * HMMParam.EmissionProb[st][HMMParam.Observations[0]], "1": None}
    # Run Viterbi when t > 0
    for t in range(1, len(HMMParam.Observations)):
        vTable.append({})
        for st in HMMParam.States:
            max_tr_prob = vTable[t-1][HMMParam.States[0]]["0"]*HMMParam.TransProb[HMMParam.States[0]][st]
            prev_st_selected = HMMParam.States[0]
            for prev_st in HMMParam.States[1:]:
                tr_prob = vTable[t-1][prev_st]["0"]*HMMParam.TransProb[prev_st][st]
                if tr_prob > max_tr_prob:
                    max_tr_prob = tr_prob
                    prev_st_selected = prev_st
                    
            max_prob = max_tr_prob * HMMParam.EmissionProb[st][HMMParam.Observations[t]]
            vTable[t][st] = {"0": max_prob, "1": prev_st_selected}

    return vTable

def GetSequence(vTable):

    seq = []
    # The highest probability
    max_prob = max(value["0"] for value in vTable[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in vTable[-1].items():
        if data["0"] == max_prob:
            seq.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(vTable) - 2, -1, -1):
        seq.insert(0, vTable[t + 1][previous]["1"])
        previous = vTable[t + 1][previous]["1"]

    return seq,max_prob

def ConvertArrToJFormat(array):
    JArray = {}
    for i,prob in zip(list(range(len(array))),array):
        JArray[i]=prob
    return JArray

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)