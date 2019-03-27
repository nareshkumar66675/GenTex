import numpy as np
import random
from Helper import HMMParam


''' Helper Module - All Functions related to algorithm and others
'''

class ProbValues(object):
    
    def __init__(self):
        self.ForwardValues =[]
        self.BackwardValues = []
        self.ProbForward = []
        self.PosteriorValues =[]

''' Initial Probability '''
def FindListProb(encodedTextList):
    #transitions = np.array([1,3,4,5,1,3,5,3,0,1,2])
    tranDist = sorted(set(encodedTextList))

    tranProb = []
    length= len(encodedTextList)
    for x in tranDist:
        tranProb.append(len(np.where(encodedTextList==x)[0])/length)

    return tranProb

''' Finds Transition Probability Matrix '''
def ComputeTranMatrix(encodedTextList):

    nStates = 1+max(encodedTextList)

    tranMat = np.zeros(shape=(nStates,nStates))

    for i in range(0,nStates):
        fndEl = list(np.where(encodedTextList==i)[0])
        for el in fndEl:
            if(el +1 != len(encodedTextList)):
                tranMat[i,encodedTextList[el+1]] += 1/len(fndEl)

    return LaplaceSmooth(tranMat,nStates)


''' Laplace Smoothing to normalize the matrix '''
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



#def viterbi(HMMParam,observations):
#    vTable = [{}]
#    for st in HMMParam.States:
#        vTable[0][st] = {"0": HMMParam.InitialProb[st] * HMMParam.EmissionProb[st][observations[0]], "1": None}
#    # Run Viterbi when t > 0
#    for t in range(1, len(observations)):
#        vTable.append({})
#        for st in HMMParam.States:
#            max_tr_prob = vTable[t-1][HMMParam.States[0]]["0"]*HMMParam.TransProb[HMMParam.States[0]][st]
#            prev_st_selected = HMMParam.States[0]
#            for prev_st in HMMParam.States[1:]:
#                tr_prob = vTable[t-1][prev_st]["0"]*HMMParam.TransProb[prev_st][st]
#                if tr_prob > max_tr_prob:
#                    max_tr_prob = tr_prob
#                    prev_st_selected = prev_st
                    
#            max_prob = max_tr_prob * HMMParam.EmissionProb[st][observations[t]]
#            vTable[t][st] = {"0": max_prob, "1": prev_st_selected}

#    return vTable

''' Viterbi Algorithm Implementation 
    Inspired From Wiki
'''

def ViterbiAlgo(tranMat, EmissMat, observations):
    num_obs = len(observations)
    num_states = tranMat.shape[0]
    log_probs = np.zeros(num_states)
    paths = np.zeros( (num_states, num_obs+1 ))
    paths[:, 0] = np.arange(num_states)
    for obs_ind, obs_val in enumerate(observations):
        for state_ind in range(num_states):
            val = 0
            if obs_val< np.size(EmissMat,1):
                val = np.log(EmissMat[state_ind, obs_val])
            temp_probs = log_probs + \
                          val + \
                         np.log(tranMat[:, state_ind])
            best_temp_ind = np.argmax(temp_probs)
            paths[state_ind,:] = paths[best_temp_ind,:]
            paths[state_ind,(obs_ind+1)] = state_ind
            log_probs[state_ind] = temp_probs[best_temp_ind]
    best_path_ind = np.argmax(log_probs)
    
    return (paths[best_path_ind], log_probs[best_path_ind])

def GetSequence(vTable):
    seq = []
    max_prob = max(value["0"] for value in vTable[-1].values())
    previous = None
    for st, data in vTable[-1].items():
        if data["0"] == max_prob:
            seq.append(st)
            previous = st
            break
    for t in range(len(vTable) - 2, -1, -1):
        seq.insert(0, vTable[t + 1][previous]["1"])
        previous = vTable[t + 1][previous]["1"]

    return seq,max_prob

def ConvertArrToJFormat(array):
    JArray = {}
    for i,prob in zip(list(range(len(array))),array):
        JArray[i]=prob
    return JArray


def posterior(HMM,ProbValues):
    posterior = []
    for i in range(len(HMM.Observations)):
        posterior.append({st: ProbValues.ForwardValues[i][st] * ProbValues.BackwardValues[i][st] / ProbValues.ProbForward for st in HMM.States})

    return posterior

''' Forward Algorithm
    Inspired From Wiki 
'''
def ForwardAlgo(HMM, EndState):
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(HMM.Observations):
        f_curr = {}
        for st in HMM.States:
            if i == 0:
                prev_f_sum = HMM.InitialProb[st]
            else:
                prev_f_sum = sum(f_prev[k]*HMM.TransProb[k][st] for k in HMM.States)

            f_curr[st] = HMM.EmissionProb[st][observation_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * HMM.TransProb[k][EndState] for k in HMM.States)


    return fwd,p_fwd

''' Backward Algorithm
    Inspired From Wiki 
'''

def BackwardAlgo(HMM, EndState = 'PERIOD'):
    bkw = []
    b_prev = {}
    for i, observation_i_plus in enumerate(reversed(HMM.Observations[1:]+[None,])):
        b_curr = {}
        for st in HMM.States:
            if i == 0:
                # base case for backward part
                b_curr[st] = HMM.TransProb[st][EndState]
            else:
                b_curr[st] = sum(HMM.TransProb[st][l] * HMM.EmissionProb[l][observation_i_plus] * b_prev[l] for l in HMM.States)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(HMM.InitialProb[l] * HMM.EmissionProb[l][HMM.Observations[0]] * b_curr[l] for l in HMM.States)

    return bkw

def FindLargest(listObj):
    largest = 0
    largest2 = 0
    for i in range(len(listObj)):      
        if listObj[i] > largest:  
            largest2 = largest 
            largest = listObj[i]
            LargestIndex = i
        elif largest2 == None or largest2 <= listObj[i]:  
            Largest2Index = i 
            largest2 = listObj[i]
    return LargestIndex,Largest2Index

def GenerateTextPath(transMat , maxCount = 100):
    #transMat = np.zeros(shape=(100,100))
    wordIndex = random.randint(0,np.size(transMat,0))

    path = []
    for i in range(maxCount):
        path.append(wordIndex)
        row = transMat[wordIndex]
        LargestIndex,Largest2Index = FindLargest(row)
        val = (transMat[wordIndex,LargestIndex]*30)/100
        transMat[wordIndex,LargestIndex] = transMat[wordIndex,LargestIndex] - val
        transMat[wordIndex,Largest2Index] = transMat[wordIndex,Largest2Index] + val
        wordIndex = LargestIndex
    
    return path
