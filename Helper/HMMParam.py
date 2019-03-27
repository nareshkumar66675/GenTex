import numpy as np
from Helper.ViterbiHelper import *

class HMM(object):
    ''' HMM Object holding Parameter Values
    '''
    def __init__(self,States=[],Observations=[], InitialProb = {},TransProb ={},EmissionProb={},TransMat=[]):
        self.States = States
        self.Observations=Observations
        self.InitialProb=InitialProb
        self.TransProb=TransProb
        self.EmissionProb=EmissionProb
        self.TransMat = TransMat


    def SetParams(self,encodedTextList):
        self.States = (list(range(max(encodedTextList))))
        self.Observations = (list(range(max(encodedTextList))))
        self.InitialProb = ConvertArrToJFormat(np.array(FindListProb(encodedTextList)))
        tempTrans = ComputeTranMatrix(encodedTextList)
        self.TransMat = tempTrans
        self.TransProb = ConvertMatToJFormat(tempTrans)
        self.EmissionProb = ConvertMatToJFormat(tempTrans.transpose(1,0))
