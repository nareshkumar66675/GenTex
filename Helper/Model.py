class Model(object):
    """Model Class for Persisting"""
    def __init__(self, hmm=[], textHelper=[], probValues =[]):
        self.HMM = hmm
        self.TextHelper = textHelper
        self.ProbValues = probValues


