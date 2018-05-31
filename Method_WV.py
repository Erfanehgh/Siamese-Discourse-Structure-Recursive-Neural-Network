import numpy as np
import re
from Methods_Preprocessing import preprocessor1
#from itertools import izip

def sibAveraging(first, second):
    return [ ((x + y)/2.0) for x, y in zip(first, second)]

def WordAveraging(sent, WV, dim):
    summ = [0.0] * (dim)
    #summ.extend([1])
    A = 0.0;
    sent_A=preprocessor1(re.sub(r"[\n(\[\])]", "", sent)).split(" ")
    for word in sent_A:
        if word in WV : #and word not in stop:
            A = A + 1.0
            for i in range(0, dim):
                summ[i] = summ[i] + float((WV[word])[i])
    if A != 0:
        #A = 1
        for i in range(0, dim):
            summ[i] = summ[i] / A

    #print len(summ)
    return summ;

def readWV(path, stop):
    WV = {}
    file_1 = open(path, "r")
    for line in file_1:
        line = line.replace("\n", "")
        wordV = line.split(" ")
        key = wordV[0]

        if key not in stop:
            del wordV[0]
            WV[key] = np.asarray(wordV,dtype=float)
            #print WV[key]
            #WV[key] = np.tanh(WV[key])
            #print WV[key]

    return WV
'''
def readEDUVec(pathEDU, pathVect):
    EDUVector = {}
    with open(pathEDU) as textfile1, open(pathVect) as textfile2:
        for EDU, Vector in izip(textfile1, textfile2):
            EDUVector[preprocessor1(EDU)] = Vector

    return EDUVector
'''