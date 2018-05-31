import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from Method_RSTTree import sortEduKey, update_EDU

import random
#from pylab import savefig
import pylab

labels = ["financial", "april", "india", "moscow", "brazil"]

docRepresentaion = np.array([[0.67226,0.0049257,0.30049,0.33917,-0.11859,-0.43213,-0.064794,-0.35349,-0.1352,-0.31288,-0.36311,0.95772,-1.3127,0.0045216,0.12404,0.30968,-0.11743,-0.55025,0.39012,0.024453,1.3455,-0.015076,-0.62894,-0.77782,-0.63266,-1.2771,-0.36317,-0.41216,-0.28301,0.9475,3.6989,0.62595,1.3024,-0.56111,-0.76491,-0.27065,-1.1322,-0.53162,1.096,-1.0186,-0.7311,-0.39185,0.82753,0.4969,-0.14401,0.0095985,-0.33367,1.2218,0.71828,0.52732],
                  [0.043106,0.048456,0.033849,-0.074085,-0.17892,0.06877,-1.1887,0.094169,-0.57696,-0.6888,0.31222,-0.69676,-0.40506,-0.49433,1.3993,0.13045,-1.3774,-0.50435,-1.702,0.85363,1.3086,-0.31586,0.7332,-0.44083,-0.7165,-1.182,0.50538,-0.40834,-0.028543,0.93309,2.7738,-0.3623,-0.974,-0.35168,0.29959,-0.7362,0.75754,0.016837,0.17139,-0.33548,-0.74626,-0.53711,-0.25172,-1.3667,-0.35269,0.15424,-0.24418,-0.62661,0.057467,0.1768],
                  [-0.20356,-0.8707,-0.19172,0.73862,0.18494,0.14926,0.48079,-0.21633,0.72753,-0.36912,0.13397,-0.1143,-0.18075,-0.64683,-0.18484,0.83575,0.48179,0.76026,-0.50381,0.80743,1.2195,0.3459,0.22185,0.31335,1.2066,-1.8441,0.14064,-0.99715,-1.1402,0.32342,3.2128,0.42708,0.19504,0.80113,0.38555,-0.12568,-0.26533,0.055264,-1.1557,0.16836,-0.82228,0.20394,0.089235,-0.60125,-0.032878,1.3735,-0.51661,0.29611,0.23951,-1.3801],
                  [0.36366,1.4939,0.25894,-0.37835,-0.37487,-0.83621,0.52872,1.3824,-1.3646,-0.16165,0.28893,-1.1491,-0.49207,0.78691,0.093677,0.40731,-0.26552,0.36091,-0.20346,0.82827,1.1568,-0.027295,-0.5743,0.11437,-0.53875,-2.2521,0.22763,0.71361,-0.055496,-0.02831,2.0417,0.30655,-0.24599,-0.9115,-0.98252,-0.33366,0.067877,0.68164,-0.1721,0.65925,0.27524,-0.16436,1.0991,-1.2045,0.062384,0.17609,-0.62426,0.96782,-0.56523,-0.45671],
                  [0.014697,-0.4585,-0.049365,1.1068,-0.34228,-0.99984,-0.59754,-0.021446,0.070466,1.0038,1.3814,-0.15615,-0.04429,-0.64585,1.2024,-0.35262,0.094759,-0.26553,-0.16363,-0.29023,-0.90477,-0.64781,-0.23873,0.42291,0.019089,-1.194,0.68176,-0.019443,-0.42695,0.073599,2.5519,0.61103,-0.57614,0.053781,-0.22584,-0.67029,-0.91667,0.71881,0.012405,-0.55416,-0.52862,0.33267,1.2562,-1.3472,-0.43544,1.2502,-0.12576,-0.060888,0.79752,-1.1951]])


def plotDocs(EDUs_Test, name, labelColor):

    labels=[]
    docRepresentaion = []
    keys = EDUs_Test.keys()
    i = 0
    for key in keys:

        label = key.split("-")[1]
        labels.append(label)
        EDUs_te = EDUs_Test[key]
        #EDUs_te = update_EDU(EDUs_te, W1, "", activationFunc)
        eduKeys = sortEduKey(EDUs_te.keys(), reverse=True)
        #print (eduKeys)
        testRep = EDUs_te[str(eduKeys[0])].vector
        docRepresentaion.append(testRep)

    scaler = MinMaxScaler()
    X_embedded = TSNE(n_components=2).fit_transform(docRepresentaion)
    #print (X_embedded)
    scaler.fit(X_embedded)
    #print(scaler.transform(X_embedded))

    X_embedded = scaler.transform(X_embedded)
    #print (X_embedded)

    for label, x, y in zip(labels, X_embedded[:,0], X_embedded[:,1]):
         #print (labelColor)
         colorlabel = labelColor[label]
         #print (colorlabel)
         plt.annotate(label, xy=(x, y), color = colorlabel)#, xytext=(-20,20))#,
        #     textcoords='offset points', ha='right', va='bottom',
        #     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        #     arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

    pylab.savefig('fig'+name+'.png')
    pylab.close()

    plt.show()
    # plt.scatter(X_embedded[:,0], X_embedded[:,1]) # plotting by columns
    # plt.show()


