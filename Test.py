

import numpy as np
from random import shuffle
from Method_WV import readWV
from Methods_Preprocessing import fileSampling_random
from Method_RSTTree import readAllTrees_IR
from Method_NeuralNets import initialize_weight_variable
from Methods_Classification import train_AttWeight_pair, calculateError_validation_pair,  test_AttWeight_DrHarati_pair, IR_Evaluation, IR_Evaluation_Test
import os
import math
from Methods_ReadWrite import Write_Weights
from datetime import datetime
from T_SNE import plotDocs

'''

'''
FMT = '%H:%M:%S'
start_time = str(datetime.now().time()).split('.')[0]
print (start_time)

from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
#stop = []


colors = ['green', 'gold', 'crimson', 'darkblue', 'orangered', 'orchid', 'olive', 'dodgerblue']
classes = ["acq", "crude", "earn", "grain", "interest", "money", "ship", "trade"]
labelColor = {}

for i in range(0, len(classes)):
    labelColor[classes[i]] = colors[i]

#path_WV ="C:/Users/eg8qe/PycharmProjects/all_Data/"
#WV=readWV(path_WV+"glove.6B/glove.6B.200d.txt", stop)
#main_path ="C:/Users/eg8qe/PycharmProjects/all_Data/"
WV = readWV("/home/erfaneh/Desktop/Drives/Datasets/WV/glove.6B.100d.txt", stop)
# main_path = "/home/erfaneh/Desktop/Drives/Datasets/bbc/"
#
# path_Folder = main_path + "All_Train/"
# path_Folder_test = main_path + "All_Test/"
# path_Folder_val = main_path + "All_Validation/"

main_path = "/home/erfaneh/Desktop/Drives/Datasets/Reuters_original/"

path_Folder = main_path + "Train_random/"
path_Folder_test = main_path + "Test_random_balance/"
path_Folder_val = main_path + "Validation_random/"

numberofDoc = 100


eta_0 = 0.0005
eta = 0.0005
decreasingSpeed = 100
dim = 100
hidden = 20
numberOfOutput = 2
activationFunc = "ReLU"
dropOutPercent = 0.5



print ("doroste")
currentPath = "/home/erfaneh/Dropbox/BU/Siamese-BestWorking-16-April"
# W1 = np.loadtxt(currentPath + "/Siamese-uniqeWeight/W1_File_softmax_bbc109.txt")
# W21 = np.loadtxt(currentPath + "/Siamese-uniqeWeight/W21_File_softmax_bbc109.txt")
# W22 = np.loadtxt(currentPath + "/Siamese-uniqeWeight/W22_File_softmax_bbc109.txt")

W1 = np.loadtxt(currentPath + "/Siamese-uniqeWeight/W1_File_softmax_newData_test199.txt")
W21 = np.loadtxt(currentPath + "/Siamese-uniqeWeight/W21_File_softmax_newData_test199.txt")
W22 = np.loadtxt(currentPath + "/Siamese-uniqeWeight/W22_File_softmax_newData_test199.txt")


print (datetime.strptime(str(datetime.now().time()).split('.')[0], FMT) - datetime.strptime(start_time, FMT))

EDUs_Train = readAllTrees_IR(path_Folder,  WV, dim)
print ("Train Reading Done")
EDUs_Test = readAllTrees_IR(path_Folder_test, WV, dim)
print ("Test Reading Done")
EDU_Validation = readAllTrees_IR(path_Folder_val, WV, dim)
print ("Validation Reading Done")

print(len(EDUs_Train))
print(len(EDUs_Test))
print(len(EDU_Validation))


print ("All Reading Done")

print (datetime.strptime(str(datetime.now().time()).split('.')[0], FMT) - datetime.strptime(start_time, FMT))

OutputFile_avg_precision = open("Test.csv", "a")

IR_Evaluation_Test(EDUs_Test, EDUs_Train, path_Folder_test, path_Folder, W1, WV, dim, activationFunc, 1, OutputFile_avg_precision, numberofDoc)
plotDocs(EDUs_Test, "Validation_Reuters_new1", labelColor)