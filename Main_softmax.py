

import numpy as np
from random import shuffle
from Method_WV import readWV
from Methods_Preprocessing import fileSampling, fileSampling_random
from Method_RSTTree import readAllTrees_IR
from Method_NeuralNets import initialize_weight_variable
from Methods_Classification import train_AttWeight_pair, calculateError_validation_pair,  test_AttWeight_DrHarati_pair, IR_Evaluation
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


#path_WV ="C:/Users/eg8qe/PycharmProjects/all_Data/"
#WV=readWV(path_WV+"glove.6B/glove.6B.200d.txt", stop)
#main_path ="C:/Users/eg8qe/PycharmProjects/all_Data/"
WV = readWV("/home/erfaneh/Desktop/Drives/Datasets/WV/glove.6B.100d.txt", stop)
main_path = "/home/erfaneh/Desktop/Drives/Datasets/Reuters_original/"

path_Folder = main_path + "Train_random/"
path_Folder_test = main_path + "Test_random_balance/"
path_Folder_val = main_path + "Validation_random/"

#pairs_Train = fileSampling (path_Folder, ["earn", "trade", "acq", "money", "ship", "crude", "interest", "grain"], "pair_Train.txt", 100)
pairs_Train , classes = fileSampling_random(path_Folder, 8, "pair_Train.txt", 10000)
print (len(pairs_Train))
numberofDoc = 100
#pairs_Test = fileSampling (path_Folder_test, ["earn", "trade", "acq", "money", "ship", "crude", "interest", "grain"], "pair_Test.txt", 10)
pairs_Test , classes = fileSampling_random(path_Folder_test, 8, "pair_Test.txt", 105)
print (len(pairs_Test))

#pairs_Val = fileSampling (path_Folder_val, ["earn", "trade", "acq", "money", "ship", "crude", "interest", "grain"], "pair_Val.txt", 5)
pairs_Val , classes = fileSampling_random(path_Folder_val, 8, "pair_Val.txt", 32)
print (len(pairs_Val))


eta_0 = 0.0005
eta = 0.0005
decreasingSpeed = 100
dim = 100
hidden = 20
numberOfOutput = 2
activationFunc = "ReLU"
dropOutPercent = 0.5


if os.path.isfile("W1_File_softmax_newDat.txt"):
    print ("doroste")
    W1 = np.loadtxt("W1_File_softmax_newData_test133.txt")
    W21 = np.loadtxt("W21_File_softmax_newData_test133.txt")
    W22 = np.loadtxt("W22_File_softmax_newData_test133.txt")



else:
    W1 = initialize_weight_variable(2 * dim + 1, dim)
    W21 = initialize_weight_variable(2 * dim + 1, hidden)
    W22 = initialize_weight_variable(hidden + 1, numberOfOutput)

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



for i in range(34, 200):
    OutputFile = open("ou.csv", "a")
    OutputFile_val_err = open("val.csv", "a")
    OutputFile_avg_precision = open("avg.csv", "a")

    W1, W21, W22, eta = train_AttWeight_pair(EDUs_Train, WV, dim, W1, W21, W22, eta, OutputFile, activationFunc, pairs_Train, dropOutPercent)
    #eta = eta/2

    #Write_Weights(W1, W21, W22, "W1_File_softmax_newData_test" + str(i+100) + ".txt", "W21_File_softmax_newData_test" + str(i+100) + ".txt", "W22_File_softmax_newData_test" + str(i+100) + ".txt")
    print (datetime.strptime(str(datetime.now().time()).split('.')[0], FMT) - datetime.strptime(start_time, FMT))


    if i % 1 == 0:
        IR_Evaluation(EDUs_Test, EDUs_Train, path_Folder_test, path_Folder, W1, WV, dim, activationFunc, i, OutputFile_avg_precision, numberofDoc)
        error = calculateError_validation_pair(EDU_Validation, "validation", WV, dim, W1, W21, W22, OutputFile_val_err, i, activationFunc, pairs_Val)
        error = calculateError_validation_pair(EDUs_Train, "train", WV, dim, W1, W21, W22, OutputFile_val_err, i, activationFunc, pairs_Train)
        #error = calculateError_validation_pair(EDUs_Test, "validation", WV, dim, W1, W21, W22, OutputFile_val_err, i, activationFunc, pairs_Test)
        #errors.append(error)
        # print("======================================================================")
        test_AttWeight_DrHarati_pair(EDUs_Train, "train", WV, dim, W1, W21, W22, OutputFile, i, activationFunc, pairs_Train)
        # print("======================================================================")
        test_AttWeight_DrHarati_pair(EDU_Validation, "validation", WV, dim, W1, W21, W22, OutputFile, i, activationFunc, pairs_Val)
        # print("======================================================================")
        test_AttWeight_DrHarati_pair(EDUs_Test, "test", WV, dim, W1, W21, W22, OutputFile, i, activationFunc, pairs_Test)
        # print("======================================================================")
        OutputFile.close()
        OutputFile_val_err.close()
        #plotDocs(EDUs_Test, "AVG10Sample" + str(dim)+"it"+str(i))

        if (i < 200):
            eta = eta_0 / (1 + (float(i) / decreasingSpeed))
        else:
            eta = eta_0 / (1 + (math.pow(1.1, i)*float(i) / decreasingSpeed))
        print(eta)



