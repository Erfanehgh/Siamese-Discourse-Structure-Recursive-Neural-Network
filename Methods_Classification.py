'''
This file contains all classification Functions
'''
import os
import numpy as np
from math import isnan
from random import shuffle
from Method_NeuralNets import feedforward, feedforward_act, softmax_error, calculate_deltaW,  BpthroughTree, update_weight, MSE, non_softmax_error, feedforward_act_dropOut, feedforward_dropOut, dropOut, cross_entropy, dropcolrow
from Method_RSTTree import update_EDU, sortEduKey, readTree_att_Scalar, sortEduKey, readTree_att_NSWeight
from sklearn.metrics.pairwise import cosine_similarity
from itertools import groupby
import operator

global_outputActivation= "tanh"




def train_for_each_Sample_AttWeight (EDUs, EDUs_test, y, W1, W21, W22, eta, dim, activationFunc, dropOutPercent):

    W1_copy = W1.copy()
    W21_copy = W21.copy()
    W22_copy = W22.copy()

    indexNode = dropOut(len(W1[0]), dropOutPercent)
    #W1_doc = dropcolrow(W1, indexNode, False)

    #indexNode = dropOut(len(W1_query[0]), dropOutPercent)
    W1 = dropcolrow(W1, indexNode, False)

    indexNode2 = []
    indexNode2.extend(indexNode)
    indexNode2.extend(indexNode*2)

    W21 = dropcolrow(W21, indexNode2, True)

    eduKeys_test = sortEduKey(EDUs_test.keys(), reverse=True)
    input2_test = EDUs_test[str(eduKeys_test[0])].vector

    eduKeys = sortEduKey(EDUs.keys(), reverse=True)
    input2 = EDUs[str(eduKeys[0])].vector

    input = (np.concatenate([input2, input2_test], 0))
    #y_in1 = feedforward(input, W21)
    output1 = feedforward_act(input, W21, activationFunc)
    y_in = feedforward(output1, W22)
    output = feedforward_act(output1, W22, global_outputActivation)

    error_soft = softmax_error(y, output, y_in, global_outputActivation)
    delta_W22 = calculate_deltaW(error_soft, output1)

    error_hidden = non_softmax_error(error_soft, W22, input, W21, activationFunc)
    delta_W21 = calculate_deltaW(error_hidden, input)

    delta_W1_doc = BpthroughTree(EDUs, error_hidden, W1, W21, dim, activationFunc, True)
    delta_W1_query = BpthroughTree(EDUs_test, error_hidden, W1, W21, dim, activationFunc, False)

    #print ("=============== : ", np.sum(delta_W1_doc[:, indexNode]))
    delta_W1_doc = dropcolrow(delta_W1_doc, indexNode, False)
    delta_W1_query = dropcolrow(delta_W1_query, indexNode, False)
    delta_W21 = dropcolrow(delta_W21, indexNode2, True)

    delta_W = np.divide(np.add(delta_W1_doc, delta_W1_query), 2)

    W21 = update_weight(eta, W21_copy, delta_W21)
    W22 = update_weight(eta, W22_copy, delta_W22)
    W1 = update_weight(eta, W1_copy, delta_W)


    return W1, W21, W22



def train_AttWeight_pair(allEDUs, WV, dim, W1, W21, W22, eta, OutputFile, activationFunc, pairs, dropOutPercent):

    numberofSamples = len(pairs)
    print (len(pairs))
    #numberofSamples = 50
    #print (pairs[0])
    shuffle (pairs)
    #print (pairs[0])
    sim = 0
    notsim = 0

    for pair in pairs: #j in range(0, numberofSamples):

        #indexW1 = dropOut(W1_doc, dropOutPercent)
        # W1_doc = np.multiply(W1_doc, indexW1)
        # W1_query = np.multiply(W1_query, indexW1)
        #
        # indexW21 = dropOut(W21, dropOutPercent)
        # W21 = np.multiply(W21, indexW21)
        # indexW22 = dropOut(W22, dropOutPercent)
        # W22 = np.multiply(W22, indexW22)

        #print W1_query[2,1:5]
        # if(((sim + notsim) % 3000 == 0) and (eta>0.0001)):
        #     eta = eta / (1 + (float((sim + notsim)/3000) / 50))
        #     print (eta)

        filenames = pair.split(' ')
        EDUs_test = allEDUs[filenames[0]]
        EDUs_test = update_EDU(EDUs_test, W1, dim, activationFunc)

        EDUs = allEDUs[filenames[1]]
        EDUs = update_EDU(EDUs, W1, dim, activationFunc)

        if len(EDUs_test) == 0 or len(EDUs)==0 :
            print (filenames)

        if (filenames[0].split("-")[1] == filenames[1].split("-")[1]):
            sim += 1
            if (len(W22[0]) == 1):
                y = [1.0]
            else:
                y = [0.8, -0.8]
            #y = [1.0, -1.0]
            W1, W21, W22 = train_for_each_Sample_AttWeight(EDUs, EDUs_test, y, W1, W21, W22, eta, dim, activationFunc, dropOutPercent)
        else:
            notsim += 1
            if (len(W22[0]) == 1):
                y = [-1.0]
            else:
                y = [-0.8, 0.8]
            #y = [-1.0, 1.0]
            W1, W21, W22 = train_for_each_Sample_AttWeight(EDUs, EDUs_test, y, W1, W21, W22, eta, dim, activationFunc, dropOutPercent)

    print ("=============================================================================")
    print (sim, notsim)
    return W1, W21, W22, eta


'''
F1 avg on two class
'''

def test_AttWeight_DrHarati_pair(allEDUs, mode, WV, dim,  W1, W21, W22, OutputFile, iteration, activationFunc, pairs):
    numberofSamples = len(pairs)
    #numberofSamples_test = 50
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    sim = 0
    notsim = 0

    for pair in pairs: #j in range(0, numberofSamples):
        filenames = pair.split(' ')
        EDUs_test = allEDUs[filenames[0]]
        EDUs_test = update_EDU(EDUs_test, W1, dim, activationFunc)
        eduKeys_test = sortEduKey(EDUs_test.keys(), reverse=True)
        input2_test = EDUs_test[str(eduKeys_test[0])].vector

        EDUs = allEDUs[filenames[1]]
        EDUs = update_EDU(EDUs, W1, dim, activationFunc)
        eduKeys = sortEduKey(EDUs.keys(), reverse=True)
        input2 = EDUs[str(eduKeys[0])].vector

        input = np.concatenate([input2, input2_test], 0)
        output1 = feedforward_act(input, W21, activationFunc)
        output = feedforward_act(output1, W22, global_outputActivation)

        if (filenames[0].split("-")[1] == filenames[1].split("-")[1]):
            sim += 1
            if (sim % 500 == 0):
                print("Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
            #if output[0] > output[1]:
            if (len(W22[0]) == 1):
                if output[0] > 0:
                    # print("Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
                    tp += 1
                else:
                    # print("Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
                    fn += 1
            else:
                if output[0] > output[1]:
                    # print("Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
                    tp += 1
                else:
                    # print("Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
                    fn += 1
        else:
            notsim += 1
            if (notsim % 500 == 0):
                print("Not-Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
            #if output[0] < output[1]:
            if (len(W22[0]) == 1):
                if output[0] < 0:
                    # print("Not-Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
                    tn += 1
                else:
                    # print("Not-Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
                    fp += 1
            else:
                if output[0] < output[1]:
                    # print("Not-Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
                    tn += 1
                else:
                    # print("Not-Similar ", filenames[0].split("-")[1], filenames[1].split("-")[1], output)
                    fp += 1

    print (sim, notsim)
    accuracy = float(tp + tn) / (tp + tn + fp + fn)
    precision, recall, F1 = calculate_eval_metrics(tp, tn, fp, fn)
    print(iteration, " ", mode , " ", tp, " ", tn, " ", fp, " ", fn, " ", accuracy, " ", precision, " ", recall, " ", F1)
    OutputFile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (iteration, mode, tp, tn, fp, fn, accuracy, precision, recall, F1))




'''
Calculate Error
'''
def calculateError_validation(path_Folder, mode, WV, dim,  W1_doc, W1_query, W2, OutputFile, iteration, activationFunc):
    FileList = os.listdir(path_Folder)
    numberofSamples = len(FileList)

    target_list=[]
    output_list=[]

    for j in range(0, numberofSamples):
        path_File_test = path_Folder + FileList[j]
        EDUs_test = readTree_att_NSWeight(path_File_test, W1_query, WV, dim, activationFunc)
        eduKeys_test = sortEduKey(EDUs_test.keys(), reverse=True)
        input2_test = EDUs_test[str(eduKeys_test[0])].vector

        for k in range(0, numberofSamples):
            path_File_test = path_Folder + FileList[k]
            EDUs = readTree_att_NSWeight(path_File_test, W1_doc, WV, dim, activationFunc)
            eduKeys = sortEduKey(EDUs.keys(), reverse=True)
            input2 = EDUs[str(eduKeys[0])].vector

            input = np.concatenate([input2, input2_test], 0)
            output = feedforward_act(input, W2, global_outputActivation)
            #print output

            if (FileList[j].split("-")[1] == FileList[k].split("-")[1]):
                y = [1.0]
                target_list.append(y)
                output_list.append(output)
            else:
                if (k % 5 == 0):
                    y = [0.0]
                    target_list.append(y)
                    output_list.append(output)



    totall_Err = cross_entropy(target_list, output_list)
    print(iteration, " ", mode , " ", totall_Err)
    OutputFile.write("%s,%s,%s\n" % (iteration, mode, totall_Err))
    return totall_Err

def calculateError_validation_pair(allEDUs, mode, WV, dim,  W1, W21, W22, OutputFile, iteration, activationFunc, pairs):
    #FileList = os.listdir(path_Folder)
    numberofSamples = len(pairs)

    target_list=[]
    output_list=[]

    for pair in pairs: #j in range(0, numberofSamples):
        filenames = pair.split(' ')
        #path_File_test = path_Folder + filenames[0]
        EDUs_test = allEDUs[filenames[0]]#readTree_att_NSWeight(path_File_test, W1_query, WV, dim, activationFunc)
        EDUs_test = update_EDU(EDUs_test, W1, dim, activationFunc)
        eduKeys_test = sortEduKey(EDUs_test.keys(), reverse=True)
        input2_test = EDUs_test[str(eduKeys_test[0])].vector

        #path_File_test = path_Folder + filenames[1]
        EDUs = allEDUs[filenames[1]]#readTree_att_NSWeight(path_File_test, W1_doc, WV, dim, activationFunc)
        EDUs = update_EDU(EDUs, W1, dim, activationFunc)
        eduKeys = sortEduKey(EDUs.keys(), reverse=True)
        input2 = EDUs[str(eduKeys[0])].vector

        input = np.concatenate([input2, input2_test], 0)
        output1 = feedforward_act(input, W21, activationFunc)
        output = feedforward_act(output1, W22, global_outputActivation)

        if (filenames[0].split("-")[1] == filenames[1].split("-")[1]):
            if len(W22[0]) == 1:
                y = [1.0]
            else:
                y = [0.8, -0.8]

            target_list.append(y)
            output_list.append(output)
        else:
            if len(W22[0]) == 1:
                y = [-1.0]
            else:
                y = [-0.8, 0.8]

            target_list.append(y)
            output_list.append(output)

    totall_Err = MSE(output_list, target_list)
    print(iteration, " ", mode , " ", totall_Err)
    OutputFile.write("%s,%s,%s\n" % (iteration, mode, totall_Err))
    return totall_Err

def calculateError_validation_EDUs(allEDUs, mode, WV, dim,  W1 , W2, OutputFile, WSat, WNu, iteration, activationFunc):
    target_list = np.zeros([0, 2])
    output_list = np.zeros([0, 2])
    EDU_key = allEDUs.keys()

    for EDUid in EDU_key:
        EDUs = allEDUs [EDUid]
        EDUs = update_EDU(EDUs, W1, WSat, WNu, dim, activationFunc)
        if (len(EDUs) > 0 and EDUid>0):
            y = [1.0, 0]

            #eduKeys = sortEduKey(EDUs.keys(), reverse=True)
            eduKeys = sortEduKey(EDUs.keys(), reverse=True)

            input2 = EDUs[str(eduKeys[0])].vector

            output = feedforward_act(input2, W2, global_outputActivation)
            target_list = np.concatenate((target_list, [y]), 0)
            output_list = np.concatenate((output_list, [output]), 0)

        if (len(EDUs) > 0 and EDUid < 0):
            y = [0, 1.0]
            # eduKeys = sortEduKey(EDUs.keys(), reverse=True)
            eduKeys = sortEduKey(EDUs.keys(), reverse=True)

            input2 = EDUs[str(eduKeys[0])].vector
            #y_in = feedforward(input2, W2)
            output = feedforward_act(input2, W2, global_outputActivation)
            target_list = np.concatenate((target_list, [y]), 0)
            output_list = np.concatenate((output_list, [output]), 0)

    totall_Err = MSE(target_list, output_list)
    print(iteration, " ", mode , " ", totall_Err)
    OutputFile.write("%s,%s,%s\n" % (iteration, mode, totall_Err))
    return totall_Err

'''
Calculate F1
'''

def calculate_eval_metrics(tp, tn, fp, fn):
    if tp == 0 or tn == 0:
        return 0, 0, 0
    else:
        recall_pos = float(tp) / (tp + fn)
        recall_neg = float(tn) / (tn + fp)
        precision_pos = float(tp) / (tp + fp)
        precision_neg = float(tn) / (tn + fn)
        F1_pos = 2 * (float(precision_pos * recall_pos)) / (precision_pos + recall_pos)
        F1_neg = 2 * (float(precision_neg * recall_neg)) / (precision_neg + recall_neg)
        F1_AVG = (F1_neg+ F1_pos)/2
        pre_AVG = (precision_neg+ precision_pos)/2
        recall_AVG = (recall_neg+ recall_pos)/2
        return  pre_AVG, recall_AVG, F1_AVG


def  IR_Evaluation(EDUs_Test, EDUs_Train, path_file_Test, path_file_Train, W1, WV, dim, activationFunc, iteration, OutputFile, numberofDoc):

    precisions = []

    TestFileList = os.listdir(path_file_Test)
    TrainFileList = os.listdir(path_file_Train)

    trainDocuments = []
    trainDocuments_label = []

    for tr in range(0, len(TrainFileList)):
        EDUs_tr = EDUs_Train[TrainFileList[tr]]
        EDUs_tr = update_EDU(EDUs_tr, W1, dim, activationFunc)
        # path_File_train = path_file_Train + TrainFileList[tr]
        # EDUs_train = readTree_att_NSWeight(path_File_train, W1_doc, WV, dim, activationFunc)

        eduKeys = sortEduKey(EDUs_tr.keys(), reverse=True)
        trainRep = EDUs_tr[str(eduKeys[0])].vector
        trainDocuments.append(trainRep)
        trainDocuments_label.append(TrainFileList[tr].split("-")[1])

    trainDocuments = np.array(trainDocuments)
    trainDocuments_label = np.array(trainDocuments_label)
    #print trainDocuments_label

    for te in range(0, len(TestFileList)):
        EDUs_te = EDUs_Test[TestFileList[te]]
        EDUs_te = update_EDU(EDUs_te, W1, dim, activationFunc)
        eduKeys = sortEduKey(EDUs_te.keys(), reverse=True)
        testRep = EDUs_te[str(eduKeys[0])].vector
        testRep = np.array(testRep).reshape(1, -1)

        if (np.isnan(testRep).any()):
            print (TestFileList[te])

        distances = cosine_similarity(testRep, trainDocuments)
        main_label = TestFileList[te].split("-")[1]
        predicted_labels = trainDocuments_label[(((distances.argsort())[0])[-1*numberofDoc:][::-1])]
        predicted_labels = sorted(predicted_labels)
        predicts = {x: predicted_labels.count(x) for x in predicted_labels}
        main_predict = max(predicts.items(), key=operator.itemgetter(1))[0]

        if main_label in predicts:
            pre_main_label = float(predicts[main_label]) / numberofDoc#len(predicted_labels)
            precisions.append(pre_main_label)
        else:
            precisions.append(0)
    avg_pre = float(sum(precisions)) / len(precisions)
    print ("average", avg_pre)

    if (OutputFile!= None):
        OutputFile.write("%s,%s\n" % (iteration, avg_pre))

def  IR_Evaluation_Test(EDUs_Test, EDUs_Train, path_file_Test, path_file_Train, W1, WV, dim, activationFunc, iteration, OutputFile, numberofDoc):

    precisions = []

    TestFileList = os.listdir(path_file_Test)
    TrainFileList = os.listdir(path_file_Train)

    trainDocuments = []
    trainDocuments_label = []

    for tr in range(0, len(TrainFileList)):
        EDUs_tr = EDUs_Train[TrainFileList[tr]]
        EDUs_tr = update_EDU(EDUs_tr, W1, dim, activationFunc)
        # path_File_train = path_file_Train + TrainFileList[tr]
        # EDUs_train = readTree_att_NSWeight(path_File_train, W1_doc, WV, dim, activationFunc)

        eduKeys = sortEduKey(EDUs_tr.keys(), reverse=True)
        trainRep = EDUs_tr[str(eduKeys[0])].vector
        trainDocuments.append(trainRep)
        trainDocuments_label.append(TrainFileList[tr].split("-")[1])

    trainDocuments = np.array(trainDocuments)
    trainDocuments_label = np.array(trainDocuments_label)
    #print trainDocuments_label

    for te in range(0, len(TestFileList)):
        EDUs_te = EDUs_Test[TestFileList[te]]
        EDUs_te = update_EDU(EDUs_te, W1, dim, activationFunc)
        eduKeys = sortEduKey(EDUs_te.keys(), reverse=True)
        testRep = EDUs_te[str(eduKeys[0])].vector
        testRep = np.array(testRep).reshape(1, -1)

        if (np.isnan(testRep).any()):
            print (TestFileList[te])

        distances = cosine_similarity(testRep, trainDocuments)
        main_label = TestFileList[te].split("-")[1]
        predicted_labels = trainDocuments_label[(((distances.argsort())[0])[-1*numberofDoc:][::-1])]


        print (predicted_labels)

        predicted_labels = sorted(predicted_labels)

        predicts = {x: predicted_labels.count(x) for x in predicted_labels}
        main_predict = max(predicts.items(), key=operator.itemgetter(1))[0]

        print (main_label, main_predict)

        if main_label in predicts:
            pre_main_label = float(predicts[main_label]) / (numberofDoc)
            precisions.append(pre_main_label)
        else:
            precisions.append(0)

    avg_pre = float(sum(precisions)) / len(precisions)
    print ("average", avg_pre)

    if (OutputFile!= None):
        OutputFile.write("%s,%s\n" % (iteration, avg_pre))


