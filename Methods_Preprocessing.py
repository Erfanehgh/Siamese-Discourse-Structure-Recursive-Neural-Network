'''
Text_Preprocessing
'''
import re, numpy as np
import os
import shutil
import math
from random import shuffle
import random

def preprocessor1(words):
    pre_word = re.sub(r"[^a-zA-Z]", " ", words.lower())
    pre_word = re.sub("\\s{2,}", " ", pre_word)
    return pre_word.lower()

def sepValidation(path_Folder):

    posFileList_main = os.listdir(path_Folder + "pos/")
    negFileList_main = os.listdir(path_Folder + "neg/")
    posFileList = []
    negFileList = []
    posFileList_val = []
    negFileList_val = []
    numberofSamples = min(len(posFileList_main), len(negFileList_main))

    for j in range(0, numberofSamples):
        if j%10==0:
            posFileList_val.append(posFileList_main[j])
            negFileList_val.append(negFileList_main[j])
        else :
            posFileList.append(posFileList_main[j])
            negFileList.append(negFileList_main[j])

    return posFileList, negFileList, posFileList_val, negFileList_val

def sepValidation_file(path_Folder_src, path_Folder_des):

    posFileList_main = os.listdir(path_Folder_src + "pos/")
    negFileList_main = os.listdir(path_Folder_src + "neg/")
    numberofSamples = min(len(posFileList_main), len(negFileList_main))

    for j in range(0, numberofSamples):
        if j % 10 == 0:
            shutil.move(path_Folder_src+ "pos/"+posFileList_main[j], path_Folder_des+"pos/")
            shutil.move(path_Folder_src+ "neg/"+negFileList_main[j], path_Folder_des+"neg/")

def sepValidation_file_IR(path_Folder_src, path_Folder_des):

    FileList_main = os.listdir(path_Folder_src)
    FileList_main.sort()
    numberofSamples = len(FileList_main)

    for j in range(0, numberofSamples):
        if j % 10 == 0:
            shutil.move(path_Folder_src  + FileList_main[j], path_Folder_des)
            #shutil.move(path_Folder_src + "neg/" + negFileList_main[j], path_Folder_des + "neg/")

#sepValidation_file_IR("/home/erfaneh/Desktop/Drives/Datasets/bbc/All_Train/", "/home/erfaneh/Desktop/Drives/Datasets/bbc/All_Validation/")


def fileSampling(path_Folder, classes, outputFileName, numberOfSample):

    pairs = []
    OutputFile = open(outputFileName, "w")

    numberOfClasses = len(classes)
    data = np.zeros([numberOfClasses, numberOfClasses])
    #print type(data)

    FileList = os.listdir(path_Folder)
    numberofSamples = len(FileList)

    for j in range(0, numberofSamples):
        test_class = FileList[j].split("-")[1]
        for k in range(0, numberofSamples):
            train_class = FileList[k].split("-")[1]
            if test_class == train_class:
                index_test = classes.index(test_class)
                index_train = classes.index(train_class)
                data[index_test, index_train] =  data[index_test, index_train] + 1
    print (data)

    for j in range(0, numberOfClasses):
        for k in range(0, numberOfClasses):
            if j != k:
                data[j][k] = math.floor((data [j][j])/(numberOfClasses-1))

    print (data)


    sample = np.zeros([numberOfClasses, numberOfClasses])
    for j in range(0, numberofSamples):
        FileList = os.listdir(path_Folder)
        test_class = FileList[j].split("-")[1]
        index_test = classes.index(test_class)
        flag = np.zeros([numberOfClasses])
        FileList2 = os.listdir(path_Folder)
        shuffle(FileList2)

        for k in range(0, numberofSamples):

            train_class = FileList2[k].split("-")[1]
            index_train = classes.index(train_class)
            if index_test == index_train:
                if sample[index_test][index_train] <= data[index_test][index_train]:
                    if FileList[j] + " " + FileList2[k] not in pairs:
                        pairs.append(FileList[j] + " " + FileList2[k])
                        sample[index_test][index_train] += 1

            if index_test!=index_train:

                condition = int ((data[index_test][index_train])/(numberOfSample))
                if condition < 1:
                    condition = 2
                if sample[index_test][index_train] <= data[index_test][index_train] and flag[index_train] < condition:
                    if FileList[j] + " " + FileList2[k] not in pairs:
                        flag[index_train] += 1
                        pairs.append(FileList[j] + " " + FileList2[k])
                        sample[index_test][index_train] += 1

    print(sample)
    for pair in pairs:
        OutputFile.write("%s\n" % (pair))

    return pairs

def fileSampling_random(path_Folder, numberOfclasses, outputFileName, numberOfSimSamples):

    classes = {}
    pairs = []
    FileList = os.listdir(path_Folder)
    numberofSamples = len(FileList)

    OutputFile = open(outputFileName, "w")

    for i in range(0, numberofSamples):
        test_class = FileList[i].split("-")[1]
        if test_class in classes:
            classes[test_class].append(i)
        else:
            classes[test_class] = []
            classes[test_class].append(i)

    for i in range(0, numberofSamples):

        test_class = FileList[i].split("-")[1]
        numSamEachClass = len(classes[test_class])
        numSamForeachFile = int(numberOfSimSamples / numSamEachClass)
        index = random.sample(range(0, numSamEachClass), numSamForeachFile)

        for indx in index:
            pairs.append(FileList[i] + " " + FileList[classes[test_class][indx]])

        for key in classes.keys():
            if (key != test_class):
                numSamEachClass_not = len(classes[key])
                index_notSim = random.sample(range(0, numSamEachClass_not), np.minimum(numSamEachClass_not, int(np.ceil(numSamForeachFile/(numberOfclasses-1)))))
                for indx_notSim in index_notSim:
                    pairs.append(FileList[i] + " " + FileList[classes[key][indx_notSim]])

    for pair in pairs:
        OutputFile.write("%s\n" % (pair))

    return pairs, classes


def fileSelection(samplePath, path, DestPath):
    sampleList = os.listdir(samplePath)
    fileList = os.listdir(path)
    for file in fileList:
        if file.replace(".txt", ".txt.bracketsTxt") in sampleList:
            shutil.copyfile(path + file, DestPath + file)




#fileSelection("/home/erfaneh/Desktop/Drives/Datasets/Reuters_original/Train_random/", "/home/erfaneh/Desktop/Drives/Datasets/Reuters_original/Train_8Topics/" ,"/home/erfaneh/Desktop/Drives/Datasets/Reuters_original/Train_8Topics_balance/")


# fileSampling_random ("/home/erfaneh/Desktop/Drives/Datasets/Reuters_original/Train_100/", 8, "pair.txt", 10000)

def tag_bbc_data(File_path):
    listFile = os.listdir(File_path)
    i = 0
    for file in listFile:
        i += 1
        if (i%10 == 0):
            os.rename(File_path+file, File_path+"test-"+File_path.split("/")[-2]+"-"+file)
        else:
            os.rename(File_path + file, File_path + "training-" + File_path.split("/")[-2] + "-" + file)


#tag_bbc_data("/home/erfaneh/Dropbox/bbc/sport/")