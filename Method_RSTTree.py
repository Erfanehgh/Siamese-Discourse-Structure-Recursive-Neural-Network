import os
import re
import ClassNode
import numpy as np
import math
from Method_NeuralNets import tanh

from Method_NeuralNets import feedforward_act, apply_attention
from Methods_Preprocessing import preprocessor1
from Method_WV import WordAveraging

def sortEduKey(eduKeys, reverse):
    eduKeys = [int(x) for x in eduKeys]
    eduKeys = (sorted(eduKeys, reverse=reverse))
    return eduKeys


def readTree(filePath, W1, WV, dim, activationFunc):
    EDUs = {}
    EDUs_main = {}
    if os.path.exists(filePath):
        if os.stat(filePath).st_size > 0:
            file_1 = open(filePath, "r")

            for line in file_1:
                line = re.sub(r"[\n(\[\])]", "", line)
                arr = re.split("\s*,\s*", line)
                arr2 = re.split("\'", line)

                relation = arr2[len(arr2) - 2]
                hierarchy = arr2[len(arr2) - 4]

                if arr[0] == arr[1]:
                    if int(arr[0]) / 10 != 0:
                        arr[0] = "9" + arr[0]
                    vector = WordAveraging(preprocessor1(re.sub(r"[\n,.:'(\[\])]", "", arr2[1])), WV, dim)
                    #vector = tanh (vector)
                    EDUs_main[arr[0]] = ClassNode.Node(True, False, 0, 0, "", hierarchy, relation, vector, "")
                    EDUs[arr[0]] = "hi"
                else:
                    numconcat = ""
                    for num in range(int(arr[0]), int(arr[1]) + 1):
                        if num / 10 != 0:
                            num = "9" + str(num)
                        numconcat += repr(int(num))
                    childs = {}
                    i = 1

                    numconcat2 = numconcat
                    EDUitem = sortEduKey(EDUs_main.keys(), reverse=True)
                    for key in EDUitem:
                        key = str(key)
                        if numconcat != '' and key in numconcat:
                            childs[i] = EDUs_main[key].vector
                            if i == 1:
                                rightChild = key
                                EDUs_main[key].child = "right"

                            elif i == 2:
                                leftChild = key
                                EDUs_main[key].child = "left"

                            i = 2
                            numconcat = numconcat.replace(key, "")
                            del EDUs[key]

                    EDUs[numconcat2] = "hi"
                    vector = feedforward_act(np.concatenate([childs[2], childs[1]], 0), W1, activationFunc)
                    EDUs_main[numconcat2] = ClassNode.Node(False, False, leftChild, rightChild, "", hierarchy, relation,
                                                           vector, "")

        eduKey = EDUs.keys()
        eduKey = list(sorted(eduKey))
        if len(eduKey) > 1:
            vector = feedforward_act(np.concatenate([EDUs_main[eduKey[0]].vector, EDUs_main[eduKey[1]].vector], 0), W1,
                                     activationFunc)
            EDUs[eduKey[0] + eduKey[1]] = "hi"
            EDUs_main[eduKey[0]].child = "left"
            EDUs_main[eduKey[1]].child = "right"
            EDUs_main[eduKey[0] + eduKey[1]] = ClassNode.Node(False, True, eduKey[0], eduKey[1], "", "", "", vector, "")
            del EDUs[eduKey[0]]
            del EDUs[eduKey[1]]

        return EDUs_main

    else:
        return EDUs_main

def readTree_att_Scalar(filePath, W1, WV, dim, hierarchyType, attScaler, activationFunc):

    EDUs = {}
    EDUs_main = {}
    if os.path.exists(filePath):
        if  os.stat(filePath).st_size > 0 :
            file_1 = open(filePath, "r")

            for line in file_1:
                line = re.sub(r"[\n(\[\])]", "", line)
                arr = re.split("\s*,\s*",line)
                arr2 = re.split("\'", line)

                relation = arr2[len(arr2)-2]
                hierarchy = arr2[len(arr2)-4]

                if arr[0] == arr[1]:

                    if int(arr[0]) / 10 != 0:
                        arr[0] = "9" + arr[0]
                    vector = WordAveraging(preprocessor1(re.sub(r"[\n,.:'(\[\])]", "", arr2[1])), WV, dim)
                    #vector = tanh(vector)
                    if hierarchy == hierarchyType:
                        vector = np.multiply(attScaler, vector)
                    EDUs_main [arr[0]] = ClassNode.Node(True, False, 0, 0, "", hierarchy, relation, vector, "")
                    EDUs[arr[0]] = "hi"

                else:
                    numconcat = ""
                    for num in range(int(arr[0]), int(arr[1])+1):
                        if num/10 !=0 :
                                num="9"+str(num)
                        numconcat += repr(int(num))
                    childs={}
                    i=1

                    numconcat2 = numconcat
                    EDUitem =  sortEduKey(EDUs_main.keys(), reverse=True)
                    for key in EDUitem:
                        key = str(key)
                        if numconcat!='' and key in numconcat:
                            childs[i] = EDUs_main[key].vector
                            if i==1:
                                rightChild = key
                                EDUs_main[key].child = "right"

                            elif i==2:
                                leftChild = key
                                EDUs_main[key].child = "left"
                            #print len(childs[i]), key
                            i = 2
                            numconcat = numconcat.replace(key,"")

                            del EDUs[key]


                    EDUs[numconcat2]="hi"


                    vector = feedforward_act(np.concatenate([childs[2], childs[1]], 0), W1, activationFunc)

                    if hierarchy == hierarchyType:
                        vector = np.multiply(attScaler, vector)

                    EDUs_main[numconcat2]=ClassNode.Node(False, False, leftChild, rightChild, "" , hierarchy, relation, vector, "")

        eduKey=EDUs.keys()
        eduKey = list(sorted(eduKey))
        if len(eduKey)>1:
            vector = feedforward_act(np.concatenate([EDUs_main[eduKey[0]].vector, EDUs_main[eduKey[1]].vector], 0), W1, activationFunc)
            EDUs[eduKey[0]+eduKey[1]] = "hi"
            EDUs_main[eduKey[0]].child = "left"
            EDUs_main[eduKey[1]].child = "right"

            if hierarchy == hierarchyType:
                vector = np.multiply(attScaler, vector)
            EDUs_main[eduKey[0]+eduKey[1]] = ClassNode.Node(False, True, eduKey[0], eduKey[1],"", "", "", vector, "")
            del EDUs[eduKey[0]]
            del EDUs[eduKey[1]]

        return EDUs_main

    else:
        return EDUs_main

def readTree_att_NSWeight(filePath, W1, WV, dim, activationFunc):
    #print filePath
    EDUs = {}
    EDUs_main = {}
    if os.path.exists(filePath):
        if  os.stat(filePath).st_size > 0 :
            file_1 = open(filePath, "r")

            for line in file_1:
                line = re.sub(r"[\n(\[\])]", "", line)
                arr = re.split("\s*,\s*",line)
                arr2 = re.split("\'", line)

                relation = arr2[len(arr2)-2]
                hierarchy = arr2[len(arr2)-4]

                if arr[0] == arr[1]:

                    if int(arr[0]) > 9:
                        arr[0] = "9" + arr[0]
                    vector = WordAveraging(preprocessor1(re.sub(r"[\n,.:'(\[\])]", "", arr2[1])), WV, dim)
                    # if np.absolute(max(vector))>2:
                    #     print ("vayyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
                    # if np.sum(np.absolute(vector))/dim >=0.8:
                    #     for item in EDUs_main.keys():
                    #         print (item, " ", EDUs_main[item].vector)
                    #     print ("khodaaaaaaaaaaaaaaaaaa")
                    vector = tanh (vector)

                    EDUs_main [arr[0]] = ClassNode.Node(True, False, 0, 0, "", hierarchy, relation, vector, "")
                    EDUs[arr[0]] = "hi"

                else:
                    numconcat = ""
                    for num in range(int(arr[0]), int(arr[1])+1):
                        if num > 9:
                                num = "9"+str(num)
                        numconcat += repr(int(num))
                    childs={}
                    i=1

                    numconcat2 = numconcat
                    EDUitem =  sortEduKey(EDUs_main.keys(), reverse=True)
                    for key in EDUitem:
                        key = str(key)
                        if numconcat!='' and key in numconcat:
                            childs[i] = EDUs_main[key].vector

                            if i==1:
                                rightChild = key
                                EDUs_main[key].child = "right"

                            elif i==2:
                                leftChild = key
                                EDUs_main[key].child = "left"

                            i = 2
                            numconcat = numconcat.replace(key,"")
                            del EDUs[key]

                    EDUs[numconcat2]="hi"
                    vector = feedforward_act((np.concatenate([childs[2], childs[1]], 0)), W1, activationFunc)
                    EDUs_main[numconcat2]=ClassNode.Node(False, False, leftChild, rightChild, "" , hierarchy, relation, vector, "")

        eduKey=EDUs.keys()
        eduKey = list(sorted(eduKey))
        if len(eduKey)>1:

            EDU_0 = EDUs_main[eduKey[0]].vector
            EDU_1 = EDUs_main[eduKey[1]].vector

            vector = feedforward_act((np.concatenate([EDU_0, EDU_1], 0)), W1, activationFunc)
            #vector = np.concatenate([vector, [1]], 0)
            EDUs[eduKey[0]+eduKey[1]] = "hi"

            EDUs_main[eduKey[0]].child = "left"
            EDUs_main[eduKey[1]].child = "right"

            # if hierarchy == "Nucleus":
            #     vector = np.matmul(vector, WNu)
            # elif hierarchy == "Satellite":
            #     vector = np.matmul(vector, WSat)


            EDUs_main[eduKey[0]+eduKey[1]] = ClassNode.Node(False, True, eduKey[0], eduKey[1],"", "", "", vector, "")
            del EDUs[eduKey[0]]
            del EDUs[eduKey[1]]

        return EDUs_main

    else:
        return EDUs_main


def readAllTrees_IR(path_Folder, WV, dim):
    allEDUs = {}
    FileList = os.listdir(path_Folder)

    for edNum in range(0, len(FileList)):
        #print (edNum)
        path_File = path_Folder + FileList[edNum]
        EDUs = {}
        EDUs_main = {}
        if os.path.exists(path_File):
            if  os.stat(path_File).st_size > 0 :
                file_1 = open(path_File, "r")

                for line in file_1:
                    line = re.sub(r"[\n(\[\])]", "", line)
                    arr = re.split("\s*,\s*", line)
                    arr2 = re.split("\'", line)
                    relation = arr2[len(arr2) - 2]
                    hierarchy = arr2[len(arr2) - 4]

                    if arr[0] == arr[1]:

                        if int(arr[0]) > 9:
                            # print (arr[0])
                            arr[0] = "9" + arr[0]
                        vector = WordAveraging(preprocessor1(re.sub(r"[\n,.:'(\[\])]", "", arr2[1])), WV, dim)
                        vector = np.tanh (vector)
                        EDUs_main[arr[0]] = ClassNode.Node(True, False, 0, 0, "", hierarchy, relation, vector, "")
                        EDUs[arr[0]] = "hi"

                    else:
                        numconcat = ""
                        for num in range(int(arr[0]), int(arr[1]) + 1):
                            if num > 9:
                                # print (arr[0])
                                num = "9" + str(num)
                            numconcat += repr(int(num))
                        childs = {}
                        i = 1

                        numconcat2 = numconcat
                        EDUitem = sortEduKey(EDUs_main.keys(), reverse=True)

                        for key in EDUitem:
                            key = str(key)

                            if numconcat != '' and key in numconcat:
                                childs[i] = EDUs_main[key].vector
                                # print key, " ", childs[i]
                                #childs[i] = apply_attention(childs[i], EDUs_main[key].nodeHierarchy, WNu, WSat, activationFunc)
                                if i == 1:
                                    rightChild = key
                                    EDUs_main[key].child = "right"
                                    i = 2

                                elif i == 2:
                                    leftChild = key
                                    EDUs_main[key].child = "left"

                                numconcat = numconcat.replace(key, "")

                                del EDUs[key]

                        EDUs[numconcat2] = "hi"
                        #vector = feedforward_act((np.concatenate([childs[2], childs[1]], 0)), W1, activationFunc)
                        # vector = np.concatenate([vector, [1]], 0)
                        EDUs_main[numconcat2] = ClassNode.Node(False, False, leftChild, rightChild, "", hierarchy, relation, "", "")

                eduKey = EDUs.keys()
                eduKey = list(sorted(eduKey))
                if len(eduKey) > 1:
                    # EDU_0 = apply_attention(EDUs_main[eduKey[0]].vector, EDUs_main[eduKey[0]].nodeHierarchy, WNu, WSat, activationFunc)
                    # EDU_1 = apply_attention(EDUs_main[eduKey[1]].vector, EDUs_main[eduKey[1]].nodeHierarchy, WNu, WSat, activationFunc)
                    #
                    # vector = feedforward_act((np.concatenate([EDU_0, EDU_1], 0)), W1, activationFunc)
                    # vector = np.concatenate([vector, [1]], 0)
                    EDUs[eduKey[0] + eduKey[1]] = "hi"

                    EDUs_main[eduKey[0]].child = "left"
                    EDUs_main[eduKey[1]].child = "right"

                    EDUs_main[eduKey[0] + eduKey[1]] = ClassNode.Node(False, True, eduKey[0], eduKey[1], "", "", "", "", "")
                    del EDUs[eduKey[0]]
                    del EDUs[eduKey[1]]

            allEDUs[FileList[edNum]] = EDUs_main
        else:
            allEDUs[FileList[edNum]] = EDUs_main

    return allEDUs

def update_EDU(EDUs, W1, dim, activationFunc):
    eduKey = sortEduKey(EDUs.keys(), reverse=False)
    eduKey = map(str, eduKey)
    for key in eduKey:
        node = EDUs[key]
        if node.isLeaf == False:
            EDU_0 = EDUs[node.leftChild].vector
            EDU_1 = EDUs[node.rightChild].vector
            vector = feedforward_act((np.concatenate([EDU_0, EDU_1], 0)), W1, activationFunc)
            EDUs[key].vector = vector

    return EDUs