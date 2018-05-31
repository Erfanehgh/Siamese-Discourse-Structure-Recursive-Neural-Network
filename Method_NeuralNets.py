'''
This file contain all neutral Network related Functions
'''

from scipy.special import expit
import numpy as np
from math import sqrt, log, isnan, floor
from sklearn.metrics import mean_squared_error


def apply_attention(vector, hierarchy, WNu, WSat, activationFunc):
    #print "applyatten", len(vector)
    if hierarchy == "Nucleus":
        vector = feedforward_act(vector, WNu, activationFunc)
    elif hierarchy == "Satellite":
        vector = feedforward_act(vector, WSat, activationFunc)

    if len(vector) == 1:
        vector = np.array(vector)[0]

    return vector


def sortEduKey(eduKeys, reverse):
    eduKeys = [int(x) for x in eduKeys]
    eduKeys = (sorted(eduKeys, reverse=reverse))
    return eduKeys

'''
Activation Functions Methods and derivates
'''


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x):
    return expit(x)
  #return np.divide((1.0), np.add((1.0), np.exp(np.negative(x))))


def sigma(x):
    num = []
    for i in x[0]:
        num.append(sigmoid(i))

    return num

def sigmaprime(x):
    if type(x)==list:
        derivate = []
        for i in x:
            derivate.append(np.multiply(sigmoid(i), np.subtract((1.0), sigmoid(i))))
        return derivate
    else:
        np.multiply(sigmoid(x), np.subtract((1.0), sigmoid(x)))


def ReLU(x):
    Rel = []
    epsilon = 0.1
    for i in x:
        Rel.append(np.maximum(i,epsilon*(i)))
    return Rel

def dReLU(x):
    derivate = []
    for i in x:
        if i>0:
            derivate.append(1)
        else:
            derivate.append(0.1)
    return derivate

def softmax(output):
    exps = np.exp(output - np.max(output))
    return exps / np.sum(exps)

def Main_softmax(output):
    exps = np.exp(output)  #- np.max(output)
    n = exps / np.sum(exps)
    return n

def softmaxprime_Jac(y, y_hat):
    jac = np.zeros([len(y_hat), len(y_hat)])
    for i in range(len(y_hat)):
        for j in range(len(y_hat)):
            if i==j:
                jac[i][j] = y_hat[i]*(1-y_hat[i])
            else:
                jac[i][j] = -1 * y_hat[i] * y_hat[j]

    return jac



def delta_cross_entropy(y_hat, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is one hot encoded labels (num_examples x num_classes)
    """
    y = np.array(y).astype(int)
    m = y.shape[0]
    grad = (y_hat)

    grad[range(m),y] -= 1
    grad = grad/m
    return grad

def cross_entropy(y, y_hat):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is one hot encoded labels (num_examples x num_classes)
    """
    y = y.astype(int)
    m = y.shape[0]
    p = (y_hat)
    log_likelihood = -np.log(p)
    log_likelihood = np.multiply(log_likelihood, y)
    #log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss


def softmaxprime(y, y_hat):

    derivate = [1, 1]
    # dscores =  y_hat
    # dscores[range(1), y] -= 1
    # dscores /= num_examples
    # print y
    # print y_hat
    #derivate = np.subtract(y_hat, y)
    # print (derivate)

    return derivate
    # if y[0] > y[1]:
    #     return [1.0, 1.0]
    # else:
    #     return [1.0, 1.0]
    #derivate= y
    #derivate = np.subtract(y_hat, y)
    # print np.abs(sum(derivate))
    # if (sum(np.abs(derivate)))>1:
    #     return [0, 0]
    # print derivate
    #return derivate

'''
Weight initialization Function
'''
def initialize_weight_variable(d1, d2):
    #initial = np.divide(np.random.uniform(-1, 1,(d1, d2)).astype(np.float32), 10)
    initial = np.divide(np.random.normal(loc = 0, scale =0.1 , size = (d1, d2)).astype(np.float32), 1)
    return initial

'''
Feed forward Functions
'''
'''
if len((np.matmul(np.matrix(ab),W1)))== 1:
            return (np.tanh(np.matmul(np.matrix(ab),W1)).tolist())[0]
        return np.tanh(np.matmul(np.matrix(ab),W1))
'''
def feedforward_act(ab , W1, activationFunc):

    if len(ab) % 2 == 0:
        ab = np.concatenate([ab, [1]], 0)

    output = (np.matmul(np.matrix(ab), W1))
    if(activationFunc=="tanh"):
        if len(output) == 1:
            return (np.tanh(output).tolist())[0]
        return np.tanh(output)

    elif (activationFunc == "sig"):
        if len(output) == 1:
            return np.asarray((sigma(output))[0])[0]
        return np.asarray((sigma(output))[0])[0]

    elif (activationFunc == "ReLU"):
        if len(output) == 1:
            return (ReLU((output).tolist()[0]))
        return (ReLU(output))

    elif (activationFunc == "softmax"):
        if len(output) == 1:
            return (softmax(output.tolist()[0]))
        else:
            return (softmax(output))


def feedforward_act_dropOut(ab, W1, activationFunc, dropOutPercent):
    if dropOutPercent > 0:
        ab = dropOut(ab, dropOutPercent)

    if len(ab) % 2 == 0:
        ab = np.concatenate([ab, [1]], 0)

    output = (np.matmul(np.matrix(ab), W1))
    if (activationFunc == "tanh"):
        if len(output) == 1:
            return (np.tanh(output).tolist())[0]
        return np.tanh(output)

    elif (activationFunc == "sig"):
        if len(output) == 1:
            return np.asarray((sigma(output))[0])[0]
        return np.asarray((sigma(output))[0])[0]

    elif (activationFunc == "ReLU"):
        if len(output) == 1:
            return (ReLU((output).tolist()[0]))
        return (ReLU(output))

    elif (activationFunc == "softmax"):
        if len(output) == 1:
            return (softmax(output.tolist()[0]))
        else:
            return (softmax(output))

def feedforward(ab , W1):
    if len(ab) %2 == 0:
        ab = np.concatenate([ab, [1]], 0)
    else:
        print("else feedforward:O")
        ab[len(ab)-1]=1
    if len((np.matmul(np.matrix(ab), W1))) == 1:
        return (np.matmul(np.matrix(ab), W1).tolist())[0]
    return np.matmul(np.matrix(ab), W1)

def feedforward_dropOut(ab , W1, dropOutPercent):
    if dropOutPercent > 0:
        ab = dropOut(ab, dropOutPercent)

    if len(ab) %2 == 0:
        ab = np.concatenate([ab, [1]], 0)
    else:
        print("else feedforward:O")
        ab[len(ab)-1]=1
    if len((np.matmul(np.matrix(ab), W1))) == 1:
        return (np.matmul(np.matrix(ab), W1).tolist())[0]
    return np.matmul(np.matrix(ab), W1)


'''
Error calculation Functions
'''
def softmax_error(y, y_hat, y_in, activationFunc):
    if activationFunc=="tanh":
        error = np.multiply(np.subtract(y, y_hat), tanh_deriv(y_in))
    elif activationFunc == "ReLU":
        error = np.multiply(np.subtract(y, y_hat), dReLU(y_in))
    elif activationFunc == "sig":
        error = np.multiply(np.subtract(y, y_hat), sigmaprime(y_in))
    elif activationFunc == "softmax":
        error = np.matmul(np.subtract(y, y_hat), softmaxprime_Jac(y, y_hat))
        #print error
    return error

def non_softmax_error(delta, W2, input , W1, activationFunc):
    # if len(input) %2 == 0:
    #     input = np.concatenate([input, [1]], 0)

    delta_in = np.matmul((delta), W2.T)
    #print delta_in
    if activationFunc=="tanh":
        if len(delta_in)%2 != 0:
            #print len(delta_in)
            delta_in = delta_in[:len(delta_in)-1]

        return np.multiply(delta_in, tanh_deriv(feedforward(input, W1)))
    elif activationFunc == "ReLU":
        dl = dReLU(feedforward(input, W1))

        #print len(delta_in)
        if len(delta_in)%2 != 0:
            #print len(delta_in)
            delta_in = delta_in[:len(delta_in)-1]
            #print len(delta_in)
    # delta_in = np.matmul((np.matrix(delta)), W2.T)
        return np.multiply(delta_in, dl)
    elif activationFunc == "sig":
        if len(delta_in) % 2 != 0:
            # print len(delta_in)
            delta_in = delta_in[:len(delta_in) - 1]
    # delta_in = np.matmul((np.matrix(delta)), W2.T)
        return np.multiply(delta_in, sigmaprime(feedforward(input, W1)))
'''
calculate delta
'''

def calculate_deltaW(error, input):
    if (len(input) % 2 == 0):
        input = np.concatenate([input, [1]], 0)
    delta_W = np.matmul(np.matrix(input).T,np.matrix(error))
    return np.asarray(delta_W)

def BpthroughTree_prev(EDUs, EDUs_Test, error_soft, W1, W2, dim, activationFunc):
    delta_W_All = np.zeros([2*dim, dim])
    eduKeys = sortEduKey(EDUs.keys(), reverse= True)
    parent_node = EDUs[str(eduKeys[0])]
    input = np.concatenate([EDUs[parent_node.leftChild].vector, EDUs[parent_node.rightChild].vector], 0)
    #input = np.concatenate([input, [1]], 0)
    delta = non_softmax_error(error_soft, W2, input, W1, activationFunc)
    EDUs[parent_node.leftChild].delta = delta
    EDUs[parent_node.rightChild].delta = delta
    deltaw = calculate_deltaW(delta, input)
    delta_W_All = np.add(deltaw, delta_W_All)


    for key in eduKeys:
        if EDUs[str(key)].isLeaf == False and EDUs[str(key)].isRoot == False:
            input = np.concatenate([EDUs[EDUs[str(key)].leftChild].vector, EDUs[EDUs[str(key)].rightChild].vector], 0)
            #input = np.concatenate([input, [1]], 0)
            W22= np.zeros([dim, dim])
            if EDUs[str(key)].child == "left":
                W22 = W1[0:dim, :]
            elif EDUs[str(key)].child == "right":
                W22 = W1[dim:2*dim, :]

            delta = non_softmax_error(EDUs[str(key)].delta, W22, input, W1, activationFunc)
            EDUs[EDUs[str(key)].leftChild].delta = delta
            EDUs[EDUs[str(key)].rightChild].delta = delta
            deltaw = calculate_deltaW(delta, input)
            delta_W_All = np.add(deltaw, delta_W_All)

    return delta_W_All

def BpthroughTree(EDUs, error_soft, W1, W2, dim, activationFunc, Flag):

    #print ("len" + str(len(EDUs)))
    #i=1
    delta_W_All = np.zeros([2*dim+1, dim])
    eduKeys = sortEduKey(EDUs.keys(), reverse= True)
    parent_node = EDUs[str(eduKeys[0])]

    leftVector = EDUs[parent_node.leftChild].vector
    rightVector = EDUs[parent_node.rightChild].vector

    input = np.concatenate([leftVector, rightVector], 0)
    #input = np.concatenate([input, [1]], 0)
    if (Flag):
        W2 = W2 [0:dim, :]
    else:
        W2 = W2 [dim:2*dim, :]

    delta = non_softmax_error(error_soft, W2, input, W1, activationFunc)
    #print (str(i) + " " +str(sum(delta)))
    EDUs[parent_node.leftChild].delta = delta
    EDUs[parent_node.rightChild].delta = delta

    delta_w = calculate_deltaW(delta, input)
    delta_W_All = np.add(delta_w, delta_W_All)


    for key in eduKeys:
        if EDUs[str(key)].isLeaf == False and EDUs[str(key)].isRoot == False:

            parent_node = EDUs[str(key)]

            leftVector = EDUs[parent_node.leftChild].vector
            rightVector = EDUs[parent_node.rightChild].vector

            input = np.concatenate([leftVector, rightVector], 0)
            # input = np.concatenate([EDUs[EDUs[str(key)].leftChild].vector, EDUs[EDUs[str(key)].rightChild].vector], 0)
            # W22  = np.zeros([dim, dim])

            if parent_node.child == "left":
                W22 = W1[0:dim, :]
            elif parent_node.child == "right":
                W22 = W1[dim:2 * dim, :]

            delta = non_softmax_error(parent_node.delta, W22, input, W1, activationFunc)
            # i+=1
            # print (str(i) + " " + str(sum(delta)))
            EDUs[parent_node.leftChild].delta = delta
            EDUs[parent_node.rightChild].delta = delta

            delta_w = calculate_deltaW(delta, input)

            delta_W_All = np.add(delta_w, delta_W_All)


            # leftVector = EDUs[parent_node.leftChild].vector
            # rightVector = EDUs[parent_node.rightChild].vector
            #
            # input = np.concatenate([leftVector, rightVector], 0)
            # #input = np.concatenate([EDUs[EDUs[str(key)].leftChild].vector, EDUs[EDUs[str(key)].rightChild].vector], 0)
            # #W22  = np.zeros([dim, dim])
            #
            # if EDUs[str(key)].child == "left":
            #     W22 = W1[0:dim, :]
            # elif EDUs[str(key)].child == "right":
            #     W22 = W1[dim:2 * dim, :]
            #
            # delta = non_softmax_error(EDUs[str(key)].delta, W22, input, W1, activationFunc)
            # #i+=1
            # #print (str(i) + " " + str(sum(delta)))
            # EDUs[EDUs[str(key)].leftChild].delta = delta
            # EDUs[EDUs[str(key)].rightChild].delta = delta
            # deltaw = calculate_deltaW(delta, input)
            # delta_W_All = np.add(deltaw, delta_W_All)
            #
            # #parent_node = EDUs[str(key)]
            # #input = np.concatenate([EDUs[EDUs[str(key)].leftChild].vector, EDUs[EDUs[str(key)].rightChild].vector], 0)
            # #input = np.concatenate([input, [1]], 0)
            # #delta = parent_node.delta

    return delta_W_All

'''
Update NeuralNet Weight
'''

def update_weight(eta, W, delta_W):
    delta_W = np.multiply(eta, delta_W)
    return np.add(W, delta_W)

'''
MSE: Mean Squared Error
'''
# def MSE(target, output):
#     mse = sqrt (np.sum((np.subtract(target, output)**2)))
#     return mse
def MSE(target, output):

    # diff = np.subtract(output, target)
    # diff_sqr = diff ** 2
    # mean_diff_sqr = diff_sqr.mean()
    # mse = np.sqrt(mean_diff_sqr)

    mse = mean_squared_error(target, output)
    return mse


'''
DropOut:
'''
'''
def dropOut(W, percent):
    # index = np.random.choice([0,1], size=len(input) , p= [(1-percent), percent])
    # input = np.multiply(input, index)
    # return input
    index = np.random.choice([0,1], size = W.shape , p = [(1-percent), percent])
    #input = np.multiply(input, index)
    #W = np.multiply(W, index)
    return index
'''

def dropOut(length, percent):
    index = np.random.choice(range(0,length-1), size = int(np.floor(length*percent)))
    #print (index)
    return index

def dropcolrow(W, index, row):
    #return W
    if (row):
        #x = np.zeros(len(W[0]))
        W[index, :] = 0
        return W
    else:
        #x = np.zeros(len(W))
        W[:, index] = 0
        return W



