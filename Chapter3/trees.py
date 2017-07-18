from math import log

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    ent = 0
    m = len(dataSet)
    labelCount = {}
    for dataVec in dataSet:
        label = dataVec[-1]
        labelCount[label] = labelCount.get(label, 0) + 1
    for label in labelCount:
        probilaty = float(labelCount[label]) / m
        ent -= probilaty*log(probilaty, 2)
    return ent

def splitDataSet(dataSet, axis, value):
    retMat = []
    for dataVec in dataSet:
        if dataVec[axis] == value:
            retMat.append(dataVec[:axis] + (dataVec[axis+1:]))
    return retMat

def chooseBestFeatureToSplit(dataSet):
    featureNum = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1
    for i in range(featureNum):
        featureList = [dataVec[i] for dataVec in dataSet]
        uniqueVals = set(featureList)
        entropy = 0
        for val in uniqueVals:
            subMat = splitDataSet(dataSet, i, val)
            prob = len(subMat)/float(len(dataSet))
            entropy += prob * calcShannonEnt(subMat)
        # infoGain代表熵降， 熵降越多，分类越好
        infoGain = baseEntropy - entropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature  