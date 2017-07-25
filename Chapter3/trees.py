#coding=utf-8
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
        #infoGain代表熵降， 熵降越多，分类越好
        infoGain = baseEntropy - entropy
        if infoGain > bestInfoGain:
            bestFeature = i
            bestInfoGain = infoGain
    return bestFeature  

def majorCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    return max(classCount.iterkeys(), key=lambda vote:classCount[vote])

def createTree(dataSet, labels):
    """
    :type dataSet: np.array  
    :type labels: List  
    :rtype: labelName or tree
    """
    classList = [dataVec[-1] for dataVec in dataSet]
    # 类别完全相同，停止划分，返回类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征，返回类别出现次数最多的
    if len(dataSet[0]) == 1:
        return majorCnt(classList)

    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])

    featureVals = [dataVec[bestFeature] for dataVec in dataSet]
    uniqueVal = set(featureVals)
    
    for val in uniqueVal:
        # 传递labels的拷贝给递归函数
        subLabels = labels[:]
        myTree[bestFeatureLabel][val] = createTree(splitDataSet(dataSet, bestFeature, val), subLabels)
    return myTree

# 上述构造决策树 下面使用决策树进行分类
def classify(tree, labels, dataVec):
    featName = tree.keys()[0]
    treeChildren = tree[featName]
    featIndex = labels.index(featName)
    for key in treeChildren:
        if key == dataVec[featIndex]:
            classLabel = classify(treeChildren[key], labels, dataVec) if type(treeChildren[key]).__name__ == 'dict' else treeChildren[key]
    return classLabel

