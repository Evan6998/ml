import numpy
import operator

def classify0(inX, dataSet, labels, k):
    dataSetNum = dataSet.shape[0]
    diffSet = numpy.tile(inX, (dataSetNum, 1)) - dataSet
    squareSet = diffSet ** 2
    sumSet = squareSet.sum(axis = 1)
    distances = sumSet ** 0.5
    sortedIndicesDistance = distances.argsort()
    labelCount = {}
    for i in range(k):
        labelName = labels[sortedIndicesDistance[i]]
        labelCount[labelName] = labelCount.get(labelName, 0) + 1
    sortedLabelCount = sorted(labelCount.iteritems(), 
    key=operator.itemgetter(1), reverse=True)
    return sortedLabelCount[0][0]

def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    file = open(filename)
    arrayOfLines = file.readlines()
    numberOfLines = len(arrayOfLines)
    retMat = numpy.zeros((numberOfLines, 3))
    classLabel = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        lineList = line.split('\t')
        retMat[index,:] = lineList[0:3]
        if(lineList[-1].isdigit()):
            classLabel.append(int(lineList[-1]))
        else:
            classLabel.append(love_dictionary.get(lineList[-1]))
        index += 1
    return retMat, classLabel
