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
