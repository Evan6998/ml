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

def autoNorm(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    range = maxValues - minValues
    normalDataSet = numpy.zeros(numpy.shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - numpy.tile(minValues, (m, 1))
    normalDataSet /= numpy.tile(range, (m, 1))
    return normalDataSet, range, minValues

def datingClassTest():
    hoRatio = 0.050
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normDataSet, ranges, minValues = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    testCases = int(m * hoRatio)
    errorCount = 0
    for i in range(testCases):
        classifierResult = classify0(normDataSet[i], normDataSet[testCases:], datingLabels[testCases:], 3)
        if classifierResult != datingLabels[i]: 
            errorCount += 1
            print "\033[93mthe classifier came out with %d, real answer is %d\033[0m" % (classifierResult, datingLabels[i])
        else:
            print "the classifier came out with %d, real answer is %d" % (classifierResult, datingLabels[i])            
    print "the total error rate is %f" % (errorCount / float(testCases))

        
    