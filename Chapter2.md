# 机器学习
## 第二章 K-邻近算法
### 2.1 算法概述
简单的说，k-邻近算法采用测量不同特征值之间的距离方法分类 

k-邻近算法
* 优点：精度高、对异常值不敏感、无数据输入假定
* 缺点：计算复杂度高、空间复杂度高
* 适用数据范围：数值型和标称型（即不同种类）

> 存在一个样本数据集合，样本中的每个数据都存在标签。输入没有标签的数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后提取出特征最相似的前k个数据的标签。

#### 2.1.1 
适用以下代码生成数据
```python
from numpy import *
#执行排序操作时用到operator
import operator

def createDataSet():
    group = array([1.0, 1.1], [1.0, 1.0], 
                  [0, 0], [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
```
下述函数进行分类
```python
import numpy
import operator

def classify0(inX, dataSet, labels, k):
    dataSetNum = dataSet.shape[0]
    diffSet = numpy.tile(inX, (dataSetNum, 1)) - dataSet
    squareSet = diffSet ** 2
    sumSet = squareSet.sum(axis = 1)
    
    #计算到每个点的距离
    distances = sumSet ** 0.5 
    
    #获得排序后的下标
    sortedIndicesDistance = distances.argsort() 
    labelCount = {}
    
    #前k个数据点
    for i in range(k): 
        labelName = labels[sortedIndicesDistance[i]]
        labelCount[labelName] = labelCount.get(labelName, 0) + 1
    sortedLabelCount = sorted(labelCount.iteritems(), 
    key=operator.itemgetter(1), reverse=True)
    return sortedLabelCount[0][0]
```

