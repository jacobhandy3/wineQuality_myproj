from NNcode import NNanalysis
from dataHandle import datasetInfo, formatData
from decision_tree import treeAnalysis

path, header, indexCol, Xmax, labelCol, classNum = datasetInfo()
dataset = formatData(path=path, head=header, indexCol=indexCol)
""" NNanalysis(path=path, dataset=dataset, Xmax=Xmax,
           labelCol=labelCol, classNum=classNum)
 """
treeAnalysis(data=dataset, Xmax=Xmax, labelCol=labelCol)
