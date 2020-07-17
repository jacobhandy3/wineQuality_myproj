from NNcode import NNanalysis
from dataHandle import datasetInfo, formatData
from decision_tree import treeAnalysis
from pydot import graph_from_dot_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path, header, indexCol, Xmax, labelCol, classNum = datasetInfo()
dataset = formatData(path=path, head=header, indexCol=indexCol)
""" NNanalysis(path=path, dataset=dataset, Xmax=Xmax,
           labelCol=labelCol, classNum=classNum)
 """
depth = list(range(list(range(1,50,1))))    #50
impurity = list(range(0,1,0.1))             #11
maxLeaf = 2                                 #not config
feat = list(None,"sqrt","log2")            #3
split = 2                                 #not config
minLeaf = 2                                 #not config
crit = list("gini", "entropy")              #2
# List to store the average RMSE for each value of max_depth:
accuracies = []
for c in crit:  #2
    for f in feat: #3
        for d in depth: #50
            for i in impurity:  #11
                score = treeAnalysis(data=dataset, Xmax=Xmax, labelCol=labelCol,
                        maxDepth=d,minImpurity=i,maxLeaf=maxLeaf,maxFeat=f,
                        minSplit=split,minLeaf=minLeaf,criterion=c)
                accuracies.append(score)

#summarize history for accuracy
plt.plot(accuracies)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('run iteration')
plt.show()
accuracies.sort()
print("The highest accuracy achieved: " + str(accuracies[0]))