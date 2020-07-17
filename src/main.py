from NNcode import NNanalysis
from dataHandle import datasetInfo, formatData
from decision_tree import treeAnalysis
from pydot import graph_from_dot_data
import numpy as np
import matplotlib.pyplot as plt
import decimal

path, header, indexCol, Xmax, labelCol, classNum = datasetInfo()
dataset = formatData(path=path, head=header, indexCol=indexCol)
""" NNanalysis(path=path, dataset=dataset, Xmax=Xmax,
           labelCol=labelCol, classNum=classNum)
 """
high_score = {"score":0,"depth":1,"impurity":0,"maxLeaf":2,"feat":None,"split":2,"minLeaf":2,"crit":"gini"}
depth = list(range(1,50,1))                 #50
impurity = np.arange(0.0,1.1,0.1)             #11
maxLeaf = None                                 #not config
feat = [None,"sqrt","log2"]            #3
split = 2                                 #not config
minLeaf = 1                                 #not config
crit = ["gini", "entropy"]              #2
for c in crit:  #2
    for f in feat: #3
        for d in depth: #50
            for i in impurity:  #11
                score = treeAnalysis(data=dataset, Xmax=Xmax, labelCol=labelCol,
                        maxDepth=d,minImpurity=i,maxLeaf=maxLeaf,maxFeat=f,
                        minSplit=split,minLeaf=minLeaf,criterion=c)
                if score > high_score["score"]:
                    high_score["score"] = score
                    high_score["depth"] = d
                    high_score["impurity"] = i
                    high_score["feat"] = f
                    high_score["crit"] = c

for keys,values in high_score.items():
    print(keys)
    print(values)
#summarize history for accuracy
# plt.plot(accuracies)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('run iteration')
# plt.show()
# accuracies.sort()
# print("The highest accuracy achieved: " + str(accuracies[0]))