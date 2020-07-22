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
depth = list(range(1,51,1))                 #50
impurity = np.arange(0.0,1.1,0.1)             #11
maxLeaf = [None,2]                                #2
feat = [None,"sqrt","log2"]            #3
split = np.arange(0.1,0.6,0.1)                                 #5
minLeaf = np.arange(0.05,0.55,0.05)                                 #10
crit = ["gini", "entropy"]              #2
for c in crit:  #2
    for f in feat: #3
        for maL in maxLeaf: #2
            for d in depth: #50
                for i in impurity:  #11
                    for s in split: #5
                        for miL in minLeaf: #10
                            score = treeAnalysis(data=dataset, Xmax=Xmax, labelCol=labelCol,
                                    maxDepth=d,minImpurity=i,maxLeaf=maL,maxFeat=f,
                                    minSplit=s,minLeaf=miL,criterion=c)
                            if score > high_score["score"]:
                                high_score["score"] = score
                                high_score["depth"] = d
                                high_score["impurity"] = i
                                high_score["feat"] = f
                                high_score["crit"] = c
                                high_score["maxLeaf"] = maL
                                high_score["split"] = s
                                high_score["minLeaf"] = miL

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