from NNcode import NNanalysis
from dataHandle import datasetInfo, formatData
from decision_tree import treeAnalysis
# from pydot import graph_from_dot_data
import numpy as np
import matplotlib.pyplot as plt
import decimal

path, header, indexCol, labelCol, classNum = datasetInfo()
dataset = formatData(path=path, head=header, indexCol=indexCol)
""" NNanalysis(path=path, dataset=dataset, Xmax=Xmax,
           labelCol=labelCol, classNum=classNum)
 """
high_score = {"score":0,"depth":0,"impurity":0,"maxLeaf":0,"feat":None,"minSplit":0,"minLeaf":0,"crit":"gini"}
depth = list(range(1,51,1))                 #50
impurity = 0.0            #1
maxLeaf = [None,2]                                #2
feat = [None,"sqrt","log2"]            #3
minSplit = list(range(2,26,1))                                 #24
minLeaf = list(range(1,51,1))                              #50
crit = ["gini", "entropy"]              #2
"""count = 0
for c in crit:  #2
    for f in feat: #3
        for d in depth: #50
            for ms in minSplit: #24
                for miL in minLeaf: #100
                    count+=1
                    print("iteration #: ", count)
                    score = treeAnalysis(data=dataset, labelCol=labelCol,
                            maxDepth=d,minImpurity=0.0,maxLeaf=None,maxFeat=f,
                            minSplit=ms,minLeaf=miL,criterion=c)
                    print("parameters: ",c," / ",f," / depth:",d," / minSplit",ms," / minLeaf:",miL," / maxLeaf",None," / minImpurity",0.0)
                    if score > high_score["score"]:
                        high_score["score"] = score
                        high_score["depth"] = d
                        high_score["impurity"] = 0.0
                        high_score["feat"] = f
                        high_score["crit"] = c
                        high_score["maxLeaf"] = None
                        high_score["minSplit"] = ms
                        high_score["minLeaf"] = miL"""

print(treeAnalysis(data=dataset, labelCol=labelCol,
                                maxDepth=20,minImpurity=0.0,maxLeaf=None,maxFeat="sqrt",
                                minSplit=2,minLeaf=1,criterion="gini"))

"""for keys,values in high_score.items():
    print(keys)
    print(values)"""
#summarize history for accuracy
# plt.plot(accuracies)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('run iteration')
# plt.show()
# accuracies.sort()
# print("The highest accuracy achieved: " + str(accuracies[0]))