from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image


def treeAnalysis(data, Xmax, labelCol,maxDepth,minImpurity,maxLeaf,maxFeat,minSplit,minLeaf,criterion):
    # split into input (X) and output (y) variables
    print("Separating the data from the labels")
    X = data.iloc[:, 0:Xmax]
    y = data.iloc[:, labelCol]
    # one hot encode the label column
    y = pd.get_dummies(y)
    # split data with 0.33 test size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    print("Now onto the ML code")
    # Create the decision tree model
    dt = DecisionTreeClassifier(max_depth=maxDepth,
                                random_state=0,
                                min_impurity_decrease=minImpurity,
                                max_leaf_nodes=maxLeaf,
                                max_features=maxFeat,
                                min_samples_split=minSplit,
                                min_samples_leaf=minLeaf,
                                criterion=criterion)
    # fit the model with train sets
    dt.fit(X_train, y_train)
    score = dt.score(X_test, y_test)
    return(score)

    # summarize history for accuracy
    # plt.plot(accuracies)
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('max_depth')
    # plt.show()

    """ ratings = np.array(y_test).argmax(axis=1)
    predictions = np.array(y_pred).argmax(axis=1)
    confusion_matrix(ratings, predictions) """
