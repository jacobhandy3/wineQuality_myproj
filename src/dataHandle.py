# Import Libraries
import pandas as pd
import numpy as np
import glob as glob
import os


def loadDataset(path, header, indexCol, colL, labelCol, dropFeats=[], missReplacement=[], missCols=[]):
    print("Pre-processing data...")
    # get formatted pandas dataset
    dataset = formatData(path, header, indexCol)

    # Drop columns not using
    if(len(dropFeats) != 0):
        dataset.drop(dropFeats, axis=1, inplace=True)

    # replace the Infinity values with NaNs
    dataset = dataset.replace(to_replace=np.inf, value=np.nan)
    # drop any row with a NaN
    dataset = dataset.dropna(how="any")

    print("done")
    return dataset


def datasetInfo():
    return r"C:\Users\jakem\Documents\GitHub\2020\wineQuality_myproj\data", 0, None, 10, 11, 11

# takes a folder path to find csv files, 0 for a header and None for no header
# and 0 for an index column or None for no index column


def formatData(path, head, indexCol):
    # open dataset folder path
    os.chdir(path)
    # find files with glob
    fileList = glob.glob("*.csv")
    # create a temp list
    dataList = []
    # loop thorugh the files
    for file in fileList:
        # read each file as csv with pandas
        data = pd.read_csv(file, sep=';', header=head, index_col=indexCol)
        # append to temp list
        dataList.append(data)
    # concat vertically
    dataset = pd.concat(dataList, axis=0)
    # return the dataset
    return dataset
