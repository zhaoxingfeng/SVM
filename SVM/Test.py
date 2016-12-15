# coding:utf-8
"""
作者：zhaoxingfeng	日期：2016.12.15
功能：两个测试样本集对SVM进行性能测试
    1、UCI公共库Wine数据集（1，2），二分类
    2、手写字数据库digits（0，1，2），多分类
"""
from __future__ import division
import numpy as np
from SVM import *
import sys
if sys.version_info[0] >= 3:
    xrange = range

# UCI数据集wine
def wine():
    def loadDataSet(fileName):
        dataMat, labelMat = [], []
        with open(fileName) as fr:
            for line in fr.readlines():
                lineArr = line.strip().split(',')
                dataMat.append([float(data) for data in lineArr[:-1]])
                if int(lineArr[-1]) == 1:
                    labelMat.append(1)
                else:
                    labelMat.append(-1)
        return np.mat(dataMat), np.mat(labelMat).T
    print("Step1: Loading data......")
    filename = r'wine.txt'
    dataMat, labelMat = loadDataSet(filename)
    train_x1 = dataMat[0:41]
    train_x2 = dataMat[60:110]
    train_x = np.vstack((train_x1, train_x2))
    train_y1 = labelMat[0:41]
    train_y2 = labelMat[60:110]
    train_y = np.vstack((train_y1, train_y2))

    test_x1 = dataMat[41:59]
    test_x2 = dataMat[110:]
    test_x = np.vstack((test_x1, test_x2))
    test_y1 = labelMat[41:59]
    test_y2 = labelMat[110:]
    test_y = np.vstack((test_y1, test_y2))

    print("Step2: Training SVM......")
    maxIter = 50
    C = 10.6
    toler = 0.001
    kernelOption = ("rbf", 20)
    svmClassifier = trainSVM(train_x, train_y, C, toler, maxIter, kernelOption)

    print("Step3: Testing classifier......")
    accuracy, labelpredict, num = testSVM(svmClassifier, test_x, test_y)
    print("\tAccuracy = %.2f%%" % (accuracy * 100))
wine()

# 手写字digits, 0/1/2
def handwriting():
    def vector(filename):
        vector1024 = np.zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            linestr = fr.readline()
            for j in range(32):
                vector1024[0, 32 * i + j] = int(linestr[j])
        return vector1024

    def loadfiles(dirName):
        from os import listdir
        fileList = listdir(dirName)
        labelMat = []
        dataMat = np.zeros((len(fileList), 1024))
        for i in range(len(fileList)):
            num = int(fileList[i].split('_')[0])
            labelMat.append(num)
            dataMat[i, :] = vector('%s\%s' % (dirName, fileList[i]))
        return np.mat(dataMat), np.mat(labelMat).T
    print("Step1: Loading data......")
    traindataMat, trainlabelMat = loadfiles(r"trainingDigits")
    testdataMat, testlabelMat = loadfiles(r"testDigits")

    print("Step2: Training SVM......")
    # 第一个分类器：0 -> 12
    # 训练样本
    trainlabel012 = []
    for i in range(len(trainlabelMat)):
        if trainlabelMat[i] == 0:
            trainlabel012.append(-1)
        else:
            trainlabel012.append(1)
    trainlabel012Mat = np.mat(trainlabel012).T
    traindata012Mat = traindataMat[::]
    # 测试样本
    testlabel012 = []
    for i in range(len(testlabelMat)):
        if testlabelMat[i] == 0:
            testlabel012.append(-1)
        else:
            testlabel012.append(1)
    testlabel012Mat = np.mat(testlabel012).T
    testdata012Mat = testdataMat[::]

    maxIter = 100
    C = 200
    toler = 0.0001
    kernelOption = ("rbf", 10)
    svmClassifier012 = trainSVM(traindata012Mat, trainlabel012Mat, C, toler, maxIter, kernelOption)
    accuracy012, labelpredict012, numright012 = testSVM(svmClassifier012, testdata012Mat, testlabel012Mat)
    print("\tAccuracy012 = %.2f%%" % (accuracy012 * 100))

    # 第二个分类器：1 -> 2
    # 训练样本
    trainlabel12 = []
    for i in range(trainlabel012.count(-1),len(trainlabelMat)):
        if trainlabelMat[i] == 1:
            trainlabel12.append(-1)
        else:
            trainlabel12.append(1)
    trainlabel12Mat = np.mat(trainlabel12).T
    traindata12Mat = traindataMat[trainlabel012.count(-1):]
    # 测试样本，被前一个分类器分为 1 且不包含本来是 -1 结果被分类为 1 的样本
    testlabel12index = []
    for i in range(len(labelpredict012)):
        if labelpredict012[i] == 1 and testlabel012Mat[i] == 1:
            testlabel12index.append(i)

    num12 = len(testlabel12index)
    testlabel12raw = testlabelMat[testlabel12index]
    testlabel12 = [0 for i in range(num12)]
    for i in range(num12):
        if testlabel12raw[i] == 1:
            testlabel12[i] = -1
        else:
            testlabel12[i] = 1
    testlabel12Mat = np.mat(testlabel12).T
    testdata12Mat = testdataMat[testlabel12index]

    svmClassifier12 = trainSVM(traindata12Mat, trainlabel12Mat, C, toler, maxIter, kernelOption)
    accuracy12, labelpredict12, numright12 = testSVM(svmClassifier12, testdata12Mat, testlabel12Mat)
    print("\tAccuracy12 = %.2f%%" % (accuracy12 * 100))

    total_accuracy = (numright012 + num12 * accuracy12) / len(testlabelMat)
    print("Final accuracy = %.3f%%" % (total_accuracy * 100))
handwriting()