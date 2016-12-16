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
from sklearn.cross_validation import train_test_split
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
    train_x, test_x, train_y, test_y = train_test_split(dataMat, labelMat, test_size = 0.2)

    print("Step2: Training SVM......")
    C = 20
    toler = 0.001
    maxIter = 50
    kernelOption = ("rbf", 20)
    svmClassifier = trainSVM(train_x, train_y, C, toler, maxIter, kernelOption)

    print("Step3: Testing classifier......")
    accuracy, labelpredict, num = testSVM(svmClassifier, test_x, test_y)
    print("\tAccuracy = %.2f%%" % (accuracy * 100))
wine()

print("------------------------------------------------------------------")

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
    C = 200
    toler = 0.0001
    maxIter = 100
    kernelOption = ("rbf", 10)
    # 第一个分类器：0 / 12
    trainlabel012 = []
    for i in range(len(trainlabelMat)):
        if trainlabelMat[i] == 0:
            trainlabel012.append(-1)
        else:
            trainlabel012.append(1)
    trainlabel012Mat = np.mat(trainlabel012).T
    traindata012Mat = traindataMat[::]
    svmClassifier012 = trainSVM(traindata012Mat, trainlabel012Mat, C, toler, maxIter, kernelOption)
    # 第二个分类器：1 / 2
    trainlabel12 = []
    for i in range(trainlabel012.count(-1),len(trainlabelMat)):
        if trainlabelMat[i] == 1:
            trainlabel12.append(-1)
        else:
            trainlabel12.append(1)
    trainlabel12Mat = np.mat(trainlabel12).T
    traindata12Mat = traindataMat[trainlabel012.count(-1):]
    svmClassifier12 = trainSVM(traindata12Mat, trainlabel12Mat, C, toler, maxIter, kernelOption)

    print("Step3: Testing classifier......")
    # 测试样本 0 / 12
    testlabel012 = []
    for i in range(len(testlabelMat)):
        if testlabelMat[i] == 0:
            testlabel012.append(-1)
        else:
            testlabel012.append(1)
    testlabel012Mat = np.mat(testlabel012).T
    testdata012Mat = testdataMat[::]
    accuracy012, labelpredict012, numright012 = testSVM(svmClassifier012, testdata012Mat, testlabel012Mat)
    # 测试样本 1 / 2
    # 被前一个分类器分为 1 且不包含本来是 -1 结果被分类为 1 的样本
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
    accuracy12, labelpredict12, numright12 = testSVM(svmClassifier12, testdata12Mat, testlabel12Mat)
    # 统计被正确分类的样本数量，包括：
    # 1、第一轮分类 -> 本来是-1结果被分为-1的个数。
    #    第一轮分类结束后，本来是1结果被分类为1的样本进入第二轮分类；本来是-1结果本分类为1的样本已经分类错误，不参与第二轮分类
    # 2、第二轮分类 -> -1、1被正确分类的个数。
    total_accuracy = (numright012 + num12 * accuracy12) / len(testlabelMat)
    print("\tFinal accuracy = %.3f%%" % (total_accuracy * 100))
handwriting()
