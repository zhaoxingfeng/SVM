# coding:utf-8
"""
作者：zhaoxingfeng	日期：2016.12.15
功能：支持向量机，support vector machine(SVM)
版本：2.0
参考文献：
[1] 支持向量机训练算法实现及其改进[D].南京理工大学,2005,26-33.
[2] v_JULY_v.支持向量机通俗导论（理解SVM的三层境界）[DB/OL].http://blog.csdn.net/v_july_v/article/details/7624837,2012-06-01.
[3] zouxy09.机器学习算法与Python实践之（四）支持向量机（SVM）实现[DB/OL].http://blog.csdn.net/zouxy09/article/details/17292011,2013-12-13.
[4] JerryLead.支持向量机（五）SMO算法[DB/OL].http://www.cnblogs.com/jerrylead/archive/2011/03/18/1988419.html,2013-12-13.
[5] 黄啸. 支持向量机核函数的研究［D].苏州大学,2008.
"""
from __future__ import division
import numpy as np
import random
import copy
import math
import time
import sys

if sys.version_info[0] >= 3:
    xrange = range

def calcuKernelValue(train_x, sample_x, kernelOpt = ("linear", 0)):
    kernelType = kernelOpt[0]
    kernelPara = kernelOpt[1]
    numSamples = np.shape(train_x)[0]
    kernelValue = np.mat(np.zeros((numSamples, 1)))
    if kernelType == "linear":
        kernelValue = train_x * sample_x.T
    elif kernelOpt[0] == "rbf":
        sigma = kernelPara
        for i in xrange(numSamples):
            diff = train_x[i, :] - sample_x
            kernelValue[i] = math.exp(diff * diff.T / (-2 * sigma ** 2))
    else:
        print("The kernel is not supported")
    return kernelValue

# 核函数求内积
def calcKernelMat(train_x, kernelOpt):
    numSamples = np.shape(train_x)[0]
    kernealMat = np.mat(np.zeros((numSamples, numSamples)))
    for i in xrange(numSamples):
        kernealMat[:, i] = calcuKernelValue(train_x, train_x[i], kernelOpt)
    return kernealMat

# SVM参数
class svmSruct(object):
    def __init__(self, trainX, trainY, c, tolerance, maxIteration, kernelOption):
        self.train_x = trainX
        self.train_y = trainY
        self.C = c
        self.toler = tolerance
        self.maxIter = maxIteration
        self.numSamples = np.shape(trainX)[0]
        self.alphas = np.mat(np.zeros((self.numSamples, 1)))
        self.b = 0
        self.errorCache = np.mat(np.zeros((self.numSamples, 2)))
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMat(self.train_x, self.kernelOpt)

def calcError(svm, alpha_i):
    func_i = np.multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_i] + svm.b
    erro_i = func_i - svm.train_y[alpha_i]
    return erro_i

def updateError(svm, alpha_j):
    error = calcError(svm, alpha_j)
    svm.errorCache[alpha_j] = [1, error]

# 选取一对 alpha_i 和 alpha_j，使用启发式方法
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]
    alpha_index = np.nonzero(svm.errorCache[:, 0])[0]
    maxstep = float("-inf")
    alpha_j, error_j = 0, 0
    if len(alpha_index) > 1:
        # 遍历选择最大化 |error_i - error_j| 的 alpha_j
        for alpha_k in alpha_index:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_i - error_k) > maxstep:
                maxstep = abs(error_i - error_k)
                alpha_j = alpha_k
                error_j = error_k
    else:
        # 最后一个样本，与之配对的 alpha_j采用随机选择
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = random.randint(0, svm.numSamples - 1)
        error_j = calcError(svm, alpha_j)
    return alpha_j, error_j

# 内循环
def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)
    error_i_ago = copy.deepcopy(error_i)
    if (svm.train_y[alpha_i] * error_i < -svm.toler and svm.alphas[alpha_i] < svm.C) or \
        (svm.train_y[alpha_i] * error_i > svm.toler and svm.alphas[alpha_i] > 0):
        # 选择aplha_j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_ago = copy.deepcopy(svm.alphas[alpha_i])
        alpha_j_ago = copy.deepcopy(svm.alphas[alpha_j])
        error_j_ago = copy.deepcopy(error_j)
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] - \
                svm.kernelMat[alpha_j, alpha_j]

        # 更新aplha_j, alpha_i
        svm.alphas[alpha_j] = alpha_j_ago - svm.train_y[alpha_j] * (error_i - error_j) / eta
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        elif svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L
        svm.alphas[alpha_i] = alpha_i_ago + svm.train_y[alpha_i] * svm.train_y[alpha_j] * \
                                            (alpha_j_ago - svm.alphas[alpha_j])
        # 问题：为什么只判断alpha_j?
        if abs(alpha_j_ago - svm.alphas[alpha_j]) < 10 ** (-5):
            return 0

        # 更新 b
        b1 = svm.b - error_i_ago - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_ago) * \
            svm.kernelMat[alpha_i, alpha_i] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_ago) * \
            svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j_ago - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_ago) * \
            svm.kernelMat[alpha_i, alpha_j] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_ago) * \
            svm.kernelMat[alpha_j, alpha_j]
        if (svm.alphas[alpha_i] > 0) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (svm.alphas[alpha_j] > 0) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2

        # 更新 b 之后再更新误差
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0

# 训练SVM
def trainSVM(train_x, train_y, c, toler, maxIter, kernelOpt):
    train_start = time.time()
    svm = svmSruct(train_x, train_y, c, toler, maxIter, kernelOpt)
    entire = True
    alphaPairsChanged = 0
    iter = 0
    while (iter < svm.maxIter) and ((alphaPairsChanged > 0) or entire):
        alphaPairsChanged = 0
        if entire:
            for i in xrange(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i)
            print("\tIter = %d, entire set, alpha2 changed = %d" % (iter, alphaPairsChanged))
            iter += 1
        else:
            nonBound_index = np.nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBound_index:
                alphaPairsChanged += innerLoop(svm, i)
            print("\tIter = %d, non boundary, alpha2 changed = %d" % (iter, alphaPairsChanged))
            iter += 1
        if entire:
            entire = False
        elif alphaPairsChanged == 0:
            entire = True
    train_end = time.time()
    print("\tnumVector VS numSamples == %d -- %d" % (len(np.nonzero(svm.alphas.A > 0)[0]), svm.numSamples))
    print("\tTraining complete! ---------------- %.3fs" % (train_end - train_start))
    return svm

# 测试样本
def testSVM(svm, test_x, test_y):
    numTest = np.shape(test_x)[0]
    supportVect_index = np.nonzero(svm.alphas.A > 0)[0]
    supportVect = svm.train_x[supportVect_index]
    supportLabels = svm.train_y[supportVect_index]
    supportAlphas = svm.alphas[supportVect_index]
    num = 0
    numright = 0
    labelpredict = []
    for i in xrange(numTest):
        kernelValue = calcuKernelValue(supportVect, test_x[i, :], svm.kernelOpt)
        predict = kernelValue.T * np.multiply(supportLabels, supportAlphas) + svm.b
        labelpredict.append(int(np.sign(predict)))
        if np.sign(predict) == np.sign(test_y[i]):
            num += 1
            if np.sign(test_y[i]) == -1:
               numright += 1
    print("\tnumRight VS numTest == %d -- %d" % (num, numTest))
    accuracy = num / numTest
    return accuracy, labelpredict, numright

