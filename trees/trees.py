#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
import operator

def createDataSet():
	dataSet = [[1, 1, 'yes'],		# 是否是鱼类
		[1, 1, 'yes'],
		[1, 0, 'no'],
		[0, 1, 'no'],
		[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']	#属性名称

	return dataSet, labels


#
# 信息：		l(xi) = -log2 p(xi)
# 熵:		H = -∑ p(xi) log2 p(xi)
#
# 熵越大，越无序。熵减少，信息增加
#

# 计算数据集熵值
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}

	# 为所有可能的分类创建字典,并计算各个分类的记录次数
	for featVec in dataSet:
		currentLabel = featVec[-1]		#最后一项为yes, no类别
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1

	shannonEnt = 0.0

	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries	# p(xi)
		shannonEnt -= prob * log(prob, 2)		# H

	return shannonEnt


# 根据属性值划分子集
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

# 找出使熵减最多(信息增益最大)的属性
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1	# 属性数目
	baseEntropy = calcShannonEnt(dataSet)	# 初始熵值

	bestInfoGain = 0.0
	baseFeature = -1

	for i in range(numFeatures):	# 循环所有属性
		featList = [example[i] for example in dataSet]	# i属性所有可能的值
		uniqueVals =set(featList)
		newEntropy = 0.0

		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet) # 条件熵

		infoGain = baseEntropy - newEntropy	# 信息增益

		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i

	return bestFeature

# 找出计数最多的类别
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1

	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

	return sortedClassCount[0][0]


# 终止条件：
#	1.分支下的实例类别相同
#	2.已无属用来划分数据集, 如果类别仍然不唯一，则用多数表决确定该叶节点的类别
#

def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]

	# 类别完全相同，停止继续划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]

	# 属性用完，多数表达确定类别
	if len(dataSet[0]) == 1: # 数据集中已无属性，只有类别标签了
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]

	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])

	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)

	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

	return myTree


def storeTree(inputTree, filename):
	import pickle
	with open(filename, 'w') as f:
		pickle.dump(inputTree, f)

def grubTree(filename):
	import pickle
	with open(filename) as f:
		return pickle.load(f)




# 使用决策树分类
def classify(inputTree, featLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]

	featIndex = featLabels.index(firstStr) # 从根属性找起
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict': # 不能确定，接着往下找
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel


if __name__ == '__main__':
	dataSet, labels = createDataSet()

	#print splitDataSet(myDat, 0, 0）
	#print splitDataSet(myDat, 0, 1)
	#print chooseBestFeatureToSplit(myDat)

	#myTree = createTree(dataSet, labels[:])
	#storeTree(myTree, 'classiferStorage.txt')
	myTree = grubTree('classiferStorage.txt')

	td = [0, 0]
	print classify(myTree,labels, td)


