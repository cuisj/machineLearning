# coding:utf-8

from numpy import *

def loadDataSet(fileName, delim='\t'):
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr = [map(float, arr) for arr in stringArr]

	return mat(datArr)


def pca(dataMat, topNfeat=9999999):
	meanVals = mean(dataMat, axis=0)	# 向量均值(期望)
	meanRemoved = dataMat - meanVals	# 差

	covMat = cov(meanRemoved, rowvar=0)	# 样本协方差

	eigVals, eigVects = linalg.eig(mat(covMat))	# 特征值，特征向量
	eigValInd = argsort(eigVals)

	# 从小到大对N值排序
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVects = eigVects[:,eigValInd]

	# 把数据转换到新空间
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals

	return lowDDataMat, reconMat





if __name__ == '__main__':
	dataMat = loadDataSet('testSet.txt')

	lowDMat, reconMat = pca(dataMat, 1)

	import matplotlib
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
	ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
	plt.show()

