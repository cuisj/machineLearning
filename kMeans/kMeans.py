# coding:utf-8

from numpy import *

def loadDataSet(fileName):
	dataMat = []
	with open(fileName) as fr:
		for line in fr.readlines():
			curLine = line.strip().split('\t')
			fltLine = map(float, curLine)
			dataMat.append(fltLine)

	return dataMat

# 欧基里德距离
def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k, n)))	# k个n维向量

	# 构建簇质心
	for j in range(n):				# 选择每一维的合适的值
		minJ = min(dataSet[:, j])
		rangeJ = float(max(dataSet[:, j]) - minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
		# print centroids

	return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))	# 存储每个点的归属簇: [0]簇索引，[1]误差即距离

	centroids = createCent(dataSet, k)

	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = inf
			minIndex = -1
			for j in range(k):	# 寻找最近质心
				distJI = distMeas(centroids[j,:], dataSet[i,:])
				if distJI < minDist:
					minDist = distJI
					minIndex = j

			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True

			clusterAssment[i,:] = minIndex, minDist**2

		for cent in range(k):	# 簇有新元素进入，更新质心
			pstlnClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]	#给定簇索引的所有点
			centroids[cent,:] = mean(pstlnClust, axis=0)	# 属性的均值

	return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2)))			# [0]所属簇心，[1]簇心距离(平方误差)

	centroid0 = mean(dataSet, axis=0).tolist()[0]	# 所有的点看成一个簇, 簇心
	centList = [centroid0]							# 所有的簇心列表

	for j in range(m):
		clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j,:]) ** 2	# 所有点到簇0的距离s

	while (len(centList) < k):		#达到划分的簇数,结束划分
		lowestSSE = inf
		for i in range(len(centList)):	# 尝试划分每一簇
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]	# 属于当前簇的节点们
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
			sseSplit = sum(splitClustAss[:,1])
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0],1])

			if (sseSplit + sseNotSplit) < lowestSSE:		# 如果划分后簇的总误差变小
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
				lowestSSE = sseSplit + sseNotSplit

		# 找到了最小的误差值及对应的划分簇 bestCentToSplit
		print '最佳划分簇: ', bestCentToSplit

		# 更新簇分配结果(kMeans得到两簇编号分别为0和1的簇)
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList)	# 新加簇编号
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit	# 划分簇编号

		centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
		centList.append(bestNewCents[1,:].tolist()[0])
		clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

	return mat(centList), clusterAssment


if __name__ == '__main__':
	#datMat = mat(loadDataSet('testSet.txt'))
	#print randCent(datMat, 3)
	#myCentroids, clustAssing = kMeans(datMat, 4)
	#print myCentroids
	#print clustAssing

	datMat = mat(loadDataSet('testSet2.txt'))
	centList, clusterAssment = biKmeans(datMat, 3)
	print centList
