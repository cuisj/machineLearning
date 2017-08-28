# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
	dataMat = []
	labelMat = []

	with open('testSet.txt') as f:
		for line in f.readlines():
			lineArr = line.strip().split()
			dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) # wx + b = wx + w0*1
			labelMat.append(int(lineArr[2]))

	return dataMat, labelMat


def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))


# 最大似然估计，找到最大的估计值
# 参考: http://secfree.github.io/blog/2017/01/01/questions-about-logistic-regression-in-machine-learning-in-action-and-full-explanation.html
# P(y|x;w) = (h(x))**y * (1 - h(x))**(1-y)
def gradAscent(dataMatin, classLabels):
	dataMatrix = np.mat(dataMatin)
	labelMat = np.mat(classLabels).transpose()	# 转置
	m,n = np.shape(dataMatrix)					# m x n
	alpha = 0.001
	maxCycles = 500

	weights = np.ones((n, 1))	# n x 1

	for k in range(maxCycles):
		h = sigmoid(dataMatrix*weights)	# 预测值, 所有记录
		error = (labelMat - h)			# 实差值, 所有记录

		#print (labelMat - h).sum()		# 总差值, 趋近于零

		grad = dataMatrix.transpose() * error	# 对w的梯度
		weights = weights + alpha * grad		# 根据梯度不断调整变量weights值，最终的值就是能使似然函数取得最大的值

	return weights

# 随机梯度上升法, 减少计算量，逐个记录进行计算
def stocGradAscent0(dataMatrix, classLabels, numlter = 150):
	m, n = np.shape(dataMatrix)
	alpha = 0.01
	weights = np.ones(n)

	for j in range(numlter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4 / (1.0 + j + i) + 0.01
			randIndex = int(np.random.uniform(0, len(dataIndex)))	# 随机采样

			h = sigmoid(sum(dataMatrix[randIndex]*weights))
			error = classLabels[randIndex] - h
			grad = error * dataMatrix[randIndex]
			weights = weights + alpha * grad
			del(dataIndex[randIndex])

	return weights



def plotBestFit(wei):
	weights = wei.getA()				# matrix转成数组
	dataMat, labelMat = loadDataSet()
	dataArr = np.array(dataMat)			# matrix转成数组
	n = np.shape(dataArr)[0]			# 记录数

	xcord1 = []
	ycord1 = []
	xcord2 = []
	ycord2 = []

	for i in range(n):
		if int(labelMat[i] == 1):
			xcord1.append(dataArr[i, 1])	# 类别为1的记录的属性1
			ycord1.append(dataArr[i, 2])	# 类别为1的记录的属性2
		else:
			xcord2.append(dataArr[i, 1])
			ycord2.append(dataArr[i, 2])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')		# s:点的形状大小, c: 颜色 marker:点的形状样式,
	ax.scatter(xcord2, ycord2, s=30, c='blue')
	x1 = np.arange(-3.0, 3.0, 0.1)
	x2 = (-weights[0] - weights[1] * x1) / weights[2]				# WX=0, w0 + w1x + w2y = 0, sigmoid(X) = 0.5 分界线
	ax.plot(x1,x2)
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()



if __name__ == '__main__':
	ds, lb = loadDataSet()
	#weights = gradAscent(ds, lb)
	#plotBestFit(weights)

	weights = stocGradAscent0(np.array(ds), lb, 500)
	weights = np.mat(weights).reshape(3, 1)
	plotBestFit(weights)
