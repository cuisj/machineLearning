# coding:utf-8

from numpy import *


def loadSimpleData():
	datMat = matrix([[1.0, 2.1],
					[2.0, 1.1],
					[1.3, 1.0],
					[1.0, 1.0],
					[2.0, 1.0]])

	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

	return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshineq):
	m, _ = shape(dataMatrix)
	retArray = ones((m,1))

	if threshineq == 'lt': # 属性dimen小于threshVal的样本retArray都记为-1类
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0

	return retArray

# 单层决策树生成函数（基分类器）
# 其时不知道按哪个属性的哪个值作为分界点，就逐个尝试，然后得到加权的错误率
# 然后选择使加权错误率最小的那个

def buildStump(dataArr, classLabels, D):
	dataMatrix = mat(dataArr)
	labelMat = mat(classLabels).T
	m, n = shape(dataMatrix)

	numSteps = 10.0
	bestStump = {}						# 分类字典
	bestClassEst = mat(zeros((m,1)))	# 类别估计值
	minError = inf						# 错误率

	for i in range(n): # 每一个属性
		rangeMin = dataMatrix[:, i].min()			# 属性最小值
		rangeMax = dataMatrix[:, i].max()			# 属性最大值
		stepSize = (rangeMax - rangeMin) / numSteps	# 步长

		for j in range(-1, int(numSteps) + 1):
			for inequal in ['lt', 'gt']:
				threshVal = (rangeMin + float(j) * stepSize)		# 逐步分析属性在何值时分类样本最好
				predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)

				errArr = mat(ones((m,1)))	# 被错分的样本
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T * errArr	# 加权错误, 越小越好

				#print "split: dim %d, thresh %0.2f, thresh ineqal: %s, weighterr is %f" % (i, threshVal, inequal, weightedError)

				if weightedError < minError:
					minError = weightedError
					bestClassEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal

	return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabels, numlt=40):
	weakClassArr = []	# 弱分类器组

	m = shape(dataArr)[0]
	D = mat(ones((m, 1))/m)

	aggClassEst = mat(zeros((m,1)))		# 分类器估计分类加权线性组合

	for i in range(numlt):
		bestStump, error, classEst = buildStump(dataArr, classLabels, D) # 弱分类器结果
		# print "D:", D.T
		alpha = float( 0.5 * log((1 - error) / max(error, 1e-16)))	# 防止error为零时出错
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)	# 完成一个弱分类器，进入下一轮

		# 更新D
		expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
		D = multiply(D, exp(expon))
		D = D / D.sum()

		aggClassEst += alpha * classEst
		Hx = sign(aggClassEst)			# 得出估计分类值
		# 计算总体错误率
		aggErrors = multiply(Hx != mat(classLabels).T, ones((m,1)))	# 错误的分类向量
		errorRate = aggErrors.sum() / m
		#print "total error:", errorRate

		if errorRate == 0.0:	# 全部分类正确
			break

	return weakClassArr

# 分类
def adaClassify(datToClass, classifierArr):
	dataMatrix = mat(datToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))

	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix,
								 classifierArr[i]['dim'],
								 classifierArr[i]['thresh'],
								 classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha'] * classEst
		#print aggClassEst

	return sign(aggClassEst)



if __name__ == '__main__':
	datMat, classLabels = loadSimpleData()
	#D = mat(ones((5,1))/5)					# 初始样本权重
	#bestStump, minError, bestClassEst = buildStump(datMat, classLabels, D)
	#print "bestStump", bestStump
	#print "minError", minError
	#print "bestClassEst", bestClassEst

	weakClassArr = adaBoostTrainDS(datMat, classLabels, 9)
	print adaClassify(datMat, weakClassArr)