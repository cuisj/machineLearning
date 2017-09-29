# coding:utf-8


from numpy import *

def loadDataSet(filename):
	dataMat = []

	with open(filename) as fr:
		for line in fr.readlines():
			curLine = line.strip().split('\t')
			fitLine = map(float, curLine)		# 属性数据都转成float型
			dataMat.append(fitLine)

	return dataMat


def regLeaf(dataSet):
	return mean(dataSet[:, -1])

def regErr(dataSet):
	return var(dataSet[:, -1]) * shape(dataSet)[0]

# 二分
def binSplitDataSet(dataSet, feature, value):
	# nonzero(dataSet[:, feature] > value) 返回大于value值的属性所在的行值与列值

	mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0],:][0]
	mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0],:][0]
	return mat0, mat1

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None:	# 叶子节点
		return val

	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val

	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)

	return retTree		# 分支节点

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	toIS = ops[0]	# 容许的误差下降值
	toIN = ops[1]	# 切分的最少样本数

	if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)

	m,n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf
	bestIndex = 0
	bestValue = 0

	# 找到最小的总偏差对应的属性
	for featIndex in range(n-1):
		for splitVal in set(dataSet[:, featIndex]):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if (shape(mat0)[0] < toIN) or (shape(mat1)[0] < toIN):
				continue
			newS = errType(mat0) + errType(mat1)

			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS

	if (S - bestS) < toIS:				# 偏差太小就不再划分
		return None, leafType(dataSet)

	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	if (shape(mat0)[0] < toIN) or (shape(mat1)[0] < toIN):	# 划分的元素太少不再划分
		return None, leafType(dataSet)

	return bestIndex, bestValue


def isTree(obj):
	return (type(obj).__name__ == 'dict')

def getMean(tree):	# 找到树的平均值
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])

	return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    if shape(testData)[0] == 0:	# 没有测试数据了
		return getMean(tree)

	# 如果左右是树，就对左右分枝剪枝
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

    if isTree(tree['left']):
		tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
		tree['right'] =  prune(tree['right'], rSet)


    # 如果左右都不是分枝了，试着合并，进行剪枝操作
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) + sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))

        if errorMerge < errorNoMerge:	# 合并后误差小于合并前， 合并
            print "merging"
            return treeMean
        else: return tree
    else: return tree



def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

if __name__ == '__main__':
	#testMat = mat(eye(4))
	#print(testMat)

	#mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
	#print mat0
	#print mat1

	#myDat1 = loadDataSet('ex0.txt')
	#myMat1 = mat(myDat1)
	#t = createTree(myMat1)
	#print(t)

	#myDat2 = loadDataSet('ex2.txt')
	#myMat2 = mat(myDat2)
	#t = createTree(myMat2)
	#print(t)

	#myTree = createTree(myMat2, ops=(0,1))
	#print myTree
	#myDatTest =loadDataSet('ex2test.txt')
	#myMat2Test = mat(myDatTest)
	#t = prune(myTree, myMat2Test)
	#print t

	myMat2 = mat(loadDataSet('exp2.txt'))
	t = createTree(myMat2, modelLeaf, modelErr, (1,10))
	print t
