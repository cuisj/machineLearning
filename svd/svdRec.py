# coding:utf-8

from numpy import *
from numpy import linalg as la

set_printoptions(linewidth='nan')

def loadExData():
	return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


def loadExData2():
	return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSim(inA, inB):
	return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
	if len(inA) < 3:
		return 1.0

	return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
	num = float(inA.T*inB)
	denom = la.norm(inA) * la.norm(inB)

	return 0.5 + 0.5 * (num / denom)


# 用户待评分的物品: dataMat[user][item]
# simMeas: 相似度计算算法

def standEst(dataMat, user, simMeas, item):
	# print "user:", user, "item:", item

	n = shape(dataMat)[1]	# 物品数

	simTotal = 0.0
	ratSimTotal = 0.0

	for j in range(n):
		userRating = dataMat[user, j]
		# print userRating
		if userRating == 0:				# 把user没有评分过的物品滤掉
			continue

		#print dataMat[:,item].A
		#print dataMat[:,j].A

		# 寻找user和其它用户都评级的物品
		overLap = nonzero(logical_and(dataMat[:,item].A > 0, dataMat[:,j].A > 0))[0] # 都评过级的物品的用户

		#print "overLap:", overLap, "item:", item, "j:", j

		if len(overLap) == 0:
			similary = 0
		else:
			similary = simMeas(dataMat[overLap, item], dataMat[overLap, j])

		# print dataMat[overLap, item]
		# print dataMat[overLap, j]

		simTotal += similary					# 总相似度
		ratSimTotal += similary * userRating	# 有用户偏好的相似度

	if simTotal == 0:
		return 0
	else:
		return ratSimTotal / simTotal			# 得到预测值


# 给指定的用户推荐其未评分的物品, 最多N个
# 返回此用户对相应物品的评分
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
	# 寻找未评级的物品

	# print dataMat[user, :].A1
	# print nonzero(dataMat[user,:].A1 == 0)

	unratedItems = nonzero(dataMat[user,:].A1 == 0)[0]	# 求评级的物品列表

	if len(unratedItems) == 0:
		return 'you rated everything'

	itemScores = []
	for item in unratedItems:
		estimatedScore = estMethod(dataMat, user, simMeas, item)
		itemScores.append((item, estimatedScore))	# 物品，评分

	return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]	# 按评分大小排序


# 总能量信息：奇异值的平方和累加

def svdEst(dataMat, user, simMeas, item):
	n = shape(dataMat)[1]

	simTotal = 0.0
	ratSimTotal = 0.0

	U, Sigma, VT = la.svd(dataMat)
	Sig4 = mat(diag(Sigma[:4]))

	# print dataMat.shape

	xformedItems = dataMat.T * U[:,:4] * Sig4.I
	# print xformedItems.shape

	for j in range(n):
		userRating = dataMat[user, j]
		if userRating == 0 or j == item:
			continue
		similary = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
		print 'the %d and %d similarity is: %f' % (item, j, similary)
		simTotal += similary
		ratSimTotal += similary * userRating

	if simTotal == 0:
		return 0
	else:
		return ratSimTotal / simTotal

# 压缩图像

def printMat(inMat, thresh=0.8):
	for i in range(32):
		for k in range(32):
			if float(inMat[i,k]) > thresh:
				print 1,
			else:
				print 0,
		print ''

def imgCompress(numSV=3, thresh=0.8):
	myl = []
	with open('0_5.txt') as f:
		for line in f.readlines():
			newRow = []
			for i in range(32):
				newRow.append(int(line[i]))
			myl.append(newRow)
	myMat = mat(myl)

	print "初始图像矩阵"
	printMat(myMat, thresh)

	U, Sigma, VT = la.svd(myMat)
	SigRecon = mat(zeros((numSV, numSV)))
	for k in range(numSV):
		SigRecon[k,k] = Sigma[k]

	reconMat = U[:,:numSV] * SigRecon * VT[:numSV, :]
	print "\n压缩图像矩阵"
	printMat(reconMat, thresh)



if __name__ == '__main__':
	#Data = loadExData()
	#U, Sigma, VT = linalg.svd(Data)

	#Sig3 = mat(diag(Sigma[:3])) #Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
	#print Sig3

	#data = U[:,:3] * Sig3 * VT[:3,:]
	#print data

	#DatMat = mat(Data)
	#print ecludSim(DatMat[:,0], DatMat[:,4])
	#print ecludSim(DatMat[:,0], DatMat[:,0])

	#print pearsSim(DatMat[:,0], DatMat[:,4])
	#print pearsSim(DatMat[:,0], DatMat[:,0])

	#print cosSim(DatMat[:,0], DatMat[:,4])
	#print cosSim(DatMat[:,0], DatMat[:,0])

	#myMat = mat(loadExData())
	#myMat[0,1]=myMat[0,0]=myMat[1,0]=myMat[2,0]=4
	#myMat[3,3]=2

	#print myMat
	#print recommend(myMat, 2)

	myMat = mat(loadExData2())
	print recommend(myMat, 1, estMethod=svdEst)

	#imgCompress(2)
