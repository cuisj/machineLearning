# coding:utf-8


def loadDataSet():
	return [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]

def createC1(dataSet):						# 创建候选项集
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()

	return map(frozenset, C1)


def scanD(D, Ck, minSupport):
	ssCnt = {}

	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				if not ssCnt.has_key(can):
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1

	numItems = float(len(D))

	retList = []		# 满足最小支持度要求的项集
	supportData = {}	# 项集支持度

	for key in ssCnt:						# 支持度项集
		support = ssCnt[key] / numItems
		if support >= minSupport:
			retList.insert(0, key)
		supportData[key] = support

	return retList, supportData


def aprioriGen(Lk, k):			# 生成项集
	retList = []
	lenLk = len(Lk)

	for i in range(lenLk):
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2]		# k-2 减少重复计算
			L2 = list(Lk[j])[:k-2]

			L1.sort()
			L2.sort()
			if L1 == L2:
				retList.append(Lk[i] | Lk[j])

	return retList


def apriori(dataSet, minSupport = 0.5):					# 得到各个项集的支持度
	C1 = createC1(dataSet)
	D = map(set, dataSet)
	L1, supportData = scanD(D, C1, minSupport)

	L = [L1]
	k = 2

	while (len(L[k-2]) > 0):
		Ck = aprioriGen(L[k-2], k)
		Lk, supK = scanD(D, Ck, minSupport)		# 过滤项集
		supportData.update(supK)
		L.append(Lk)
		k += 1

	return L, supportData


def generateRules(L, supportData, minConf = 0.7):		# 根据项集的支持度得到各个关联规则
	bigRuleList = []

	for i in range(1, len(L)):		# 从包含两个及以上元素的项集开始
		for freqSet in L[i]:
			H1 = [frozenset([item]) for item in freqSet]
			if (i > 1):	# 两个以上元素的需要找出规则
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:		# 两个元素的直接计算
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)

	return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf=0.7):		# 计算可信度
	prunedH = []
	for conseq in H:
		conf = supportData[freqSet] / supportData[freqSet-conseq]
		if conf >= minConf:
			print freqSet - conseq, '-->', conseq, 'conf:', conf
			brl.append((freqSet-conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):	# 从项集中找出规则
	m = len(H[0])
	if (len(freqSet) > (m + 1)):
		Hmp1 = aprioriGen(H, m + 1)
		Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)

		if (len(Hmp1) > 1):
			rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


if __name__ == '__main__':
	dataSet = loadDataSet()
	print "dataSet: ", dataSet, "\n"

	#C1 = createC1(dataSet)
	#print C1

	#D = map(set, dataSet)
	#print D

	#L1, supportData0 = scanD(D, C1, 0.5)
	#print L1
	#print supportData0

	#print aprioriGen(L1, 2)

	#L, suppData = apriori(dataSet, minSupport = 0.7)
	#print L

	L, suppData = apriori(dataSet, minSupport = 0.5)
	#print "L: ", L
	#print "suppData: ", suppData, "\n"

	rules = generateRules(L, suppData, minConf = 0.7)
	print rules
