# coding:utf-8

class treeNode:
	def __init__(self, nameValue, numOccur, parentNode):
		self.name = nameValue
		self.count = numOccur
		self.parent = parentNode
		self.nodeLink = None
		self.children = {}

	def inc(self, numOccur):
		self.count += numOccur

	def disp(self, ind = 1):
		print ' '*ind, self.name, '', self.count

		for child in self.children.values():
			child.disp(ind+1)


def createTree(dataSet, minSup=1):
	headerTable = {}

	for trans in dataSet:
		for item in trans:
			headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

	# 移除不满足最小支持度的元素项
	for k in headerTable.keys():
		if headerTable[k] < minSup:
			del(headerTable[k])

	freqItemSet = set(headerTable.keys())

	# 如果没有元素满足要求，则退出
	if len(freqItemSet) == 0:
		return None, None

	for k in headerTable:
		headerTable[k] = [headerTable[k], None]		# [1] 指向此类型第一个元素的引用

	retTree = treeNode('Null Set', 1, None)			# 空节点（根节点）

	for tranSet, count in dataSet.items():
		localD = {}

		# 根据全局频率对每个事务中的元素进行排序, 排好序之后插入树中时，从上到下无需再排序了
		for item in tranSet:
			if item in freqItemSet:
				localD[item] = headerTable[item][0]

		#
		if len(localD) > 0:
			orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
			updateTree(orderedItems, retTree, headerTable, count)

	return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
	if items[0] in inTree.children:						# 第一个元素项在树中，更新计数
		inTree.children[items[0]].inc(count)
	else:
		inTree.children[items[0]] = treeNode(items[0], count, inTree)	# 如果不再，添加元素项
		if headerTable[items[0]][1] == None:			# 更新头指针
			headerTable[items[0]][1] = inTree.children[items[0]]			# growth的来历，根深叶茂
		else:
			updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

	if len(items) > 1:			# 递归处理后续元素项
		updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
	while (nodeToTest.nodeLink != None):
		nodeToTest = nodeToTest.nodeLink

	nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):		# 上溯整颗树
	if leafNode.parent != None:
		prefixPath.append(leafNode.name)
		ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
	condPats = {}

	while treeNode != None:
		prefixPath = []
		ascendTree(treeNode, prefixPath)
		if len(prefixPath) > 1:				# 去掉自已
			condPats[frozenset(prefixPath[1:])] = treeNode.count
		treeNode = treeNode.nodeLink		# 后续元素项

	return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):

	# 从头指针表的底端开始
	bigL = [v[0] for v in sorted(headerTable.items(), key = lambda p: p[1])]

	for basePat in bigL:
		newFreqSet = preFix.copy()
		newFreqSet.add(basePat)
		freqItemList.append(newFreqSet)
		condPattBases = findPrefixPath(basePat, headerTable[basePat][1])

		# 从条件模式基来构建条件FP树
		myCondTree, myHead = createTree(condPattBases, minSup)

		# 挖掘条件FP树
		if myHead != None:
			#print 'condition tree for:', newFreqSet
			#myCondTree.disp(1)
			mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def loadSimpDat():
	simpDat = [['r', 'z', 'h', 'j', 'p'],
				['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
				['z'],
				['r', 'x', 'n', 'o', 's'],
				['y', 'r', 'x', 'z', 'q', 't', 'p'],
				['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]

	return simpDat

def createInitSet(dataSet):
	retDict = {}
	for trans in dataSet:
		retDict[frozenset(trans)] = 1

	return retDict


if __name__ == '__main__':
	simpDat = loadSimpDat()
	initSet = createInitSet(simpDat)

	print initSet

	myFPTree, myHeaderTab = createTree(initSet, 3)
	#myFPTree.disp()
	#print myHeaderTab

	#print findPrefixPath('x', myHeaderTab['x'][1])
	#print findPrefixPath('r', myHeaderTab['r'][1])
	#print findPrefixPath('t', myHeaderTab['t'][1])

	freqItems = []
	mineTree(myFPTree, myHeaderTab, 3, set([]), freqItems)
	print freqItems
