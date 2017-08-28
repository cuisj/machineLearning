#! /usr/bin/python
#-* encoding: utf8 -*-


import struct
import numpy
import os
import csv
import operator
import matplotlib.pyplot as plt

numpy.set_printoptions(linewidth=128)

#展示原始图像
def showImagePicture(imageFilename):
	with open(imageFilename, 'rb') as f:
		imgdat = f.read()
		pixdat = struct.unpack_from('>784B', imgdat, 0)
		#print(pixdat)

		img = numpy.array(pixdat)
		#print(img)

		img = img.reshape(28, 28)
		print(img)
		
		#fig = plt.figure()
		#wnd = fig.add_subplot(111)
		#plt.imshow(img, cmap='gray')
		#plt.show()

#归一化
def loadImageAndUnify(imageFilename):
	with open(imageFilename, 'rb') as f:
		imgdat = f.read()
		pixdat = struct.unpack_from('>784B', imgdat, 0)
		img = numpy.array(pixdat)
		img = img.reshape(28, 28)
		#print(img)

		img[img > 0] = 1
		print (img)

def loadTrainImages(imagePath, offset, count):
	imageMap = {}

	# 读取图像
	imageFilenames = os.listdir(imagePath)
	for fn in imageFilenames[offset: offset + count]:
		fullname = os.path.join(imagePath, fn)
		
		with open(fullname, 'rb') as f:
			imgdat = f.read()
			pixdat = struct.unpack_from('>784B', imgdat, 0)
			img = numpy.array(pixdat)
			img[img > 0] = 1
			imageMap[fn] = img

	return imageMap

def loadTrainLabels(labelFilename):
	labelMap = {}
	
	#读取标签
	with open(labelFilename, 'rb') as f:
		labd = csv.reader(f)
		
		for (fn, lab) in labd:
			labelMap[fn] = lab

	return labelMap


def classify0(imageX, trainImages, trainLabels, k):
	labelDistances = []

	guessLabelDistance = 784
	guessLabel = -1

	# 计算矩离 ||x-y||**2 = (x1-y1)**2 + (x2 - y2)**2 + ... + (xn- yn)**2
	for (fn, img) in trainImages.items():
		distance = imageX - img
		squaredDistance = distance**2
		sumSquaredDistance = squaredDistance.sum()

		if sumSquaredDistance < guessLabelDistance:
			guessLabelDistance = sumSquaredDistance
			guessLabel = trainLabels[fn]

		if sumSquaredDistance > 98:
			continue

		labelDistances.append((sumSquaredDistance, trainLabels[fn]))

	#print(labelDistances)

	# 矩离超出98的，选择一个最近的
	if len(labelDistances) <= 0:
		return guessLabel

	# 根据距离排序
	labelDistances.sort()
	#print(labelDistances)
	
	# 选择距离最小的k个点, 并计算标签频率
	labelCount = {}
	labelTotalDist = {}
	for (count, label)  in labelDistances[:k]:
		labelCount[label] = labelCount.get(label, 0) + 1
		labelTotalDist[label] = labelTotalDist.get(label, 0) + count

	#print(labelCount)

	# 根据频率排序
	sortedLabelCount = sorted(labelCount.iteritems(), key = operator.itemgetter(1), reverse=True)

	likelyLabel, likelyLabelFreq = sortedLabelCount[0]
	likelyLabelTotalDist = labelTotalDist[likelyLabel]

	# 当频率相同时，找距离更近的那个标签
	for label, freq in sortedLabelCount[1:]:
		if freq >= likelyLabelFreq and labelTotalDist[label] < labelTotalDist[likelyLabel]:
			likelyLabel = label
			likelyLabelFreq = freq

	print(likelyLabel)

	# 返回频率最高的标签
	return likelyLabel


# 提升准确率，自测之用
def classifyTrain():
	trainImages = loadTrainImages('train', 10000, 50)
	trainLabels = loadTrainLabels('train.csv')
	testImages = loadTrainImages('train', 25000, 3)

	tot = 0
	err = 0
	cnt = 0.0
	for (fn, img) in testImages.items():
		probeValue = classify0(img, trainImages, trainLabels, 3)

		tot += 1
		
		if trainLabels[fn] == probeValue:
			cnt += 1
		else:
			err += 1

			print "wrong:", probeValue, "≠", "[", trainLabels[fn], "]"

	print "正确率:", cnt / len(testImages)

if __name__ == '__main__':
    #showImagePicture('train/0000fc0d40fe4c18a72481f50b305e54')
    #loadImageAndUnify('train/0000fc0d40fe4c18a72481f50b305e54')
    classifyTrain()
