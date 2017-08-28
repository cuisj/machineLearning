# -*- encoding: utf-8 -*-

from numpy import *

#输出: 文章列表和对应分类标签
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    # 侮辱: 1
    return postingList,classVec

# 输入: 所有文章的列表
# 输出: 所有词汇集
def createVocabList(dataSet):
    vocabSet = set([])  # 词汇集
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 并集
    return list(vocabSet)

# 输入: 词汇集，文章
# 输出: 基为词汇表的文档词汇坐标向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec


#
#           p(w|ci)p(ci)
# p(ci|w) = ------------
#               p(w)
#

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) # 文章总量
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 得p1 侮辱性文章概率, p0 = 1 - p1

    numWords = len(trainMatrix[0]) # 词汇总量

    p0Num = zeros(numWords)
    p0Denom = 0.0

    p1Num = zeros(numWords)
    p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    #print p1Num
    #print p1Denom

    p1Vect = p1Num/p1Denom # 侮辱性文章类别中，各个词汇出现的概率     p(w0|c1), p(w1|c1), ..., p(wn|c1)
    p0Vect = p0Num/p0Denom # 非侮辱性文章类别中，各个词汇出现的概率   p(w0|c0), p(w1|c0), ..., p(wn|c0)

    #假设词汇独立，那么p(w|ci) = p(w0|ci)p(w1|ci)...p(wn|ci)

    return p0Vect,p1Vect,pAbusive


def trainNB1(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    numWords = len(trainMatrix[0])

    p0Num = ones(numWords)  # 为防止概率出现0，初始化为1
    p1Num = ones(numWords)
    p0Denom = 2.0           # 合理假设分母为2
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num/p1Denom)          # 防目连乘时数太小而失真，转为求对数和的形式
    p0Vect = log(p0Num/p0Denom)

    # ln(a * b) = ln(a) + ln(b)

    return p0Vect,p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):

    # sum(vec2Classify * p1Vec) 求出联合概率 p(w0|c1)p(w1|c1)...p(wn|c1)
    # 根据全概率公式 p(w) = p(w|c0) + p(w|c1), 但p(w)没有计算的必要, 因为对于p(w|ci)比较来说没有贡献

    p1 = sum(vec2Classify * p1Vec) + log(pClass1)           # p(w0|c1)p(w1|c1)...p(wn|c1)p(c1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)     # p(w0|c1)p(w1|c1)...p(wn|c1)p(c1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB1(array(trainMat),array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

    testEntry = ['stupid', 'garbage', 'stupid']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    #print listOPosts
    #print listClasses

    myVocabList = createVocabList(listOPosts)
    #print(myVocabList)

    postingVocabVec = setOfWords2Vec(myVocabList, listOPosts[0])
    #print postingVocabVec

    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #print trainMat

    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    #print p0V
    #print p1V
    #print pAb

    p0V, p1V, pAb = trainNB1(trainMat, listClasses)
    #print p0V
    #print p1V
    #print pAb

    testingNB()