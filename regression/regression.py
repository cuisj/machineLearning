# coding:utf-8

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []

    with open(fileName) as f:
        numFeat = len(f.readline().split('\t')) - 1
        f.seek(0, 0)
        for line in f.readlines():
            curLine = line.strip().split('\t')
            dataMat.append([float(feat) for feat in curLine[:numFeat]])
            labelMat.append(float(curLine[-1]))

        return dataMat, labelMat


# y = ws[0] * ws[1] * X1
def standRegres(xArr, yArr):
    # 转为向量
    xMat = mat(xArr)
    yMat = mat(yArr).T

    xTx = xMat.T * xMat

    if linalg.det(xTx) == 0.0:
        print "xTx不可逆"
        return

    ws = xTx.I * (xMat.T * yMat)

    # 返回回归系数
    return ws

def showStandRegresFigure(xArr, yArr, ws):
    xMat = mat(xArr)
    yMat = mat(yArr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 原始点集
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])

    # 回归曲线
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws

    ax.plot(xCopy[:,1].flatten().A[0], yHat.flatten().A[0])
    plt.show()

def correlationCoefficient(xArr, yArr, ws):
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = (xMat * ws)

    return  corrcoef(yHat.T, yMat)

def testStandRegres():
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    print correlationCoefficient(xArr, yArr, ws)
    showStandRegresFigure(xArr, yArr, ws)


# 局部加权线性回归
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))

    # 创建对角加权矩阵, 使用高斯核
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))

    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0:
        print "xTx不可逆"
        return

    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)

    return yHat

def showLWLRFigure(xArr, yArr, k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 原始点集
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s=2, c='red')

    yHat = lwlrTest(xArr, xArr, yArr, k)

    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:,0]

    ax.plot(xSort[:,1].flatten().A[0], yHat[srtInd].flatten())

    plt.show()


def testLWLR():
    xArr, yArr = loadDataSet('ex0.txt')
    # print lwlr(xArr[0], xArr, yArr, 0.01)
    # print lwlrTest(xArr, xArr, yArr, 0.01)
    showLWLRFigure(xArr, yArr, 0.01)

if __name__ == '__main__':
    pass