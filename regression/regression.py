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

def showFigure(xArr, yArr, ws):
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

if __name__ == '__main__':
    xArr, yArr = loadDataSet('ex0.txt')
    ws = standRegres(xArr, yArr)
    print correlationCoefficient(xArr, yArr, ws)
    showFigure(xArr, yArr, ws)
