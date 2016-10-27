#!usr/bin/python
#coding=utf-8
import reader
import bpConfig as CONFIG
import random
import bpUtils as utils
import math
import time

def initWeightAndBias():
    #初始化隐藏层
    for i in range(CONFIG.NUM_H):
        CONFIG.bias_h[i] = getRandomNumber()
        for j in range(0, CONFIG.NUM_I):
            CONFIG.weight_ih[j][i] = getRandomNumber()

    #初始化输出层
    for i in range(CONFIG.NUM_O):
        CONFIG.bias_o[i] = getRandomNumber()
        for j in range(CONFIG.NUM_H):
            CONFIG.weight_ho[j][i] = getRandomNumber()

def getRandomNumber():
    return (random.randint(0, 1024) - 512) / 512.0

def getMaxIndex(inList):
    max = -999
    index = 0
    for i in range(len(inList)):
        if (inList[i] > max):
            max = inList[i]
            index = i
    return index

'''
计算输入层
'''
def initInputLayer(inList):
    if len(inList) != CONFIG.NUM_I:
        print '从文件将读取的属性个数与您定义的NUM_I不一致！'
    return inList

'''
根据输入层，权值矩阵和偏移量计算隐藏层每个神经单元
'''
def computHiddenUnit(inList):
    newList = [0] * CONFIG.NUM_H
    for i in range(CONFIG.NUM_H):
        sum = 0.0
        for j in range(CONFIG.NUM_I):
            sum += inList[j] * CONFIG.weight_ih[j][i]
        sum += CONFIG.bias_h[i]
        # print 'b_' + str(CONFIG.bias_h[i])
        # print 'sum_' + str(sum)
        if -sum > 700:
            newList[i] = 0
        else:
            newList[i] = 1 / (1 + math.exp(-sum))
    return newList

'''
根据隐藏层，权值矩阵，和偏移量计算输出层每个神经单元
'''
def computOutputUnit(inList):
    newList = [0] * CONFIG.NUM_O
    for i in range(CONFIG.NUM_O):
        sum = 0.0
        for j in range(CONFIG.NUM_H):
            sum += inList[j] * CONFIG.weight_ho[j][i]
        sum += CONFIG.bias_h[i]
        if -sum > 700:
            newList[i] = 0
        else:
            newList[i] = 1 / (1 + math.exp(-sum))
    return newList


'''
@description 对数据进行测试，并计算正确率
@param {list<int>} labelList 正确的label列表
@param {list<list>} imageList 图片灰度列表
'''
def experimentOnModel(labelList, imageList):
    labelListOutput = [0]*CONFIG.NUM_O
    labelListCorrect = [0]*CONFIG.NUM_O
    correctTotal = 0
    for i in range(len(imageList)):
        image = imageList[i]
        if (len(image) == CONFIG.NUM_I):
            labelListOutput[labelList[i]] += 1
            inputLayer = initInputLayer(imageList[i]) #计算输入层
            hiddenLayter = computHiddenUnit(inputLayer); #计算隐藏层
            outputLayer = computOutputUnit(hiddenLayter); #计算输出层

            label = getMaxIndex(outputLayer)

            if labelList[i] == label:
                correctTotal += 1
                labelListCorrect[label] += 1

                if label >= 16:
                    print '大于16了，这个逻辑我真是看不懂'

    return correctTotal / float(len(imageList))

def computOutputError(outLayer, labels):
    outError = [0] * CONFIG.NUM_O
    for i in range(CONFIG.NUM_O):
        outError[i] = outLayer[i] * (1.0 - outLayer[i]) * (labels[i] - outLayer[i])
    return outError

def computHiddenError(outError, hiddenLayer):
    hiddenError = [0] * CONFIG.NUM_H
    for i in range(CONFIG.NUM_H):
        sum = 0.0
        for j in range(CONFIG.NUM_O):
            sum += outError[j] * CONFIG.weight_ho[i][j]
        hiddenError[i] = hiddenLayer[i] * (1.0 - hiddenLayer[i]) * sum
    return hiddenError

def updateHidden(hiddenError, inputLayer):
    for j in range(CONFIG.NUM_H):
        for i in range(CONFIG.NUM_I):
            CONFIG.weight_ih[i][j] += hiddenError[j] * 1.0 * inputLayer[i] * CONFIG.RATE
        CONFIG.bias_h[j] += hiddenError[j] * CONFIG.RATE

def updateOutput(outputError,hiddenLayer):
    for j in range(CONFIG.NUM_O):
        for i in range(CONFIG.NUM_H):
            CONFIG.weight_ho[i][j] += outputError[j] * 1.0 * hiddenLayer[i] * CONFIG.RATE
        CONFIG.bias_o[j] += outputError[j] * CONFIG.RATE

'''
@description 保存训练后的参数到文件
'''
def trainModel(label, image):
    if len(image) == CONFIG.NUM_I:
        inputLayer = initInputLayer(image); #计算输入层
        hiddenLayer = computHiddenUnit(inputLayer); #计算隐藏层
        outputLayer = computOutputUnit(hiddenLayer); #计算输出层
        expectedLabelList = [0] * CONFIG.NUM_O #期望得到的label
        expectedLabelList[label] = 1

        outputError = computOutputError(outputLayer, expectedLabelList); #计算输出层误差
        hiddenError = computHiddenError(outputError,hiddenLayer); #计算隐藏层误差
        updateHidden(hiddenError,inputLayer); #更新隐藏层的权重和偏移量
        updateOutput(outputError,hiddenLayer); #更新输出层的权重和偏移量

'''
@description 保存训练后的参数到文件
'''
def saveResultToFile():
    reader.saveList(CONFIG.WEIGHT_IH_FILE, CONFIG.weight_ih)
    reader.saveList(CONFIG.WEIGHT_HO_FILE, CONFIG.weight_ho)
    reader.saveList(CONFIG.BIAS_O_FILE, CONFIG.bias_o)
    reader.saveList(CONFIG.BIAS_H_FILE, CONFIG.bias_h)


if __name__ == '__main__':
    print '----------bp main----------'

    #初始参数
    imageNumber = 10

    #初始化权值矩阵等
    initWeightAndBias()

    #读取label数据
    labelReader = reader.BitFileReader()
    labelReader.open(CONFIG.LABEL_FILE)
    labelReader.step(8)
    labelList = []
    for i in range(imageNumber):
        labelList.append(labelReader.read())
    labelReader.close()
    print labelList

    #读取图片数据
    imageReader = reader.BitFileReader()
    imageReader.open(CONFIG.TRAIN_FILE)
    imageReader.step(16) #由于前面的都是无用数据，直接跳到16这个位置
    #utils.printImage(utils.readImage(imageReader))

    imageList = []
    for i in range(imageNumber):
        imageList.append(utils.readImageLinear(imageReader))
    imageReader.close()
    # for i in range(2):
    #     utils.printImage(imageList[i])

    # 训练开始
    accuracy = experimentOnModel(labelList, imageList)
    print '初始准确率：' + str(accuracy)

    trainTimes = 0
    startTime = time.time()
    while True:
        trainTimes += 1
        if accuracy > CONFIG.THRESHOD or trainTimes > CONFIG.TRAIN_TIMES:
            break
        for i in range(len(imageList)):
            trainModel(labelList[i], imageList[i])

        if trainTimes % 10 == 0:
            accuracy = experimentOnModel(labelList, imageList)
            print '第%d次训练，准确率：%f' % (trainTimes, accuracy)
            print '耗时%fs' % (time.time() - startTime)

    saveResultToFile()

