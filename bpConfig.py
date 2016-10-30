#!usr/bin/python
#coding=utf-8

'''
返回列表
'''
def nList(i, j=0):
    if j==0:
        return [0] * i
    else:
        return [[0 for col in range(j)] for row in range(i)]

NUM_I = 784 #输入层单元个数
NUM_H = 20 #隐藏层单元个数
NUM_O = 10 #输出层单元个数

THRESHOD = 0.99 #正确率
RATE =0.05 #学习速率
TRAIN_TIMES =20 #最大训练次数

PATH = "./dataset/";
TRAIN_FILE = PATH+"train-images-idx3-ubyte" #训练文件
LABEL_FILE = './dataset/train-labels-idx1-ubyte'
TEST_FILE = PATH+"train-labels-idx1-ubyte" #测试文件

WEIGHT_IH_FILE = './save/weight_ih'
WEIGHT_HO_FILE = './save/weight_ho'
BIAS_H_FILE = './save/bias_h'
BIAS_O_FILE = './save/bias_o'

'''偏移量和权值矩阵需要训练，作为全局变量'''
weight_ih = nList(NUM_I, NUM_H) #输入层到隐藏层的权值矩阵
weight_ho = nList(NUM_H, NUM_O) #隐藏层到输出层的权值矩阵
bias_h = nList(NUM_H) #隐藏层的偏移量
bias_o = nList(NUM_O) #输出层的偏移量

delta_weight_ih = nList(NUM_I, NUM_H) #输入层到隐藏层的权值增量矩阵
delta_weight_ho = nList(NUM_H, NUM_O) #隐藏层到输出层的权值增量矩阵
delta_bias_h = nList(NUM_H) #隐藏层的偏移量增量
delta_bias_o = nList(NUM_O) #输出层的偏移量增量