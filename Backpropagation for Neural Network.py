# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 11:06:08 2020

@author: Zhan ao Huang, Backpropagation for Neural Network
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import random
#f(x)=x-y-1
def f(x,y):
    return x-y
def data():
    x0=[]
    y0=[]
    i=0 
    while True:
        tmp0=np.random.uniform(-1,1)
        tmp1=np.random.uniform(-1,1)
        if f(tmp0,tmp1)>0:
            x0.append(np.array([tmp0,tmp1]))
            y0.append(np.array([0,1]))
            i+=1
        elif f(tmp0,tmp1)<0:
            x0.append(np.array([tmp0,tmp1]))
            y0.append(np.array([1,0]))
            i+=1
        if i==100:
            break
    return np.array(x0),np.array(y0)
def SinData():
    x0=[]
    y0=[]
    for i in np.arange(-2,2,0.2):
        x0.append(np.array([i]))
        y0.append(np.array([1+math.sin(math.pi*i/4)]))
    return x0,y0
class NN():
    def __init__(self,input_number,hidden_number,output_number):
        self.input_layer=input_number
        self.hidden_layer=hidden_number
        self.output_layer=output_number
        
        self.weight1=np.zeros((self.hidden_layer,self.input_layer)) #self.hidden_layer,当前神经元个数，self.input_layer,前一层神经元个数
        for i in range(len(self.weight1)):
            for j in range(len(self.weight1[i])):
                self.weight1[i][j]=random.random()
        self.weight2=np.zeros((self.output_layer,self.hidden_layer))
        for i in range(len(self.weight2)):
            for j in range(len(self.weight2[i])):
                self.weight2[i][j]=random.random()
        self.bias1=np.zeros((self.hidden_layer,1))
        self.bias2=np.zeros((self.output_layer,1))
        
    def sigmoid(self,x):
        return 1.0/(1+math.e**(-x))
    def sigmoid_derivate(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def forward(self,x):
        self.x1=np.dot(self.weight1,x)+self.bias1;#要求输入x为列向量
        self.x1_sigmoid=self.sigmoid(self.x1)
        
        self.x2=np.dot(self.weight2,self.x1_sigmoid)+self.bias2;
        self.x2_sigmoid=self.sigmoid(self.x2)
        
        return self.x2_sigmoid
    
    def loss(self,y):#要求输入y为行向量
        return ((y-self.x2_sigmoid)**2).sum()
    def F_m(self,x):
        derivate_matrix=np.zeros((len(x),len(x)))
        for i in range(len(x)):
                derivate_matrix[i][i]=self.sigmoid_derivate(x[i][0])
        return derivate_matrix
    def loss_m(self,y):
        return -2*np.dot(self.F_m(self.x2),(y-self.x2_sigmoid))
        #return -2*(y-self.x2_sigmoid)
    def backpropagation(self,x,y):
        s2=self.loss_m(y)
        self.weight2=self.weight2-0.1*np.dot(s2,self.x1_sigmoid.transpose())
        self.bias2=self.bias2-0.1*s2
        
        s1=np.dot(np.dot(self.F_m(self.x1),self.weight2.transpose()),s2)
        
        self.weight1=self.weight1-0.1*np.dot(s1,x.transpose())
        self.bias1=self.bias1-0.1*s1
    def predict(self,x):
        result=self.forward(x)
        return result
        
if __name__=='__main__':
    #x,y=SinData()
    x,y=data()
    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1)
    #plt.plot(x,y,color='r')
    for i in range(len(x)):
        for j in range(len(x[i])):
            if y[i][0]==0:
                plt.scatter(x[i][0],x[i][1],color='g')
            else:
                plt.scatter(x[i][0],x[i][1],color='r')
    plt.title('Input Data')
    myNN=NN(2,4,2)
    rloss=[]
    racc=[]
    #print(myNN.weight1,myNN.weight2,myNN.bias1,myNN.bias2)
    for j in range(100):
        loss=0
        acc=0
        for i in range(len(x)):
            myNN.forward(x[i].reshape(-1,1))
            myNN.backpropagation(x[i].reshape(-1,1),y[i].reshape(-1,1))
            loss=myNN.loss(y[i].reshape(-1,1))
        rloss.append(loss)
        for m in range(len(x)):
            result=myNN.predict(x[m].reshape(-1,1))
            if result[0][0]>result[1][0] and y[m][0]>y[m][1]:
                acc+=1
            elif result[0][0]<result[1][0] and y[m][0]<y[m][1]:
                acc+=1
        racc.append(acc)
        print(j,'loss=',loss,'acc=',acc/len(x))
        if loss<=0.001:
            break
    #print(myNN.weight1,myNN.weight2,myNN.bias1,myNN.bias2)
    """print('weight1:\n',myNN.weight1,
          '\nweigh2:\n',myNN.weight2,
          '\nbias1\n',myNN.bias1,
          '\nbias2:\n',myNN.bias2)"""
    
    plt.subplot(1,4,2)
    plt.plot(range(len(rloss)),rloss,'r')
    plt.title('Loss')
    plt.subplot(1,4,3)
    #plt.plot(x,np.array(racc).reshape(-1),'g')
    plt.plot(range(len(racc)),racc,'g')
    plt.title('Acc')
    plt.subplot(1,4,4)
    for i in np.arange(-1,1,0.025):
        for j in np.arange(-1,1,0.025):
            label=np.zeros(2)
            if i>j:
                label=np.array([0,1])
            elif i<j:
                label=np.array([1,0])
            result=myNN.predict(np.array([i,j]).reshape(-1,1))
            if result[0][0]>result[1][0]:
                plt.scatter(i,j,color='r')
            elif result[0][0]<result[1][0]:
                plt.scatter(i,j,color='g')
    plt.title('Boundary')
    plt.show()