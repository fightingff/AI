# use neural network to fit a function
import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2 as cv
from sklearn.utils import shuffle
import tensorflow as tf
from scipy.special import expit

# Data
Datas_N = 0
datas,labels = [],[]
train_data,train_label = [],[]
test_data,test_label = [],[]
Num = 0


class Sigmoid:
    def activate(x):
        return expit(x)
    
    def differentiate(x):
        return x*(1-x) 
    
    
class Relu:
    def activate(x):
        return np.maximum(0,x)
    
    def differentiate(x):
        return np.where(x>0,1.0,0.0)

class Softmax:
    def activate(x):
        # print(x)
        # print(x - np.max(x,axis=0,keepdims=True))
        # input()
        v = np.exp(x - np.max(x,axis=0,keepdims=True))
        return v / np.sum(v,axis=0,keepdims=True)
    
    def differentiate(x):
        pass

class Conv2D:
    filter_size = 3
    filter = np.random.randn(filter_size,filter_size)
    padding = 1
    stride = 1
    
    def __init__(self,filter):
        filter = np.array(filter)
        self.filter = filter
        self.filter_size = filter.shape[0]
    
    def Conv(self,img):
        img = np.array(img)
        img = np.pad(img,((self.padding,self.padding),(self.padding,self.padding)),'constant')
        img_size = img.shape[0]
        output_size = int((img_size-self.filter_size)/self.stride)+1
        output = np.zeros((output_size,output_size))
        for i in range(output_size):
            for j in range(output_size):
                output[i,j] = np.sum(img[i:i+self.filter_size,j:j+self.filter_size]*self.filter)
        return Relu.activate(output)
    
    def Pooling_Max(self,data,size=2):
        N = data.shape[0]
        output_size = int(N/size)
        output = np.zeros((output_size,output_size))
        for i in range(output_size):
            for j in range(output_size):
                output[i,j] = np.max(data[i*size:(i+1)*size,j*size:(j+1)*size])
        return output
    
    def Pooling_Avg(self,data,size=2):
        N = data.shape[0]
        output_size = int(N/size)
        output = np.zeros((output_size,output_size))
        for i in range(output_size):
            for j in range(output_size):
                output[i,j] = np.mean(data[i*size:(i+1)*size,j*size:(j+1)*size])
        return output
    
    def Result(self,image):
        result = self.Pooling_Max(self.Conv(image))
        return result / 255.0

class Layer:
    N,M = 0,0 
    W,B = [],[]
    dW,dB = [],[]
    Xi,Xo = [],[]
    Activation = Sigmoid
    # N: number of neurons, M: number of features
    # W (N,M) , B (N,)
    # input Xi(M,K) -> output Xo(N,K)
    
    def __init__(self,N,M,func=Sigmoid):
        # initialize W and B with random values
        self.N,self.M = N,M
        self.W = np.random.normal(size=(N,M))
        self.B = np.random.normal(size=(N,1))
        self.Activation = func
        self.dW = np.zeros((N,M))
        self.dB = np.zeros(N)
        self.Xi = np.zeros(M)
        self.Yi = np.zeros(N)
    
    def forward(self,X):
        # perform forward propagation
        self.Xi = X
        self.Xo = self.Activation.activate(self.W @ X + self.B)
        return self.Xo
    
    
class NeuralNetwork:
    Depth = 0
    Layers = []
    def __init__(self):
        # initialize layers
        self.Depth = 0
        self.Layers = []
    
    def AddLayer(self,layer):
        # add a layer to the network
        self.Layers.append(layer)    
        self.Depth += 1
    
    def Forward(self,X):
        # perform forward propagation
        X_ = copy.deepcopy(X)
        for layer in self.Layers:
            X_ = layer.forward(X_)
        
        return X_
    
    def Predict(self,X):
        # predict the output of X
        v = self.Forward(X)
        # print(v)
        # print(np.max(v,axis=0,keepdims=True))
        # print(np.where(v == np.max(v,axis=0,keepdims=True),1.0,0.0))
        # input()
        return np.where(v == np.max(v,axis=0,keepdims=True),1.0,0.0)
    
    def Loss_MSE(self,X,Y):
        # calculate the loss of the network
        return 0.5*np.sum((self.Predict(X) - Y)**2)
    
    def Loss_CrossEntropy(self,X,Y):
        # calculate the loss of the network
        return -np.sum(Y*np.log(self.Forward(X)))
    
    def Accuracy(self,X,Y):
        # calculate the accuracy of the network
        return np.sum(self.Predict(X) * Y) / Y.shape[1]
    
    def Backward(self,i,dX,alpha):
        # perform backward propagation
        # i: index of the layer
        if i < 0:
            # print(dX)
            return
        layer = self.Layers[i]
        D = dX * np.sum(layer.Activation.differentiate(layer.Xo),axis=1,keepdims=True)
        dW = D * np.sum(layer.Xi,axis=1,keepdims=True).T
        dB = D
        dX = layer.W.T @ D
        
        self.Layers[i].W -= alpha * dW
        self.Layers[i].B -= alpha * dB
        
        self.Backward(i-1,dX,alpha)
        
    def Train(self,X,Y,alpha=0.01):
        # train the network
        # alpha is the learning rate
        Y_ = self.Forward(X)
        
        # print(Y_)
        # input()
        Output = self.Layers[-1]
        D =  - Y / Y_  # d(L) / d(Y_)
        dW = D * np.sum(Output.Xi,axis=1,keepdims=True).T 
        dB = D
        dX = (Y_ - Y) * np.sum(Output.W,axis=1,keepdims=True)
            
        Output.W -= alpha * dW
        Output.B -= alpha * dB
            
        self.Backward(self.Depth - 2,dX,alpha)
        
    def Save(self,File):
        with open(File,"w") as f:
            f.write(str(self.Depth)+"\n")
            for layer in self.Layers:
                f.write(str(layer.N)+" "+str(layer.M)+"\n")
                for i in range(layer.N):
                    for j in range(layer.M):
                        f.write(str(layer.W[i,j])+" ")
                    f.write("\n")
                for i in range(layer.N):
                    f.write(str(layer.B[i,0])+" ")
                f.write("\n")
                
    def Load(self,File):
        with open(File,"r") as f:
            self.Depth = int(f.readline())
            for i in range(self.Depth):
                N,M = map(int,f.readline().split())
                layer = Layer(N,M)
                for i in range(N):
                    layer.W[i,:] = np.array(list(map(float,f.readline().split())))
                layer.B = np.array(list(map(float,f.readline().split()))).reshape(-1,1)
                self.Layers.append(layer)
        self.Layers[0].Activation = Relu
        self.Layers[1].Activation = Relu
        self.Layers[-1].Activation = Softmax
        
    
def Show(model,BestModel,X,Y):
    plt.plot(X,Y,'go')
    xi = np.linspace(min(X),max(X),100).reshape(1,-1)
    xi.sort()
    y1 = model.Predict(xi)
    y2 = BestModel.Predict(xi)
    xi = xi.reshape(-1).tolist()
    y1 = y1.reshape(-1).tolist()
    y2 = y2.reshape(-1).tolist()
    plt.plot(xi,y1,'b-')
    plt.plot(xi,y2,'r-')
    plt.show()

def Training(Epoch=10,iteration=1000,alpha=0.01):
    global datas,labels,train_data,train_label,test_data,test_label,Num
    model = NeuralNetwork()
    model.AddLayer(Layer(128,datas[0].shape[0],Relu))
    model.AddLayer(Layer(64,128,Relu))
    model.AddLayer(Layer(10,64,Sigmoid))
    model.AddLayer(Layer(10,10,Softmax))
    
    # model.Train(X,Y)
    Loss = []
    Best = 0
    BestModel = NeuralNetwork()
    Num = int(Datas_N / Epoch)
    for i in range(Epoch):
        alpha *= 0.99
        train_data,train_label = datas[Num * i:Num * (i+1)],labels[Num * i:Num * (i+1)]
        test_data,test_label = train_data[:Num // 10],train_label[:Num // 10]
        train_data,train_label = train_data[Num // 10:],train_label[Num // 10:]
        
        N = test_data[0].shape[0]
        train_data = np.array(train_data).reshape(-1,N).T
        train_label = np.array(train_label).reshape(-1,10).T
        test_data = np.array(test_data).reshape(-1,N).T
        test_label = np.array(test_label).reshape(-1,10).T
        
        print("Epoch %d"%i)
        # print(train_data)
        # print(train_data.shape)
        # print(train_label.shape)
        for case in range(iteration):
            for k in range (train_data.shape[1]):
                model.Train(train_data[:,k].reshape(-1,1),train_label[:,k].reshape(-1,1),alpha)
            
            g = model.Loss_CrossEntropy(test_data,test_label)
            f = model.Accuracy(test_data,test_label)
            if f > Best:
                Best = f
                BestModel = copy.deepcopy(model)
            Loss.append(g)
    
        print("Best Accuracy = %f%%"%(Best*100))
        BestModel.Save("model_cross_down.net")
    plt.plot(Loss)
    plt.show()
    
    

def Load_digit():
    print("Loading")
    global train_data,train_label,test_data,test_label,datas,labels,Datas_N
    minist = tf.keras.datasets.mnist
    (train_data, train_label), (test_data, test_label) = minist.load_data()
    train_data = train_data.reshape(-1,28,28)
    test_data = test_data.reshape(-1,28,28)
    Conv = Conv2D([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    Conv1 = Conv2D([[0,-1,0],[-1,5,-1],[0,-1,0]])
    Conv2 = Conv2D([[0,-1,0],[-1,4,-1],[0,-1,0]])
    
    # for i in range(700):
    #     img = train_data[i,:,:].reshape(28,28)
    #     datas.append(np.array([Conv.Result(Conv.Result(img))]).reshape(-1,1))
    #     labels.append(tf.keras.utils.to_categorical(train_label[i],10).reshape(1,10))
    for i in range(train_data.shape[0]):
        img = train_data[i,:,:].reshape(28,28)
        datas.append(np.array([Conv.Result(Conv.Result(img)),Conv1.Result(Conv1.Result(img)),Conv2.Result(Conv2.Result(img))]).reshape(-1,1))
        labels.append(tf.keras.utils.to_categorical(train_label[i],10).reshape(1,10))
    for i in range(test_data.shape[0]):
        img = test_data[i,:,:].reshape(28,28)
        datas.append(np.array([Conv.Result(Conv.Result(img)),Conv1.Result(Conv1.Result(img)),Conv2.Result(Conv2.Result(img))]).reshape(-1,1))
        labels.append(tf.keras.utils.to_categorical(test_label[i],10).reshape(1,10))
        
    Datas_N = len(datas)
    datas,labels = shuffle(datas,labels)
    # print(labels)
    
if __name__ == "__main__":
    Load_digit()
    Training(Epoch=100,iteration=1000,alpha=1e-5)