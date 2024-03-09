# use neural network to fit a function
import numpy as np
import matplotlib.pyplot as plt
import copy
import random

def Sigmoid(x):
    return 1/(1+np.exp(-x))

class Layer:
    N,M = 0,0 
    W,B = [],[]
    dW,dB = [],[]
    Xi,Xo = [],[]
    # N: number of neurons, M: number of features
    # W (N,M) , B (N,)
    # input Xi(M,K) -> output Xo(N,K)
    
    def __init__(self,N,M):
        # initialize W and B with random values
        self.N,self.M = N,M
        self.W = np.random.normal(size=(N,M))
        self.B = np.random.normal(size=(N,1))
        print(self.W)
        self.dW = np.zeros((N,M))
        self.dB = np.zeros(N)
        self.Xi = np.zeros(M)
        self.Yi = np.zeros(N)
    
    def forward(self,X):
        # perform forward propagation
        self.Xi = X
        self.Xo = Sigmoid(self.W @ X + self.B)
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
        Y = self.Forward(X)
        return -np.log(1/Y - 1)
    
    def Loss(self,X,Y):
        # calculate the loss of the network
        return 0.5*np.sum((self.Predict(X)-Y)**2)
    
    def Backward(self,i,dX,alpha):
        # perform backward propagation
        # i: index of the layer
        if i < 0:
            return
        
        D = dX * np.sum(self.Layers[i].Xo * (1 - self.Layers[i].Xo),axis=1,keepdims=True)
        dW = D * np.sum(self.Layers[i].Xi,axis=1,keepdims=True).T
        dB = D
        dX = self.Layers[i].W.T @ D
        
        self.Layers[i].W -= alpha * dW
        self.Layers[i].B -= alpha * dB
        
        self.Backward(i-1,dX,alpha)
        
    def Train(self,X,Y,alpha=0.01):
        # train the network
        # alpha is the learning rate
        Y_ = self.Predict(X)
        Output = self.Layers[-1]
            
        D = np.sum(Y_ - Y,axis=1,keepdims=True) 
        dW = D * np.sum(Output.Xi,axis=1,keepdims=True).T
        dB = D
        dX = D * Output.W.T
            
        Output.W -= alpha * dW
        Output.B -= alpha * dB
            
        self.Backward(self.Depth - 2,dX,alpha)
        
    
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
    
def Main(Epoch=1000,Sample=1,alpha=0.1,Maxh=10):
    K = 0
    X,Y = [],[]
    with open("data.in","r") as f:
        K = int(f.readline())
        for i in range(K):
            x,y = f.readline().split()
            X.append(float(x))
            Y.append(float(y))
    X = np.array(X).reshape(1,K)
    Y = np.array(Y).reshape(1,K)
    
    model = NeuralNetwork()
    Pow = 1 << Maxh
    model.AddLayer(Layer(Pow,1))
    for i in range(Maxh):
        model.AddLayer(Layer(Pow>>1,Pow))
        Pow >>= 1
    
    
    # model.Train(X,Y)
    Loss = []
    Best = 1e9
    BestModel = None
    for i in range(Epoch):
        idx = random.sample(list(range(K)),Sample)
        X_ = X[:,idx]
        Y_ = Y[:,idx]
        model.Train(X_,Y_,alpha)
        g = model.Loss(X,Y)
        print("Epoch %d: Loss = %f"%(i,g))
        if g < Best:
            Best = g
            BestModel = copy.deepcopy(model)
        Loss.append(g)
    
    plt.plot(Loss)
    plt.show()
    
    Show(model,BestModel,X,Y)
    
if __name__ == "__main__":
    Main(Epoch=2000,Sample=1,alpha=0.01,Maxh=12)