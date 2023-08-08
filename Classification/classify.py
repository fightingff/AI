from anyio import sleep
import numpy as np
import re
import matplotlib.pyplot as plt

# logistic regression

alpha = 0.1 # learning rate
ld = 1 # regularization parameter
eps = 1e-3 # threshold
M = 0
K = 0
datas = []
Vs = []
test_M = 0
test = []
W = []
memory = []

def Correctness():  # use different accuracy to show correctness
    cnt_true = 0
    cnt_false = 0
    True_cnt = 0
    for i in range(test_M):
        f = 1 if W @ test[i].T >= 0 else 0
        if f == test[i].item(K):
            if f==1:
                cnt_true += 1
            else: 
                cnt_false += 1
        if(test[i].item(K)==1):
            True_cnt += 1
    return ((cnt_true+cnt_false)/test_M,cnt_true/True_cnt,cnt_false/(test_M-True_cnt))

def J():
    # define the J(w)
    sum = 0
    for i in range(test_M):
        y = datas[i].item(K)
        g = 1 / (1 + np.exp(-W @ datas[i].T))   # use the sigmod function
        sum += y * np.log(g) + (1-y) * np.log(1-g) 
    return float(-sum / M)

def Draw(delay):
    # visualize (only for 2D)
    X,Y,C = [],[],[]
    for i in range(len(memory)):
        X.append(memory[i].item(1))
        Y.append(memory[i].item(2))
        C.append('r' if memory[i].item(K)==1 else 'b')
    plt.scatter(X,Y,c=C)
    
    # draw the line
    X = np.linspace(-10,10,100)
    Y = []
    for x in X:
        Y.append((-W.item(0)-W.item(1)*x)/W.item(2))
    plt.plot(X,Y)
    plt.draw()
    plt.pause(delay)
    plt.close()
    
with open("data.txt", 'r') as f:
    data = f.read()
    numbers = re.findall(r"-?\d+\.?\d*",data)
    M = int(numbers[0])
    K = int(numbers[1]) + 1
    datas = []
    for i in range(M):
        item = [float(x) for x in numbers[2+i*K:2+(i+1)*K]]
        item.insert(0,1)    # stands for B to simplify the calculation
        item = np.array(item)
        datas.append(item)

    # use z-score standardzation
    datas = np.matrix(datas)
    memory = datas
    W = np.matrix(np.zeros(K+1))

    test_M = M//3 + 1
    M -= test_M
    test = datas[M:]
    datas = datas[:M]
    
    cnt = 0
    while J() > eps and min(Correctness()) < 0.95:
        lst = W.copy()
        for i in range(M):  # update the W
            W -= alpha / M * (1 / (1 + np.exp(-lst @ datas[i].T)) - datas[i].item(K)) * datas[i]
        W.itemset(K,0)
        W -= alpha * ld / M * lst  #regularized logistic regression
        
        cnt += 1
        print(cnt,"times correctness: ",Correctness())
        print("J(W): ",J())
        print()
        Draw(0.1)
        
        # Draw()
    Draw(10)
    W = W.tolist()[0]
    for i in range(K):
        W[i] /= -W[K-1]         # make it easier to check the answer
    print("W: ",W[:K])