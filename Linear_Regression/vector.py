import re
import numpy as np
from chart import Draw

# problem: too easy to overflow when calculating J(w) and D(w,k)

alpha = 0.1 # learning rate
eps = 1e-10 # threshold
M = 0
K = 0
datas = []
Vs = []
test_M = 0
test = []
Jw = []
Cs = []

# J(w) estimates the error of w
def J(w):
    sum = 0
    for i in range(test_M):
        Dt = w @ test[i].T
        sum += float(Dt ** 2)
    return sum/(2*M)

# show correctness of w on test data
def correctness(w):
    cnt = 0
    for i in range(test_M):
        Dt = w @ test[i].T
        if Dt ** 2 <= eps:
            cnt += 1
    return cnt / test_M 
        
def Standardzation():
    # use z-score standardzation
    
    for i in range(1,K+1):
        mean = 0
        for j in range(M):
            mean += datas.item((j,i))
        mean /= M
        
        std = 0
        for j in range(M):
            std += abs(datas.item((j,i)) - mean)
            
        std /= M
        for j in range(M):
            datas.itemset((j,i),(datas.item((j,i))-mean)/std)

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
    Standardzation()
    
    # split datas into train and test
    test_M = M//5 + 1
    M -= test_M
    test = datas[M:]
    datas = datas[:M]
    
    # get Vs
    for i in range(K):
        cols = []
        for j in range(M):
            cols.append(datas.item((j,i)))
        Vs.append(cols)
    Vs.append([0.0 for i in range(M)])
    Vs = np.matrix(Vs)
    
    # initialize W with 0
    W = [0.0 for i in range(K)]
    W.append(-1.0)  # simplify the calculation
    W = np.matrix(W)
    
    # gradient descent
    cnt = 0
    while correctness(W) < 0.9:
        lst = W.copy()
        W -= alpha / M * (lst @ datas.T @ Vs.T)
        cnt += 1
        Jw.append(J(W))
        Cs.append(correctness(W))
        print(cnt,"times correctness: ",correctness(W))
    
    # Ans
    W = W.tolist()[0]
    print("W=",W[:K])
    
    Jw.append(J(W))
    Cs.append(correctness(W))
    Painter = Draw(records=Jw,corrcetness=Cs)
    Painter.Show(cnt)