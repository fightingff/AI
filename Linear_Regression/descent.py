import re

# problem: too easy to overflow when calculating J(w) and D(w,k)

alpha = 0.1
eps = 1e-10 # threshold
M = 0
K = 0
datas = []
test_M = 0
test = []

# J(w) estimates the error of w
def J(w):
    sum = 0
    for i in range(M):
        Dt = 0
        for _i,x in enumerate(datas[i]):
            Dt += w[_i] * x
        sum += Dt ** 2
    return sum/(2*M)

# D(w,k) estimates the partial derivative of J(w) with respect to w[k]
def D(w,k):
    sum = 0
    for i in range(M):
        Fd = 0
        for _i,x in enumerate(datas[i]):
            Fd += w[_i] * x
        sum += Fd * datas[i][k]
    return sum/M

# show correctness of w on test data
def correctness(w):
    cnt = 0
    for i in range(test_M):
        Dt = 0
        for _i,x in enumerate(test[i]):
            Dt += w[_i] * x
        if Dt ** 2 <= eps:
            cnt += 1
    return cnt/test_M*100
        
with open("data.txt", 'r') as f:
    data = f.read()
    numbers = re.findall(r"-?\d+\.?\d*",data)
    M = int(numbers[0])
    K = int(numbers[1])
    datas = []
    for i in range(M):
        item = [float(x) for x in numbers[2+i*(K+1):2+(i+1)*(K+1)]]
        item.insert(0,1)    # stands for B to simplify the calculation
        datas.append(item)

    # split datas into train and test
    test = datas[M-M//5:]
    test_M = M//5
    M = M - test_M
    datas = datas[:M]
    
    # initialize W with 0
    W = [0.0 for i in range(K+1)]
    W.append(-1.0)  # simplify the calculation
    lst = W.copy()
    lst[0] = 1
    
    cnt = 0
    while correctness(W) < 90:
        lst = W.copy()
        for i in range(K+1):
            W[i] -= alpha * D(lst,i)    # gradient descent
        cnt += 1
        print(cnt,"times correctness: ",correctness(W))
    
    # Ans
    print("W = ",W)
    with open ("data.out","w") as fo:
        fo.write(str(W[0])+" ")
        fo.write(str(W[1]))
    
    