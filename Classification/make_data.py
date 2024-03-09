import random
def R(L,R):
    return random.random()*(R-L)+L
P = 10
W = []
with open("data.txt","w") as f:
    M = 100
    N = 2
    f.write(str(M)+" "+str(N)+"\n")
    B = R(-P,P)
    for i in range(N):
        W.append(R(-P,P))
    for i in range(M):
        y = B
        for j in range(N):
            x = R(-P,P)
            y += W[j] * x
            f.write(str(x)+" ")
        f.write(str(1 if y >=0 else 0)+"\n")
        
with open("Ans.txt","w") as f:
    W.insert(0,B)
    for i in range(N+1):
        W[i] /= -W[2]
    f.write(str(W))
