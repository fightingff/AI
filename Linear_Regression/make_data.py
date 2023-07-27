import random
def R(L,R):
    return random.random()*(R-L)+L
P = 5

with open("data.txt","w") as f:
    M = 1000
    N = 20
    f.write(str(M)+" "+str(N)+"\n")
    W = []
    B = R(-P,P)
    for i in range(N):
        W.append(R(-P,P))
    for i in range(M):
        y = B
        for j in range(N):
            x = R(-P,P)
            y += W[j] * x
            f.write(str(x)+" ")
        f.write(str(y)+"\n")

        