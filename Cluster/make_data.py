import random
import math

with open("data.txt","w") as f:
    N = 1000
    D = 2
    f.write(str(N)+" "+str(D)+"\n")
    centers = [random.random()*1000 for i in range(6)]
    
    for i in range(N):
        k = random.randint(0,2)
        r = random.random()*100
        theta = random.random()*2*3.1415926
        x = centers[k*2] + r * math.cos(theta)
        y = centers[k*2+1] + r * math.sin(theta)
        f.write(str(x)+" "+str(y)+"\n")
        