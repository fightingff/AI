import numpy as np
N = 100
L,R = 1,10
with open("data.in","w") as f:
    f.write(str(N)+"\n")
    xi = np.linspace(L,R,N)
    yi = np.sin(xi)*np.log(xi)*np.cos(xi)
    # yi = xi ** 2
    for i in range(N):
        f.writelines("%f %f\n"%(xi[i],yi[i]))