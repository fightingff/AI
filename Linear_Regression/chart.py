import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Draw:
    def __init__(self, records,corrcetness):
        self.Jw = records
        self.cs = corrcetness
        self.fig = plt.figure(figsize=(10, 5))  # 创建图
        plt.xlabel("Count")  # X轴标签
        plt.ylabel("J(w) / Correctness")  # Y轴标签
        self.x, self.y, self.y_= [], [] , []  # 用于保存绘图数据

    def update(self,n):  # 更新函数
        self.x.append(n)  # 添加x坐标值
        if self.Jw[n] < 1e-5:
            self.y.append(self.Jw[n]) 
        else:
            self.y.append(1e-5)
        self.y_.append(self.cs[n]*1e-5)
        plt.plot(self.x,self.y,self.y_, color='b')  # 绘制图形

    def Show(self,count):
        ani = FuncAnimation(self.fig, self.update, interval=5,frames=count,blit=False, repeat=False)  # 创建动画效果
        plt.show()  # 显示图片