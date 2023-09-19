from PyQt5.Qt import QWidget, QColor, QPixmap, QIcon, QSize, QCheckBox
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPainter, QPen, QImage, QBrush, QPalette, QColor, QFont
import sys
from PyQt5.QtCore import Qt, QPoint
import tensorflow as tf
import numpy as np

W = 560 #画板宽度
Edge = 100 #边距
Px = 40 #画笔粗细
class PaintBoard(QWidget):


    def __init__(self, Parent=None):
        '''
        Constructor
        '''
        super().__init__(Parent)

        self.__InitData() #先初始化数据，再初始化界面
        self.__InitView()
        
    def __InitData(self):
        
        self.__size = QSize(W,W)
        
        #新建QPixmap作为画板，尺寸为__size
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.black) #用黑色填充画板
        
        self.__IsEmpty = True #默认为空画板 
        self.EraserMode = False #默认为禁用橡皮擦模式
        
        self.__lastPos = QPoint(0,0)#上一次鼠标位置
        self.__currentPos = QPoint(0,0)#当前的鼠标位置
        
        self.__painter = QPainter()#新建绘图工具
        
        self.__thickness = Px       #默认画笔粗细为50px
        self.__penColor = QColor("white")#设置默认画笔颜色为黑色
        self.__colorList = QColor.colorNames() #获取颜色列表
     
    def __InitView(self):
        #设置界面的尺寸为__size
        self.setFixedSize(self.__size)
        
    def Clear(self):
        #清空画板
        self.__board.fill(Qt.black)
        self.update()
        self.__IsEmpty = True
        
    def ChangePenColor(self, color="black"):
        #改变画笔颜色
        self.__penColor = QColor(color)
        
    def ChangePenThickness(self, thickness=Px):
        #改变画笔粗细
        self.__thickness = thickness
        
    def IsEmpty(self):
        #返回画板是否为空
        return self.__IsEmpty
    
    def GetContentAsQImage(self):
        #获取画板内容（返回QImage）
        return self.__board.toImage()
        
    def paintEvent(self, paintEvent):
        #绘图事件
        #绘图时必须使用QPainter的实例，此处为__painter
        #绘图在begin()函数与end()函数间进行
        #begin(param)的参数要指定绘图设备，即把图画在哪里
        #drawPixmap用于绘制QPixmap类型的对象
        self.__painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter.drawPixmap(0,0,self.__board)
        self.__painter.end()
        
    def mousePressEvent(self, mouseEvent):
        #鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos =  mouseEvent.pos()
        self.__lastPos = self.__currentPos
        
        
    def mouseMoveEvent(self, mouseEvent):
        #鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos =  mouseEvent.pos()
        self.__painter.begin(self.__board)
        
        if self.EraserMode == False:
            #非橡皮擦模式
            self.__painter.setPen(QPen(self.__penColor,self.__thickness)) #设置画笔颜色，粗细
        else:
            #橡皮擦模式下画笔为纯白色，粗细为10
            self.__painter.setPen(QPen(Qt.black,10))
            
        #画线    
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos
                
        self.update() #更新显示
        
    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False #画板不再为空
 
# 主界面 支持手写数字识别 和 识别结果显示       
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('手写数字识别')
        self.setFixedSize(W + Edge, W + Edge)
        self.__InitData()
        self.__InitView()
    
    def __InitData(self):
        self.__paintBoard = PaintBoard(self)
        
        self.__hintLabel = QLabel(self)
        self.__hintLabel.move(W, Edge)
        self.__hintLabel.setFont(QFont("Roman times", 20, QFont.Bold))
        self.__hintLabel.setText("点击\n空白\n重新\n绘制\n\n\n\n按键\n识别")
        
        self.__resultLabel = QLabel(self)
        self.__resultLabel.move(0, W + Edge//2)
        self.__resultLabel.setStyleSheet("QLabel{background:white;}")
        self.__resultLabel.setFont(QFont("Roman times", 20, QFont.Bold))
        self.__resultLabel.setAutoFillBackground(True)
        self.__resultLabel.setAlignment(Qt.AlignCenter)
        self.__resultLabel.setText("识别结果为：")
        self.__resultLabel.hide()
        
        self.__net = tf.keras.models.load_model('model.h5')
    
    def __InitView(self):
        self.__paintBoard.move(0, 0)
        self.__paintBoard.setStyleSheet("QWidget{background:white;}")
        self.__paintBoard.setPalette(QPalette(Qt.white))
        self.__paintBoard.setAutoFillBackground(True)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.__resultLabel.setText("识别结果为：")
            self.__resultLabel.hide()
            self.__paintBoard.Clear()       
            
    def keyPressEvent(self, event):
        image = self.__paintBoard.GetContentAsQImage()
        image.save("pictrue.png")
            
        data = tf.keras.preprocessing.image.load_img("pictrue.png",target_size=(28,28))
        data = data.convert('L')
        data = tf.keras.preprocessing.image.img_to_array(data)
        data = data.reshape(-1, 28, 28, 1)
        data = data / 255.0
        result = self.__net.predict(data)
        result = np.argmax(result)
        print(result)
        self.__resultLabel.setText("识别结果为："+str(result))
        self.__resultLabel.show()
        
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
