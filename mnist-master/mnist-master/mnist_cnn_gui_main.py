#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import glob
import sys, os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image, ImageQt


from qt.test import Ui_MainWindow # 通过qt自动生成的界面
from qt.paintboard import PaintBoard

from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtWidgets import QLabel, QMessageBox, QPushButton, QFrame
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtWidgets import QFileDialog

import cv2
from simple_convnet import SimpleConvNet
from common.functions import softmax
from deep_convnet import DeepConvNet



MODE_MNIST = 1    # MNIST随机抽取
MODE_WRITE = 2    # 手写输入
MODE_PICTURE = 3   # 手写图片识别
Thresh = 0.5      # 识别结果置信度阈值



# 读取MNIST数据集
(_, _), (x_test, _) = load_mnist(normalize=True, flatten=False, one_hot_label=False)


# 初始化网络

# 网络1：简单CNN
"""
conv - relu - pool - affine - relu - affine - softmax
"""
# 建立一个卷积神经网络的类。
# 本人对于卷积神经网络还不是很了解，不过其实现思路和我前面写的my_app_cnn中的卷积神经网络的步骤相同。
# 关于卷积神经网络将在后面学习，但是这次没有使用tf
network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params("mnist-master\mnist-master\params.pkl")

# 网络2：深度CNN
# network = DeepConvNet()
# network.load_params("deep_convnet_params.pkl")


class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        # 初始化继承的父类(Qmainwindow)
        super(MainWindow,self).__init__()
    
        # 初始化参数
        self.mode = MODE_MNIST
        self.result = [0, 0]

        # 初始化UI
        self.setupUi(self)
        self.center()
     
        # 初始化画板，可以用鼠标来输入数据的
        # 将画板变为gui画面的一部分。
        self.paintBoard = PaintBoard(self, Size = QSize(224, 224), Fill = QColor(0,0,0,0))
        self.paintBoard.setPenColor(QColor(0,0,0,0))
        self.dArea_Layout.addWidget(self.paintBoard) # gui界面中加入paintboard这个部分
        self.imgName = ''
        self.clearDataArea() # 清除数据区域
        png = QtGui.QPixmap('mnist-master\mnist-master\hj.jpg').scaled(160, 150)
        self.lbDataArea1.setPixmap(png)
        png1 = QtGui.QPixmap('mnist-master\mnist-master\hk.jpg').scaled(160, 115)
        self.lbDataArea2.setPixmap(png1)
        
    # 窗口居中，这次设置的是那个整体的窗口
    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心,将窗口设置在屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())
    
    
    # 窗口关闭事件，是否关闭窗口，点击右上角关闭窗口
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
            "Are you sure to quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()   
    
    # 清除数据待输入区
    def clearDataArea(self):
        self.paintBoard.Clear()#清除画板
        self.lbDataArea.clear()
        self.lbResult.clear()   # 确认的数字
        self.lbCofidence.clear()    # softmax的值
        self.result = [0, 0]    # 一个用来存确认数字，一个用来存softmax值

    """
    回调函数
    """
    # 模式下拉列表回调，两种模式
    def cbBox_Mode_Callback(self, text):
        if text == '1：MINIST随机抽取':
            print(text)
            self.mode = MODE_MNIST
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(True)
            self.pbtjiazai.setEnabled(False)
            self.paintBoard.setBoardFill(QColor(0,0,0,0))
            self.paintBoard.setPenColor(QColor(0,0,0,0))

        elif text == '2：鼠标手写输入':
            #print(text)
            self.mode = MODE_WRITE
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)
            self.pbtjiazai.setEnabled(False)
            # 更改背景
            self.paintBoard.setBoardFill(QColor(0,0,0,255))
            self.paintBoard.setPenColor(QColor(255,255,255,255))

        elif text == '3：手写图片识别':
            #print('111')
            self.mode = MODE_PICTURE
            self.clearDataArea()
            self.pbtGetMnist.setEnabled(False)
            self.pbtjiazai.setEnabled(True)
            # 更改背景
            self.paintBoard.setBoardFill(QColor(0,0,0,0))
            self.paintBoard.setPenColor(QColor(0,0,0,0))



#    def get_picture(self):
        



    # 数据清除，点击数据清除区域
    def pbtClear_Callback(self):
        self.clearDataArea()
 

    # 识别
    def pbtPredict_Callback(self):
        __img, img_array =[],[]      # 将图像统一从qimage->pil image -> np.array [1, 1, 28, 28]
        
        # 获取qimage格式图像
        if self.mode == MODE_MNIST: #随机抽取
            __img = self.lbDataArea.pixmap()  # label内若无图像返回None
            if __img == None:   # 无图像则用纯黑代替
                # __img = QImage(224, 224, QImage.Format_Grayscale8)
                __img = ImageQt.ImageQt(Image.fromarray(np.uint8(np.zeros([224,224]))))
            else: __img = __img.toImage()
            pil_img = ImageQt.fromqimage(__img)
            pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
            img_array = np.array(pil_img.convert('L')).reshape(1,1,28, 28) / 255.0
            
        elif self.mode == MODE_WRITE:
            __img = self.paintBoard.getContentAsQImage()
            pil_img = ImageQt.fromqimage(__img)
            pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
            pil_img.save('test.png')
            img_array = np.array(pil_img.convert('L')).reshape(1,1,28, 28) / 255.0
            
        elif self.mode == MODE_PICTURE:
           
            pic = image_prepare(self.imgName)   # 得到28*28的数组
            img_array = pic.reshape(1,1,28, 28) / 255.0 # 转换成神经网络要求的输入格式
            '''
                关于这部分有一点需要说明：
                    开始通过image_prepare得到预处理的图片之后，最开始是变成一维数组，然后一直想着变成二维数组并进行灰度反转。
                    为什么会这样想呢，因为我开始在变成reshape(1,1,28, 28)格式时采用的是网上的另一种方法，以至于格式基本是没有变化的。
                    其实直接变成一维数组然后进行上面的转换也是一样可以的，这个部分的失误浪费了很长时间！以后debug时需要多注意。
            '''
        # reshape成网络输入类型 
        __result = network.predict(img_array)      # shape:[1, 10]

        # print (__result)

        # 将预测结果使用softmax输出，得到输出结果
        __result = softmax(__result)
       
        self.result[0] = np.argmax(__result)          # 预测的数字
        self.result[1] = __result[0, self.result[0]]     # 置信度

        # 结果显示在界面上
        self.lbResult.setText("%d" % (self.result[0]))
        self.lbCofidence.setText("%.8f" % (self.result[1]))


    # 随机抽取
    def pbtGetMnist_Callback(self):
        self.clearDataArea()
        
        # 随机抽取一张测试集图片，放大后显示，x_text为前面读取的数据集
        # 随机从10000个照片中选择一张。
        img = x_test[np.random.randint(0, 9999)]    # shape:[1,28,28] 
        img = img.reshape(28, 28)                   # shape:[28,28]  

        img = img * 0xff      # 恢复灰度值大小 
        pil_img = Image.fromarray(np.uint8(img))
        pil_img = pil_img.resize((224, 224))        # 图像放大显示，和绘图窗口大小一致

        # 将pil图像转换成qimage类型
        qimage = ImageQt.ImageQt(pil_img)
        
        # 将qimage类型图像显示在label
        pix = QPixmap.fromImage(qimage)
        self.lbDataArea.setPixmap(pix)

    def pbtjiazai_Callback(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", " *.png;;*.jpg;;*.jpeg;;*.bmp;;All Files (*)")
        print(self.imgName) # imgName就是图片的路径，为了之后在识别的时候也可以用,写成了self.imgName,self代表创建的实例本身
        png = QtGui.QPixmap(self.imgName).scaled(240, 240)
        # 显示图片并设置图片格式
        self.lbDataArea.setPixmap(png)


# 不放在类中，当作一个函数来写
def image_prepare(pic_name): 
    # 读取图像，第二个参数是读取方式
    img = cv2.imread(pic_name, 1)
    # 使用全局阈值，降噪
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # 把opencv图像转化为PIL图像
    im = Image.fromarray(cv2.cvtColor(th1,cv2.COLOR_BGR2RGB))
    # 灰度化
    # im = im.convert('L')
    # 为图片重新指定尺寸
    im = im.resize((28,28), Image.ANTIALIAS)
    img_array = np.array(im.convert('L'))
    # plt.imshow(im)
    # plt.show()
    # 图像转换为list
    for i in range(28):
        for j in range(28):
            img_array[i][j] = (255-img_array[i][j])*1.0
    '''
    im_list = list(im.getdata())
    # 图像灰度反转
    result = [(255-x)*1.0/255.0 for x in im_list]
    '''
    return img_array



if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()

    sys.exit(app.exec_())