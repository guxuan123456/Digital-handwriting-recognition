# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'layout.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.cbBox_Mode = QtWidgets.QComboBox(MainWindow)
        self.cbBox_Mode.setGeometry(QtCore.QRect(60, 300, 211, 31))
        self.cbBox_Mode.setObjectName("cbBox_Mode")
        self.cbBox_Mode.addItem("")
        self.cbBox_Mode.addItem("")
        self.cbBox_Mode.addItem("")
        self.label = QtWidgets.QLabel(MainWindow)
        self.label.setGeometry(QtCore.QRect(60, 270, 80, 20))
        self.label.setObjectName("label")
        self.pbtClear = QtWidgets.QPushButton(MainWindow)
        self.pbtClear.setGeometry(QtCore.QRect(80, 440, 120, 30))
        self.pbtClear.setStyleSheet("")
        self.pbtClear.setCheckable(False)
        self.pbtClear.setChecked(False)
        self.pbtClear.setObjectName("pbtClear")
        self.pbtGetMnist = QtWidgets.QPushButton(MainWindow)
        self.pbtGetMnist.setGeometry(QtCore.QRect(80, 380, 120, 30))
        self.pbtGetMnist.setCheckable(False)
        self.pbtGetMnist.setObjectName("pbtGetMnist")
        self.pbtPredict = QtWidgets.QPushButton(MainWindow)
        self.pbtPredict.setGeometry(QtCore.QRect(80, 500, 120, 30))
        self.pbtPredict.setStyleSheet("")
        self.pbtPredict.setObjectName("pbtPredict")
        self.lbDataArea = QtWidgets.QLabel(MainWindow)
        self.lbDataArea.setGeometry(QtCore.QRect(540, 350, 224, 224))
        self.lbDataArea.setMouseTracking(False)
        self.lbDataArea.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lbDataArea.setFrameShape(QtWidgets.QFrame.Box)
        self.lbDataArea.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lbDataArea.setLineWidth(4)
        self.lbDataArea.setMidLineWidth(0)
        self.lbDataArea.setText("")
        self.lbDataArea.setObjectName("lbDataArea")

        self.lbDataArea1 = QtWidgets.QLabel(MainWindow)
        self.lbDataArea1.setGeometry(QtCore.QRect(570, 150, 160, 150))
        self.lbDataArea1.setMouseTracking(False)
        self.lbDataArea1.setFrameShape(QtWidgets.QFrame.Box)
        self.lbDataArea1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lbDataArea1.setLineWidth(1)
        self.lbDataArea1.setMidLineWidth(0)
        self.lbDataArea1.setText("")
        self.lbDataArea1.setObjectName("lbDataArea1")

        self.lbDataArea2 = QtWidgets.QLabel(MainWindow)
        self.lbDataArea2.setGeometry(QtCore.QRect(565, 15, 170, 120))
        self.lbDataArea2.setMouseTracking(False)
        self.lbDataArea2.setFrameShape(QtWidgets.QFrame.Box)
        self.lbDataArea2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lbDataArea2.setLineWidth(1)
        self.lbDataArea2.setMidLineWidth(0)
        self.lbDataArea2.setText("")
        self.lbDataArea2.setObjectName("lbDataArea2")

        self.label_3 = QtWidgets.QLabel(MainWindow)
        self.label_3.setGeometry(QtCore.QRect(260, 340, 91, 181))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(MainWindow)
        self.label_4.setGeometry(QtCore.QRect(540, 316, 90, 31))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(MainWindow)
        self.label_5.setGeometry(QtCore.QRect(20, 10, 471, 241))
        self.label_5.setObjectName("label_5")
        self.verticalLayoutWidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(540, 350, 221, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.dArea_Layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dArea_Layout.setContentsMargins(0, 0, 0, 0)
        self.dArea_Layout.setSpacing(0)
        self.dArea_Layout.setObjectName("dArea_Layout")
        self.lbResult = QtWidgets.QLabel(MainWindow)
        self.lbResult.setGeometry(QtCore.QRect(380, 350, 91, 131))
        font = QtGui.QFont()
        font.setPointSize(48)
        self.lbResult.setFont(font)
        self.lbResult.setObjectName("lbResult")
        self.lbCofidence = QtWidgets.QLabel(MainWindow)
        self.lbCofidence.setGeometry(QtCore.QRect(360, 500, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbCofidence.setFont(font)
        self.lbCofidence.setObjectName("lbCofidence")
        self.pbtjiazai = QtWidgets.QPushButton(MainWindow)
        self.pbtjiazai.setGeometry(QtCore.QRect(660, 317, 99, 31))
        self.pbtjiazai.setCheckable(False)
        self.pbtjiazai.setObjectName("pbtjiazai")

        self.retranslateUi(MainWindow)
        self.cbBox_Mode.activated['QString'].connect(MainWindow.cbBox_Mode_Callback)
        self.pbtClear.clicked.connect(MainWindow.pbtClear_Callback)
        self.pbtPredict.clicked.connect(MainWindow.pbtPredict_Callback)
        self.pbtGetMnist.clicked.connect(MainWindow.pbtGetMnist_Callback)
        self.pbtjiazai.clicked.connect(MainWindow.pbtjiazai_Callback)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "手写数字识别CNN-顾轩-2020.5.2"))
        self.cbBox_Mode.setItemText(0, _translate("MainWindow", "1：MINIST随机抽取"))
        self.cbBox_Mode.setItemText(1, _translate("MainWindow", "2：鼠标手写输入"))
        self.cbBox_Mode.setItemText(2, _translate("MainWindow", "3：手写图片识别"))
        self.label.setText(_translate("MainWindow", "模式选择"))
        self.pbtClear.setText(_translate("MainWindow", "清除数据"))
        self.pbtGetMnist.setText(_translate("MainWindow", "MNIST随机抽取"))
        self.pbtPredict.setText(_translate("MainWindow", "识别数字"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">识别结果：</span></p><p><br/></p><p><br/></p><p><span style=\" font-size:12pt; font-weight:600;\">Softmax：</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "数据输入区域"))
        self.label_5.setText(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:600;\">使用说明</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">1、点击下拉列表进行模式选择</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">2、经CNN网络计算后输出，显示识别结果与Softmax值</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">3、点击“清除数据”按键重新输入数据</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">模式1：随机从测试集抽取图像作为待识别数据</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">模式2：使用鼠标在数据输入区域手写输入作为待识别数据</span></p>\n"
"<p style=\" margin-top:12px; margin-bottom:12px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">模式3：输入自己书写的数字手写体照片作为识别数据</span></p></body></html>"))
        self.lbResult.setText(_translate("MainWindow", "9"))
        self.lbCofidence.setText(_translate("MainWindow", "0.99999999"))
        self.pbtjiazai.setText(_translate("MainWindow", "点击选择图片"))

