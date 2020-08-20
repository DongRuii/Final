from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
from labelpage import Ui_MainWindow as labelpage
import pandas as pd
import sys
import numpy as np

#read label and state into the dataframe
isstateList = []
istimeList = []
f = open('/Users/rui/desktop/label.csv','r')
ff = open('/Users/rui/desktop/timeplace.csv','r')
res = f.readlines()[1:]
ress = ff.readlines()[1:]
for i in res:
    i_str = i.replace('\n','')
    ilist = i_str.split(',')
    isstateList.append(ilist)

for i in ress:
    i_str = i.replace('\n','')
    ilistt = i_str.split(',')
    istimeList.append(ilistt)
df = pd.DataFrame(istimeList)
df.columns = ['timestamp', 'time','label', 'score1', 'place1', 'score2', 'place2', 'score3', 'place3','score4','place4','score5','place5']
dftime = df['time']
dflabel = df['label']
dftime_label = df[['time','label']]

dft_label = np.array(dftime_label)
dfl = np.array(dflabel)
labeL = dfl.tolist()
time_label = dft_label.tolist()
dffultime = np.array(df)#np.ndarray()
timelist = dffultime.tolist()
dftimelist = dftime.tolist()
labeltimelist = dftime.tolist()


df1 = pd.DataFrame(isstateList)
df1.columns = ['labelnumber', 'labelname','statenumber']
df1['labelnumber'] = df1['labelnumber'].astype('int')
#df1 = df1.set_index('labelnumber')
#df1 = df1.sort_index(ascending= True)

#output star label find_data is series findlist is list
find_data = df1[df1['statenumber']=='2']['labelname']
find_num = df1[df1['statenumber']=='2']['labelnumber']
find_nordata = df1[df1['statenumber']!='0']['labelname']
finda_nornum = df1[df1['statenumber']!='0']['statenumber']


ElseList=[]
findlist = find_data.tolist()
findnum = find_num.tolist()
norlist = find_nordata.tolist()
numnorlist = finda_nornum.tolist()


class Mylabelpage(QMainWindow, labelpage):
    def __init__(self, parent=None):
        fornum = 0

        #varlist star label list
        self.varList = findnum
        #norList first letter label

        self.norList = norlist
        self.jpgList = labeltimelist
        super(Mylabelpage, self).__init__(parent)
        self.setupUi(self)
        self.scrollArea_2.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scrollArea_2.setWidgetResizable(True)
       # self.left_frame = QFrame(self)
       # self.left_frame.setFixedSize(1421, 831)

        self.pushButton.clicked.connect(lambda: self.manage_button_click())


    #    self.gridLayout.maximumSize()
       # self.gridLayout.addWidget(self)
       # labelpage.scrollArea_3.setWidget(labelpage.scrollAreaWidgetContents_3)
        #star label

        #star label show
        for item in findlist:
            colnum = fornum % 4
            rownum = int(fornum * 0.25)
            print(colnum)
          #  print(rownum)
            self.varList[fornum] = QPushButton(self.frame)
            self.varList[fornum].setText(str(item))
        #    self.varList[fornum].setObjectName(str(item))
            self.varList[fornum].setFixedSize(300, 41)
            self.varList[fornum].setGeometry(QtCore.QRect(colnum * 310 + 10, rownum * 50 + 10, 300, 41))
         #   self.gridLayout.addWidget(self.varList[fornum],rownum,colnum,1,1)
            '''
            if (fornum < 4):
                self.varList[fornum].setGeometry(QtCore.QRect(310*fornum+60, 180, 301, 41))
                
            elif ((fornum >= 4) and (fornum < 8)):
                self.varList[fornum].setGeometry(QtCore.QRect(310*fornum-1180, 230, 301, 41))
                
            '''
            self.varList[fornum].clicked.connect(lambda: self.nor_button_click(self.sender().text()))
            font = QtGui.QFont()
            font.setPointSize(14)

            self.varList[fornum].setFont(font)
            self.varList[fornum].setStyleSheet("QPushButton{\n"
                                            "   border-radius: 20px;  \n"
                                            "   border: 6px groove rgb(255, 204, 0);\n"
                                            "   \n"
                                            "}")
           # self.gridLayout.addWidget(self.varList[fornum],int(0.25*fornum),fornum)



            fornum = fornum + 1

        #first letter label

        nA = 0
        nB = 0
        nC = 0
        nD = 0
        nE = 0
        nF = 0
        nG = 0
        nH = 0
        nI = 0
        nJ = 0
        nK = 0
        nL = 0
        nM = 0
        nN = 0
        nO = 0
        nP = 0
        nQ = 0
        nR = 0
        nS = 0
        nT = 0
        nU = 0
        nV = 0
        nW = 0
        nX = 0
        nY = 0
        nZ = 0

        #normal label show
        fornum = 0
        for item in norlist:
            self.norList[fornum] = QPushButton(self.frame)
            self.norList[fornum].setFixedSize(270, 30)
            self.norList[fornum].setText(str(item))
        #    self.norList[fornum].setObjectName(str(item+'_nor'))
            self.norList[fornum].setStyleSheet("QPushButton{\n"
                                               "   border-radius: 10px;  \n"
                                               "   border: 2px groove grey;\n"
                                               "   \n"
                                               "}")

            if numnorlist[fornum]=='2':
                self.norList[fornum].setStyleSheet("QPushButton{\n"
                                                   "   border-radius: 10px;  \n"
                                                   "   border: 6px groove rgb(255, 204, 0);\n"
                                                   "   \n"
                                                   "}")
          #  self.norList[fornum].clicked.connect(lambda:self.nor_button_click(self.sender().text()))
         #   self.norList[fornum].clicked.connect(lambda:self.nor_button_click())



            if str(item)[0]=='a':
                rn = int(nA*0.25)
                cn = nA % 4
               # self.norList[fornum] = QPushButton(self.tab)
               # self.norList[fornum].setText(str(item))
              #  self.norList[fornum].setObjectName(str(item))
                #self.norList[fornum].setGeometry(QtCore.QRect(20+300*fornum, 20, 271, 41))
                self.gridLayout_2.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nA = nA+1
            elif str(item)[0]=='b':
                rn = int(nB * 0.25)
                cn = nB % 4
                self.gridLayout_3.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nB = nB + 1
            elif str(item)[0]=='c':
                rn = int(nC * 0.25)
                cn = nC % 4
                self.gridLayout_4.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nC = nC + 1
            elif str(item)[0] == 'd':
                rn = int(nD * 0.25)
                cn = nD % 4
                self.gridLayout_5.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nD = nD + 1
            elif str(item)[0] == 'e':
                rn = int(nE * 0.25)
                cn = nE % 4
                self.gridLayout_6.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nE = nE + 1
            elif str(item)[0] == 'f':
                rn = int(nF * 0.25)
                cn = nF % 4
                self.gridLayout_7.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nF = nF + 1
            elif str(item)[0] == 'g':
                rn = int(nG * 0.25)
                cn = nG % 4
                self.gridLayout_8.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nG = nG + 1
            elif str(item)[0] == 'h':
                rn = int(nH * 0.25)
                cn = nH % 4
                self.gridLayout_9.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nH = nH + 1
            elif str(item)[0] == 'i':
                rn = int(nI * 0.25)
                cn = nI % 4
                self.gridLayout_10.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nI = nI + 1
            elif str(item)[0] == 'j':
                rn = int(nJ * 0.25)
                cn = nJ % 4
                self.gridLayout_11.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nJ = nJ + 1
            elif str(item)[0] == 'k':
                rn = int(nK * 0.25)
                cn = nK % 4
                self.gridLayout_12.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nK = nK + 1
            elif str(item)[0] == 'l':
                rn = int(nL * 0.25)
                cn = nL % 4
                self.gridLayout_13.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nL = nL + 1
            elif str(item)[0] == 'm':
                rn = int(nM * 0.25)
                cn = nM % 4
                self.gridLayout_14.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nM = nM + 1
            elif str(item)[0] == 'n':
                rn = int(nN * 0.25)
                cn = nN % 4
                self.gridLayout_17.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nN = nN + 1
            elif str(item)[0] == 'o':
                rn = int(nO * 0.25)
                cn = nO % 4
                self.gridLayout_18.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nO = nO + 1
            elif str(item)[0] == 'p':
                rn = int(nP * 0.25)
                cn = nP % 4
                self.gridLayout_19.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nP = nP + 1
            elif str(item)[0] == 'q':
                rn = int(nQ * 0.25)
                cn = nQ % 4
                self.gridLayout_20.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nQ = nQ + 1
            elif str(item)[0] == 'r':
                rn = int(nR * 0.25)
                cn = nR % 4
                self.gridLayout_21.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nR = nR + 1
            elif str(item)[0] == 's':
                rn = int(nS * 0.25)
                cn = nS % 4
                self.gridLayout_22.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nS = nS + 1
            elif str(item)[0] == 't':
                rn = int(nT * 0.25)
                cn = nT % 4
                self.gridLayout_23.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nT = nT + 1
            elif str(item)[0] == 'u':
                rn = int(nU * 0.25)
                cn = nU % 4
                self.gridLayout_24.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nU = nU + 1
            elif str(item)[0] == 'v':
                rn = int(nV * 0.25)
                cn = nV % 4
                self.gridLayout_25.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nV = nV + 1
            elif str(item)[0] == 'w':
                rn = int(nW * 0.25)
                cn = nW % 4
                self.gridLayout_26.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nW = nW + 1
            elif str(item)[0] == 'x':
                rn = int(nX * 0.25)
                cn = nX % 4
                self.gridLayout_27.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nX = nX + 1
            elif str(item)[0] == 'y':
                rn = int(nY * 0.25)
                cn = nY % 4
                self.gridLayout_28.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nY = nY + 1
            elif str(item)[0] == 'z':
                rn = int(nZ * 0.25)
                cn = nZ % 4
                self.gridLayout_29.addWidget(self.norList[fornum], rn, cn, 1, 1)
                nZ = nZ + 1
            else:
                print('others')
            self.norList[fornum].clicked.connect(lambda: self.nor_button_click(self.sender().text()))
            fornum = fornum + 1

        '''

        a = 0
        for item in dftimelist:
            path = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007' + str(item) + '.jpg'
            print(path)

            jpg = QtGui.QPixmap(path)
          #self.label_pic[a] = QLabel()
            #self.label_pic[a].setPixmap(QPixmap(jpg))
            self.jpgList[a] = QtWidgets.QLabel(self.centralwidget)

            self.jpgList[a].setPixmap(jpg)
            self.gridLayout.addWidget(self.jpgList[a], 0, a, 1, 1)
            a = a + 1
        '''

        # elif (item[0]=='b'):
    def nor_button_click(self,name):
        print('clicked!')
        print(name)
        self.stackedWidget.setCurrentIndex(1)
        self.label_3.setText(str(name))
        self.label_4.setText(str(dftimelist[0]))
        l = int(len(dftimelist))-1

        self.label_5.setText(str(dftimelist[l]))
        a = 0
        path1 = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007/2020-08-02 20:17:12.jpg'
        jpgtest = QtGui.QPixmap(path1)
        self.label_6.setPixmap(jpgtest)
        for i in range(0,len(dftimelist)):
            self.jpgList[a] = QtWidgets.QLabel(self.frame)
            self.jpgList[a].setFixedSize(10,10)
            self.gridLayout.addWidget(self.jpgList[a], 0, a, 1, 1)
           # self.jpgList[a].setGeometry(QtCore.QRect(a * 10+10,10, 30, 30))

            if labeL[i] == str(name):
                path = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007/'+str(labeL[i]) + '.jpg'
                jpg = QtGui.QPixmap(path)
                print(path)
                self.jpgList[a].setPixmap(jpg)
                path2 = '/Users/rui/PycharmProjects/test1/Final/timeset/' + str(labeL[i])+'/'
                print("image")
            else:
                self.jpgList[a].setText("test")
                print("text")

            #path = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007'+str(item) + '.jpg'
           # print(path)






          #  self.gridLayout.addWidget(self.jpgList[a], 0, a, 1, 1)
           # self.jpgList[a].setText("test")
          #  self.jpgList[a].setPixmap(jpg)
            a = a + 1



    def manage_button_click(self):
        print('clicked!')

        self.stackedWidget.setCurrentIndex(1)







if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = Mylabelpage()
    #将窗口控件显示在屏幕上
    myWin.show()

    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
