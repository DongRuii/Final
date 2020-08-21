from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import*
from album import Ui_MainWindow as albumpage
import sys
import pandas as pd
import numpy as np
import time
#import sip


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
dffultime = np.array(df)#np.ndarray()
timelist = dffultime.tolist()

dftime = df['timestamp']
dt = np.array(dftime)
tstamplist = dt.tolist()
tstamplist1 = dt.tolist()
tstamplist2 = dt.tolist()

dft = df['time']
dft1 = np.array(dft)
listtime = dft1.tolist()

dflabel = df['label']
dfl = np.array(dflabel)
labeL = dfl.tolist()

dftime_label = df[['timestamp','label']]
dft_label = np.array(dftime_label)
time_label = dft_label.tolist()


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

#clickable label
class myLabel(QLabel):
    clicked = pyqtSignal()
    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.clicked.emit()

class MyAlbumpage(QMainWindow, albumpage):
    def __init__(self, parent=None):
        super(MyAlbumpage, self).__init__(parent)
        self.setupUi(self)
       # self.setScale(self)
        self.comboBox_2.currentIndexChanged.connect(self.setScale)
        self.pushButton_12.clicked.connect(lambda:self.nextScale())
        self.pushButton_13.clicked.connect(lambda: self.lastScale())
        self.pushButton_15.clicked.connect(lambda: self.showSub())

        #调day/week/month那个
        self.comboBox.currentIndexChanged.connect(self.selectionchange)
        self.dateTimeEdit.setDate(QDate.currentDate())
       # sBeginDate = self.dateEdit.date().toString(Qt.ISODate)
     #   print(sBeginDate)
        self.dateTimeEdit.dateChanged.connect(self.datechange)

        self.label_7.setText('DAY')
        self.label_5.setText(self.dateTimeEdit.date().toString(Qt.ISODate))
        self.label_6.setText(self.dateTimeEdit.date().addDays(1).toString(Qt.ISODate))
        self.pushButton_11.clicked.connect(lambda: self.showchange())

        self.timel = tstamplist

        #生成图片的按钮
        '''
        r = 0
        for item in tstamplist:
            self.timel[r] = QPushButton(self.centralwidget)
          #  self.timel[r].setFixedSize(150, 10)
            self.timel[r].setText(str(listtime[r]))
            self.gridLayout_2.addWidget(self.timel[r], r, 0, 1, 1)

            self.timel[r].clicked.connect(lambda: self.ts_button_click(self.sender().text()))
            r = r + 1
        '''

      #  self.pushButton_11.clicked.connect(self.submit())
    def ts_button_click(self,timestamp):
        i = 0
        print(self.gridLayout_3.count())
        for i in range(self.gridLayout_3.count()):
            print(i)
            self.gridLayout_3.itemAt(i).widget().delete()

            #sip.delete(self.gridLayout_3.itemAt(i).widget())
            i = i + 1

        print(timestamp)
        a = str(timestamp)
        timeArray = time.strptime(a, "%Y-%m-%d %H:%M:%S")
        tStamp = int(time.mktime(timeArray))
        path = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007/' + a + '.jpg'
        path1 = '/Users/rui/PycharmProjects/test1/Final/video/Feature/' + str(tStamp) + '-0.jpg'
        print(path)
        print(path1)
        jpgtest = QtGui.QPixmap(path)
        jpgF = QtGui.QPixmap(path1)
        self.label.setPixmap(jpgtest)
        self.label_12.setPixmap(jpgF)
      #  fdata = df[df['timestamp'] == timestamp]['labelname','place1','place2','place3','place4','place5']
      #  ft_label = np.array(fdata)
      #  fulllist = ft_label.tolist()
      #  self.pushButton.setText(fulllist[1])
    def showchange(self):
        ptime = self.comboBox.currentIndex()
        if ptime == 0:
            pt = self.dateTimeEdit.date().addDays(1)
            ptt = self.dateTimeEdit.dateTime().addDays(1)
        elif ptime == 1:
            pt = self.dateTimeEdit.date().addDays(7)
            ptt = self.dateTimeEdit.dateTime().addDays(7)
        elif ptime == 2:
            pt = self.dateTimeEdit.date().addMonths(1)
            ptt = self.dateTimeEdit.dateTime().addMonths(1)
        elif ptime == 3:
            pt = self.dateTimeEdit.date().addYears(1)
            ptt = self.dateTimeEdit.dateTime().addYears(1)
        timeEdit = self.dateTimeEdit.date().toString(Qt.ISODate)
        time1 = self.dateTimeEdit.dateTime().toString(Qt.ISODate)
        ts1 = self.dateTimeEdit.dateTime().toTime_t()
        timeStamp = int(ts1)
        timeHou = pt.toString(Qt.ISODate)
        time2 = ptt.toString(Qt.ISODate)
        ts2 = ptt.toTime_t()
        timeStamp2 = int(ts2)
        i = timeStamp
        newlist = []
        while i in range(timeStamp, timeStamp2):
            newlist.append(i)
            ran = int((timeStamp2 - timeStamp) / 50)
            i = i + ran
        print(newlist)

        self.nl = newlist

        c = 0
        i = timeStamp
        while i in range(timeStamp, timeStamp2):
            self.nl[c] = QLabel(self.centralwidget)
            self.gridLayout_3.addWidget(self.nl[c], c, 0, 1, 1)

            ran = int((timeStamp2 - timeStamp) / 50)
            print(ran)
            print(i)
            b = 0
            a = i
            for a in range(i, i + 20):
                # print(a)

                if str(a) in tstamplist1:
                    timeArraya = time.localtime(a)
                    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArraya)
                    print("yes!")
                    pa = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007/' + otherStyleTime + '.jpg'
                    jpga = QtGui.QPixmap(pa)
                    self.nl[c].setPixmap(jpga)
                else:
                    b = b + 1
                a = a + 1
            if b == 20:
                print("no!")
            i = i + ran
            c = c + 1


    def datechange(self):

        self.label_7.setText(self.comboBox.currentText())
        self.label_3.setText(self.comboBox.currentText())
        ptime = self.comboBox.currentIndex()
        if ptime ==0:
            pt = self.dateTimeEdit.date().addDays(1)
            ptt = self.dateTimeEdit.dateTime().addDays(1)
        elif ptime==1:
            pt = self.dateTimeEdit.date().addDays(7)
            ptt = self.dateTimeEdit.dateTime().addDays(7)
        elif ptime ==2:
            pt = self.dateTimeEdit.date().addMonths(1)
            ptt = self.dateTimeEdit.dateTime().addMonths(1)
        elif ptime ==3:
            pt = self.dateTimeEdit.date().addYears(1)
            ptt = self.dateTimeEdit.dateTime().addYears(1)

       # self.dateEdit.setDisplayFormat("yyyy.MM.dd")
        timeEdit = self.dateTimeEdit.date().toString(Qt.ISODate)
        time1 = self.dateTimeEdit.dateTime().toString(Qt.ISODate)
     #   timeArray = time.strptime(time1, "%Y-%m-%d %H:%M:%S")
        ts1 = self.dateTimeEdit.dateTime().toTime_t()
        timeStamp = int(ts1)
       # timeStamp = int(time.mktime(timeArray))


        timeHou = pt.toString(Qt.ISODate)
        time2 = ptt.toString(Qt.ISODate)
     #   timeArray2 = time.strptime(time2, "%Y-%m-%d %H:%M:%S")
      #  timeStamp2 = int(time.mktime(timeArray2))
        ts2 = ptt.toTime_t()
        timeStamp2 = int(ts2)


        print(timeEdit)
        self.label_5.setText(timeEdit)
        self.label_6.setText(timeHou)

        '''
        while i in range(timeStamp, timeStamp2):
            for item in time_label:
                if i in range(int(item[0]),int(item[0])+10):
                    print("yes!")
                    i = i + 10
        '''
        print(tstamplist1)



        #sBeginDate = self.dateEdit.date().toString(Qt.ISODate)
       # print(str(date))


    def selectionchange(self, i):

        # 标签用来显示选中的文本
        # currentText()：返回选中选项的文本
        #self.btn1.setText(self.cb.currentText())
        #print('Items in the list are:')
        # 输出选项集合中每个选项的索引与对应的内容
        # count()：返回选项集合中的数目

        for count in range(self.comboBox.count()):
            #print('Item' + str(count) + '=' + self.cb.itemText(count))
            print('current index', i, 'selection changed', self.comboBox.currentText())

            self.label_7.setText(self.comboBox.currentText())

        if i ==0:
            pt = self.dateTimeEdit.date().addDays(1)
            ptt = self.dateTimeEdit.dateTime().addDays(1)
        elif i==1:
            pt = self.dateTimeEdit.date().addDays(7)
            ptt = self.dateTimeEdit.dateTime().addDays(7)
        elif i ==2:
            pt = self.dateTimeEdit.date().addMonths(1)
            ptt = self.dateTimeEdit.dateTime().addMonths(1)
        elif i ==3:
            pt = self.dateTimeEdit.date().addYears(1)
            ptt = self.dateTimeEdit.dateTime().addYears(1)
        timeEdit = self.dateTimeEdit.date().toString(Qt.ISODate)
        time1 = self.dateTimeEdit.dateTime().toString(Qt.ISODate)
        ts1 = self.dateTimeEdit.dateTime().toTime_t()
      #  timeArray = time.strptime(time1, "%Y-%m-%d %H:%M:%S")
    #    timeStamp = int(time.mktime(timeArray))
        timeStamp = int(ts1)

        timeHou = pt.toString(Qt.ISODate)
        time2 = ptt.toString(Qt.ISODate)
      #  timeArray2 = time.strptime(time2, "%Y-%m-%d %H:%M:%S")
        ts2 = ptt.toTime_t()
        timeStamp2 =int(ts2)
      #  timeStamp2 = int(time.mktime(timeArray2))


        '''
        for i in range(timeStamp, timeStamp2):
            for item in time_label:
                if i ==int(item[0]):
                    print("yes!")
                #else:
                   # print("no...")
        '''



       # print(timeEdit)
        self.label_5.setText(timeEdit)
        self.label_6.setText(timeHou)
    def setScale(self):
        print("set scale")
        scalenum = self.comboBox_2.currentIndex()
        self.label_9.setText("1")
        if scalenum == 0:
            self.label_27.setText("12")
            self.pushButton_14.setGeometry(QtCore.QRect(200,150,81,71))

        elif scalenum == 1:
            self.label_27.setText("8")
            self.pushButton_14.setGeometry(QtCore.QRect(200,150,81,106))
        elif scalenum == 2:
            self.label_27.setText("4")
            self.pushButton_14.setGeometry(QtCore.QRect(200,150,81,211))
        elif scalenum == 3:
            self.label_27.setText("2")
            self.pushButton_14.setGeometry(QtCore.QRect(200,150,81,421))
    def nextScale(self):
        print("a!")
        d = 0
        scalenum = self.comboBox_2.currentIndex()
        scaleindex = int(scalenum)
        if scalenum == 0:
            d = 12
        elif scalenum == 1:
            d = 8
        elif scalenum == 2:
            d = 4
        elif scalenum ==3:
            d =2

        buttonsize = int(841 / d) * 1.02
        s1 = self.label_9.text()
        s2 = self.label_27.text()
        is1 = int(s1)
        is2 = int(s2)

        if is1 < is2:
            is1 = is1 + 1
            self.label_9.setText(str(is1))
            self.pushButton_14.setGeometry(QtCore.QRect(200, buttonsize*(is1-1) + 150, 81, buttonsize))
        else:
            is1 =1
            self.label_9.setText(str(is1))
            #self.pushButton_14.setGeometry(QtCore.QRect(200, 150, 81, buttonsize))
            self.pushButton_14.setGeometry(QtCore.QRect(200, buttonsize * (is1 - 1) + 150, 81, buttonsize))

    def lastScale(self):
        print("b!")
        scalenum = self.comboBox_2.currentIndex()
        scaleindex = int(scalenum)
        if scalenum == 0:
            d = 12
        elif scalenum == 1:
            d = 8
        elif scalenum == 2:
            d = 4
        elif scalenum == 3:
            d = 2

        buttonsize = int(841 / d) * 1.02
        s1 = self.label_9.text()
        s2 = self.label_27.text()
        is1 = int(s1)
        is2 = int(s2)

        if is1 != 1:
            is1 = is1 - 1
            self.label_9.setText(str(is1))
            self.pushButton_14.setGeometry(QtCore.QRect(200, buttonsize * (is1 - 1) + 150, 81, buttonsize))
        else:
            is1 = d
            self.label_9.setText(str(is1))
            self.pushButton_14.setGeometry(QtCore.QRect(200, buttonsize*(is1-1) + 150, 81, buttonsize))
    def showSub(self):
        s1 = self.label_9.text()
        s2 = self.label_27.text()
        is1 = int(s1)
        is2 = int(s2)
        ptime = self.comboBox.currentIndex()
        if ptime == 0:
            pt = self.dateTimeEdit.date().addDays(1)
            ptt = self.dateTimeEdit.dateTime().addDays(1)
        elif ptime == 1:
            pt = self.dateTimeEdit.date().addDays(7)
            ptt = self.dateTimeEdit.dateTime().addDays(7)
        elif ptime == 2:
            pt = self.dateTimeEdit.date().addMonths(1)
            ptt = self.dateTimeEdit.dateTime().addMonths(1)
        elif ptime == 3:
            pt = self.dateTimeEdit.date().addYears(1)
            ptt = self.dateTimeEdit.dateTime().addYears(1)
        timeEdit = self.dateTimeEdit.date().toString(Qt.ISODate)
        time1 = self.dateTimeEdit.dateTime().toString(Qt.ISODate)
        ts1 = self.dateTimeEdit.dateTime().toTime_t()
        timeStamp = int(ts1)
        timeHou = pt.toString(Qt.ISODate)
        time2 = ptt.toString(Qt.ISODate)
        ts2 = ptt.toTime_t()
        timeStamp2 = int(ts2)
        abit = int((timeStamp2 - timeStamp)/is2)


        fullbit = timeStamp+(is1-1)*abit
        l = int(abit/4)
        l2 = fullbit + l
        l3 = l2 + l
        l4 = l3 + l
        l5 = l4 + l

        timeArray1 = time.localtime(fullbit)
        otherStyleTime1 = time.strftime("%Y-%m-%d %H:%M:%S", timeArray1)
        timeArray2 = time.localtime(l2)
        otherStyleTime2 = time.strftime("%Y-%m-%d %H:%M:%S", timeArray2)
        timeArray3 = time.localtime(l3)
        otherStyleTime3 = time.strftime("%Y-%m-%d %H:%M:%S", timeArray3)
        timeArray4 = time.localtime(l4)
        otherStyleTime4 = time.strftime("%Y-%m-%d %H:%M:%S", timeArray4)
        timeArray5 = time.localtime(l5)
        otherStyleTime5 = time.strftime("%Y-%m-%d %H:%M:%S", timeArray5)
        self.label_30.setText(otherStyleTime1)
        self.label_29.setText(otherStyleTime2)
        self.label_28.setText(otherStyleTime3)
        self.label_25.setText(otherStyleTime4)
        self.label_24.setText(otherStyleTime5)
        self.timel = tstamplist

        showlist = []
        i = fullbit
        while i in range(fullbit, l5):
            showlist.append(i)
            ran = int((l5 - fullbit) / 50)
            i = i + ran
        print(showlist)

        self.sl = showlist
        c = 0
        i = fullbit
        while i in range(fullbit, l5):
            self.sl[c] = myLabel(self.centralwidget)

            self.gridLayout_2.addWidget(self.sl[c], c, 0, 1, 1)
           # self.sl[c].clicked.connect(lambda:)

            ran = int((l5 - fullbit) / 50)
            print(ran)
            print(i)
            b = 0
            a = i
            for a in range(i, i + 20):
                # print(a)

                if str(a) in tstamplist1:
                    timeArraya = time.localtime(a)
                    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArraya)
                    print("yes!")
                    pa = '/Users/rui/PycharmProjects/test1/Final/video/TLC00007/' + otherStyleTime + '.jpg'
                    jpga = QtGui.QPixmap(pa)
                    self.sl[c].setPixmap(jpga)
                else:
                    b = b + 1
                a = a + 1
            if b == 20:
                print("no!")
            i = i + ran
            c = c + 1


        rr = 0
        butlist1 = []
        butlist2 = []
        butlist3 = []
        butlist4 = []
        for item in tstamplist2:
            intitem = int(item)
            print(intitem)
            print(fullbit)
            if (intitem >= fullbit) and (intitem < l2):
                a1 = str(labeL[rr])
                if a1 not in butlist1:
                    butlist1.append(a1)
            elif intitem>= l2 and intitem < l3:
                a2 = str(labeL[rr])
                if a2 not in butlist2:
                    butlist2.append(a2)
            elif intitem>= l3 and intitem < l4:
                a3 = str(labeL[rr])
                if a3 not in butlist3:
                    butlist3.append(a3)
            elif intitem>= l4 and intitem <= l5:
                a4 = str(labeL[rr])
                if a4 not in butlist4:
                    butlist4.append(a4)
            rr = rr+1
        print(butlist1)
        print(butlist2)
        print(butlist3)
        print(butlist4)
        self.sbuttonlist1 = butlist1
        self.sbuttonlist2 = butlist2
        self.sbuttonlist3 = butlist3
        self.sbuttonlist4 = butlist4
        i = 0
        for item in butlist1:
            self.sbuttonlist1[i] = QPushButton(self.frame)
            self.sbuttonlist1[i].setText(str(item))
            self.verticalLayout_6.addWidget(self.sbuttonlist1[i])
            i = i+1
        i = 0
        for item in butlist2:
            self.sbuttonlist2[i] = QPushButton(self.frame)
            self.sbuttonlist2[i].setText(str(item))
            self.verticalLayout_7.addWidget(self.sbuttonlist2[i])
            i = i+1
        i = 0
        for item in butlist3:
            self.sbuttonlist3[i] = QPushButton(self.frame)
            self.sbuttonlist3[i].setText(str(item))
            self.verticalLayout_8.addWidget(self.sbuttonlist3[i])
            i = i+1
        i = 0
        for item in butlist4:
            self.sbuttonlist4[i] = QPushButton(self.frame)
            self.sbuttonlist4[i].setText(str(item))
            self.verticalLayout_5.addWidget(self.sbuttonlist4[i])
            i = i+1










if __name__ == "__main__":
    #固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    #初始化
    myWin = MyAlbumpage()
    #将窗口控件显示在屏幕上
    myWin.show()

    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())
